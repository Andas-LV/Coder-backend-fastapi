"""Storage for original uploaded files (the PDFs users attach to messages).

Postgres keeps only extracted text and vectors; the original bytes live in S3-compatible
object storage (Cloudflare R2). The frontend never gets the bucket credentials: it asks
the API for a short-lived presigned URL and opens the PDF from R2 directly.
"""

import asyncio
import logging
import re
import uuid
from abc import ABC, abstractmethod
from urllib.parse import quote, urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageNotConfigured(RuntimeError):
    """S3_* settings are missing, so files can be neither stored nor served."""


class StorageError(RuntimeError):
    """The object storage refused or failed a request (bad permissions, network...)."""

    def __init__(self, operation: str, code: str, message: str = "") -> None:
        self.operation = operation
        self.code = code
        super().__init__(
            f"{operation} failed: {code}{f' ({message})' if message else ''}"
        )


class FileStorage(ABC):
    @abstractmethod
    async def save(
        self, key: str, data: bytes, *, filename: str, content_type: str
    ) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove the object; a missing object is not an error."""

    @abstractmethod
    async def exists(self, key: str) -> bool: ...

    @abstractmethod
    async def presigned_url(self, key: str, expires_in: int) -> str:
        """Temporary GET link to the object (no credentials needed to open it)."""


def inline_disposition(filename: str) -> str:
    """Content-Disposition that shows the file in the browser, with a UTF-8 name.

    The plain `filename` is an ASCII fallback; `filename*` (RFC 6266/5987) carries the
    real name, e.g. Cyrillic.
    """
    ascii_name = filename.encode("ascii", "ignore").decode().replace('"', "").strip()
    if not re.search(r"[A-Za-z0-9]", ascii_name.rsplit(".", 1)[0]):
        ascii_name = "document.pdf"  # e.g. a fully Cyrillic name leaves only ".pdf"
    return f"inline; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


class S3FileStorage(FileStorage):
    def __init__(self, client, bucket: str) -> None:
        self._client = client
        self.bucket = bucket

    async def _call(self, operation: str, **params):
        """Run a boto3 call off the event loop, turning its errors into StorageError."""
        try:
            return await asyncio.to_thread(
                getattr(self._client, operation), Bucket=self.bucket, **params
            )
        except ClientError as e:
            err = e.response.get("Error", {})
            code = str(
                err.get("Code") or e.response["ResponseMetadata"]["HTTPStatusCode"]
            )
            logger.warning("S3 %s on bucket %r failed: %s", operation, self.bucket, e)
            raise StorageError(operation, code, err.get("Message", "")) from e
        except BotoCoreError as e:  # network, DNS, timeouts
            logger.warning("S3 %s on bucket %r failed: %s", operation, self.bucket, e)
            raise StorageError(operation, type(e).__name__) from e

    async def save(
        self, key: str, data: bytes, *, filename: str, content_type: str
    ) -> None:
        # Content-Type/Disposition are stored with the object, so a plain presigned GET
        # already opens inline in the browser with the right file name.
        await self._call(
            "put_object",
            Key=key,
            Body=data,
            ContentType=content_type,
            ContentDisposition=inline_disposition(filename),
        )

    async def delete(self, key: str) -> None:
        await self._call("delete_object", Key=key)

    async def exists(self, key: str) -> bool:
        try:
            await self._call("head_object", Key=key)
        except StorageError as e:
            if e.code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True

    async def presigned_url(self, key: str, expires_in: int) -> str:
        # signing is local (no network call)
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )


def split_endpoint(endpoint: str, bucket: str | None) -> tuple[str, str]:
    """S3_ENDPOINT may include the bucket as its path (Cloudflare shows it that way):
    https://<account>.r2.cloudflarestorage.com/<bucket> -> (endpoint, bucket).
    An explicit S3_BUCKET wins."""
    parts = urlsplit(endpoint.rstrip("/"))
    base = f"{parts.scheme}://{parts.netloc}"
    path_bucket = parts.path.strip("/").split("/")[0] if parts.path.strip("/") else ""
    resolved = bucket or path_bucket
    if not resolved:
        raise StorageNotConfigured("Set S3_BUCKET or put the bucket into S3_ENDPOINT")
    return base, resolved


def document_file_key(document_id: uuid.UUID) -> str:
    return f"documents/{document_id}.pdf"


_storage: FileStorage | None = None


def get_file_storage() -> FileStorage:
    global _storage
    if _storage is None:
        missing = [
            name
            for name, value in (
                ("S3_ENDPOINT", settings.s3_endpoint),
                ("S3_ACCESS_KEY_ID", settings.s3_access_key_id),
                ("S3_SECRET_ACCESS_KEY", settings.s3_secret_access_key),
            )
            if not value
        ]
        if missing:
            raise StorageNotConfigured(f"Missing settings: {', '.join(missing)}")
        endpoint, bucket = split_endpoint(settings.s3_endpoint, settings.s3_bucket)
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
            config=Config(signature_version="s3v4", retries={"mode": "standard"}),
        )
        _storage = S3FileStorage(client, bucket)
    return _storage


async def delete_quietly(key: str) -> None:
    """Best-effort cleanup: a storage hiccup must not mask the original error."""
    try:
        await get_file_storage().delete(key)
    except Exception:
        logger.warning("Could not delete %s from file storage", key, exc_info=True)
