"""Original PDFs in S3/R2 and GET /chats/{chat_id}/files/{document_id} (presigned link)."""

import unicodedata
import uuid
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import boto3
import pymupdf
import pytest
from botocore.config import Config
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from app.api.deps import get_current_user
from app.api.routes import chats as chats_route
from app.db.session import get_session
from app.main import app
from app.services import document_service, storage
from app.services.storage import (
    FileStorage,
    S3FileStorage,
    StorageError,
    StorageNotConfigured,
    document_file_key,
    inline_disposition,
    split_endpoint,
)

USER_ID = uuid.uuid4()
CHAT_ID = uuid.uuid4()
DOC_ID = uuid.uuid4()
R2 = "https://acc123.r2.cloudflarestorage.com"


def _pdf(text: str = "Hello PDF") -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


class MemoryStorage(FileStorage):
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}

    async def save(self, key, data, *, filename, content_type):
        self.objects[key] = {"data": data, "filename": filename, "type": content_type}

    async def delete(self, key):
        self.objects.pop(key, None)

    async def exists(self, key):
        return key in self.objects

    async def presigned_url(self, key, expires_in):
        return f"https://r2.example/{key}?expires={expires_in}"


# ---- configuration ---------------------------------------------------------------


def test_bucket_is_taken_from_the_endpoint_path():
    assert split_endpoint(f"{R2}/coder-llm", None) == (R2, "coder-llm")
    assert split_endpoint(f"{R2}/coder-llm/", None) == (R2, "coder-llm")


def test_explicit_bucket_wins():
    assert split_endpoint(f"{R2}/coder-llm", "other") == (R2, "other")


def test_endpoint_without_bucket_is_an_error():
    with pytest.raises(StorageNotConfigured):
        split_endpoint(R2, None)


def test_missing_credentials_are_reported(monkeypatch):
    monkeypatch.setattr(storage, "_storage", None)
    monkeypatch.setattr(storage.settings, "s3_endpoint", f"{R2}/coder-llm")
    monkeypatch.setattr(storage.settings, "s3_access_key_id", None)
    monkeypatch.setattr(storage.settings, "s3_secret_access_key", None)

    with pytest.raises(StorageNotConfigured, match="S3_ACCESS_KEY_ID"):
        storage.get_file_storage()


def test_inline_disposition_keeps_cyrillic_names():
    value = inline_disposition("Резюме Андас.pdf")
    assert value.startswith('inline; filename="document.pdf"')  # ASCII fallback
    assert "filename*=UTF-8''%D0%A0%D0%B5%D0%B7" in value
    assert inline_disposition("cv 2025.pdf").startswith(
        'inline; filename="cv 2025.pdf"'
    )


# ---- S3FileStorage against a stubbed boto3 client (no network) --------------------


@pytest.fixture
def s3():
    client = boto3.client(
        "s3",
        endpoint_url=R2,
        aws_access_key_id="AKIDTEST",
        aws_secret_access_key="secret",
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    with Stubber(client) as stub:
        yield SimpleNamespace(stub=stub, storage=S3FileStorage(client, "coder-llm"))
        stub.assert_no_pending_responses()


async def test_save_puts_the_object_with_inline_headers(s3):
    key = document_file_key(DOC_ID)
    s3.stub.add_response(
        "put_object",
        {},
        {
            "Bucket": "coder-llm",
            "Key": key,
            "Body": b"%PDF",
            "ContentType": "application/pdf",
            "ContentDisposition": inline_disposition("cv.pdf"),
        },
    )
    await s3.storage.save(
        key, b"%PDF", filename="cv.pdf", content_type="application/pdf"
    )


async def test_exists_and_delete(s3):
    key = document_file_key(DOC_ID)
    s3.stub.add_response("head_object", {}, {"Bucket": "coder-llm", "Key": key})
    s3.stub.add_client_error("head_object", "404", http_status_code=404)
    s3.stub.add_response("delete_object", {}, {"Bucket": "coder-llm", "Key": key})

    assert await s3.storage.exists(key) is True
    assert await s3.storage.exists(key) is False
    await s3.storage.delete(key)


async def test_presigned_url_points_to_the_object(s3):
    url = await s3.storage.presigned_url(document_file_key(DOC_ID), 600)

    parts = urlsplit(url)
    query = parse_qs(parts.query)
    assert parts.netloc in (
        "acc123.r2.cloudflarestorage.com",
        "coder-llm.acc123.r2.cloudflarestorage.com",
    )
    assert f"documents/{DOC_ID}.pdf" in parts.path
    assert query["X-Amz-Expires"] == ["600"]
    assert "X-Amz-Signature" in query
    assert "secret" not in url  # the secret key never leaves the server


# ---- the endpoint ----------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    """App with auth/DB replaced by fakes and an in-memory object storage."""
    mem = MemoryStorage()
    docs = {}  # (document_id, chat_id, user_id) -> document

    async def get_chat_document(session, document_id, chat_id, user_id):
        return docs.get((document_id, chat_id, user_id))

    async def no_session():
        yield None

    monkeypatch.setattr(chats_route, "get_file_storage", lambda: mem)
    monkeypatch.setattr(
        chats_route.document_repo, "get_chat_document", get_chat_document
    )
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=USER_ID)
    app.dependency_overrides[get_session] = no_session
    yield SimpleNamespace(http=TestClient(app), storage=mem, docs=docs)
    app.dependency_overrides.clear()


def _add(client, filename="cv.pdf", store=True):
    client.docs[(DOC_ID, CHAT_ID, USER_ID)] = SimpleNamespace(
        id=DOC_ID, filename=filename
    )
    if store:
        client.storage.objects[document_file_key(DOC_ID)] = {"data": b"%PDF"}


def _url(chat_id=CHAT_ID, doc_id=DOC_ID):
    return f"/chats/{chat_id}/files/{doc_id}"


def test_returns_a_presigned_link(client, monkeypatch):
    monkeypatch.setattr(chats_route.settings, "file_url_ttl_s", 900)
    _add(client, filename="Резюме.pdf")

    r = client.http.get(_url())

    assert r.status_code == 200
    assert r.json() == {
        "url": f"https://r2.example/documents/{DOC_ID}.pdf?expires=900",
        "filename": "Резюме.pdf",
        "expires_in": 900,
    }


def test_other_chat_or_document_gets_404(client):
    _add(client)

    assert client.http.get(_url(chat_id=uuid.uuid4())).status_code == 404
    assert client.http.get(_url(doc_id=uuid.uuid4())).status_code == 404


def test_document_without_stored_original_is_404(client):
    _add(client, store=False)

    r = client.http.get(_url())

    assert r.status_code == 404
    assert "not stored" in r.json()["detail"]


def test_storage_not_configured_is_503(client, monkeypatch):
    _add(client)

    def not_configured():
        raise StorageNotConfigured("Missing settings: S3_ACCESS_KEY_ID")

    monkeypatch.setattr(chats_route, "get_file_storage", not_configured)

    r = client.http.get(_url())

    assert r.status_code == 503
    assert "S3_ACCESS_KEY_ID" in r.json()["detail"]


def test_requires_auth():
    assert TestClient(app).get(_url()).status_code == 401


# ---- ingest uploads the original -------------------------------------------------


@pytest.fixture
def ingest_env(monkeypatch):
    env = SimpleNamespace(storage=MemoryStorage(), created=[], deleted=[])

    async def create_document(session, user_id, chat_id, filename, pages, total):
        env.created.append(filename)
        return SimpleNamespace(id=DOC_ID, filename=filename)

    async def delete_document(session, document_id, user_id):
        env.deleted.append(document_id)
        return True

    async def no_index(*args):  # skip background embedding
        return None

    monkeypatch.setattr(document_service, "get_file_storage", lambda: env.storage)
    monkeypatch.setattr(document_service.repo, "create_document", create_document)
    monkeypatch.setattr(document_service.repo, "delete_document", delete_document)
    monkeypatch.setattr(document_service, "_index", no_index)
    return env


async def test_ingest_uploads_the_original_pdf(ingest_env):
    data = _pdf("Some text to index")
    nfd_name = unicodedata.normalize("NFD", "СПИСОК_ОБЛАДАТЕЛЕЙ.pdf")
    service = document_service.DocumentService(None, embeddings=None)

    await service.ingest(USER_ID, CHAT_ID, nfd_name, data)

    nfc_name = unicodedata.normalize("NFC", nfd_name)
    assert ingest_env.storage.objects[document_file_key(DOC_ID)] == {
        "data": data,
        "filename": nfc_name,  # the decomposed upload name is stored composed
        "type": "application/pdf",
    }
    assert ingest_env.created == [nfc_name]


async def test_failed_upload_removes_the_document(ingest_env, monkeypatch):
    async def broken_save(key, data, *, filename, content_type):
        raise ConnectionError("R2 unreachable")

    monkeypatch.setattr(ingest_env.storage, "save", broken_save)
    service = document_service.DocumentService(None, embeddings=None)

    with pytest.raises(ConnectionError):
        await service.ingest(USER_ID, CHAT_ID, "cv.pdf", _pdf("text"))

    assert ingest_env.deleted == [DOC_ID]


# ---- storage failures (e.g. an R2 token without access to the bucket) -------------


async def test_access_denied_becomes_storage_error(s3):
    s3.stub.add_client_error(
        "put_object", "AccessDenied", "Access Denied", http_status_code=403
    )

    with pytest.raises(StorageError) as err:
        await s3.storage.save(
            "k", b"%PDF", filename="cv.pdf", content_type="application/pdf"
        )

    assert err.value.code == "AccessDenied"
    assert err.value.operation == "put_object"


async def test_exists_raises_on_forbidden_instead_of_saying_missing(s3):
    s3.stub.add_client_error("head_object", "403", http_status_code=403)

    with pytest.raises(StorageError):
        await s3.storage.exists("k")


def test_storage_failure_on_view_is_502(client, monkeypatch):
    _add(client)

    async def denied(key):
        raise StorageError("head_object", "403")

    monkeypatch.setattr(client.storage, "exists", denied)

    r = client.http.get(_url())

    assert r.status_code == 502
    assert "head_object failed: 403" in r.json()["detail"]
