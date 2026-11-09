from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB per file


async def read_pdf(file: UploadFile) -> bytes:
    """Validate an uploaded PDF (type, size) and return its bytes."""
    if file.content_type != "application/pdf":
        raise HTTPException(415, "Only application/pdf is supported")
    data = await file.read(MAX_UPLOAD_BYTES + 1)  # read one byte over the limit
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 20 MB")
    return data
