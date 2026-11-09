from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import MessagePayloadDep

app = FastAPI()


@app.post("/echo")
async def echo(payload: MessagePayloadDep):
    return {
        "content": payload.content,
        "files": [(n, len(b)) for n, b in payload.files],
    }


client = TestClient(app)
PDF = "application/pdf"


def test_json_body():
    r = client.post("/echo", json={"content": "hello"})
    assert r.status_code == 200
    assert r.json() == {"content": "hello", "files": []}


def test_multipart_with_text_and_files():
    r = client.post(
        "/echo",
        data={"content": "check these"},
        files=[
            ("files", ("a.pdf", b"%PDF-a", PDF)),
            ("files", ("b.pdf", b"%PDF-bb", PDF)),
        ],
    )
    assert r.status_code == 200
    assert r.json() == {"content": "check these", "files": [["a.pdf", 6], ["b.pdf", 7]]}


def test_multipart_without_files_is_just_text():
    r = client.post("/echo", data={"content": "hi"}, files={"x": (None, "y")})
    assert r.status_code == 200
    assert r.json()["files"] == []


def test_non_pdf_is_415():
    r = client.post(
        "/echo",
        data={"content": "x"},
        files=[("files", ("a.txt", b"hello", "text/plain"))],
    )
    assert r.status_code == 415


def test_too_many_files_is_400():
    files = [("files", (f"{i}.pdf", b"%PDF", PDF)) for i in range(6)]
    r = client.post("/echo", data={"content": "x"}, files=files)
    assert r.status_code == 400


def test_empty_or_missing_content_is_422():
    assert client.post("/echo", json={"content": ""}).status_code == 422
    assert client.post("/echo", json={}).status_code == 422
    assert (
        client.post(
            "/echo", data={"content": ""}, files=[("files", ("a.pdf", b"x", PDF))]
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/echo", content=b"not json", headers={"content-type": "application/json"}
        ).status_code
        == 422
    )
