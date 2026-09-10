import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


class DummyContentTypeInfo:
    group = "image"
    description = "PNG image data"
    is_text = False
    mime_type = "image/png"


class DummyOutput:
    ct_label = "png"
    mime_type = "image/png"


class DummyResult:
    score = 0.95
    output = DummyOutput()


class DummyMagika:
    def identify_bytes(self, b):
        return DummyResult()

    def get_output_content_type(self, res):
        return DummyContentTypeInfo()


def test_status_endpoint():
    main.model_ready = True
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "version" in data
    assert data["ready"] is True


def test_detect_extension_mismatch():
    main.model_ready = True
    main.magika_instance = DummyMagika()

    response = client.post(
        "/detect_extension_mismatch",
        params={"filename": "document.pdf"},
        files={"file": ("document.pdf", b"dummy png content", "image/png")},
    )
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["mismatch"] is True
    assert res_data["detected_label"] == "png"
    assert "raw_json" in res_data


def test_predict_slices():
    main.model_ready = True
    main.magika_instance = DummyMagika()

    files = [
        ("files", ("image1.png", b"H1", "image/png")),
        ("files", ("image2.png", b"H2", "image/png")),
    ]

    response = client.post("/predict_slices", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert "raw_json" in data["predictions"][0]


def test_predict_batch_slices():
    main.model_ready = True
    main.magika_instance = DummyMagika()

    data = {
        "filenames": [
            "folder/subfolder/image1.png",
            "folder/subfolder/image2.png",
        ]
    }

    files = [
        ("heads", ("image1.png", b"H1", "image/png")),
        ("heads", ("image2.png", b"H2", "image/png")),
        ("mids", ("image1.png", b"M1", "image/png")),
        ("mids", ("image2.png", b"M2", "image/png")),
        ("tails", ("image1.png", b"T1", "image/png")),
        ("tails", ("image2.png", b"T2", "image/png")),
    ]

    response = client.post("/predict_batch_slices", data=data, files=files)
    assert response.status_code == 200
    res = response.json()
    assert "results" in res
    assert len(res["results"]) == 2
    assert res["results"][0]["filename"] == "folder/subfolder/image1.png"
    assert res["results"][0]["label"] == "png"
    assert "raw_json" in res["results"][0]