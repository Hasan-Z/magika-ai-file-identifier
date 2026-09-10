from fastapi.testclient import TestClient
from main import app

def test_status_endpoint():
    with TestClient(app) as client:
        response = client.get("/status")

        assert response.status_code == 200

        data = response.json()

        assert data["ready"] is True
        assert data["version"] == "1.0.3"

def test_detect_extension_mismatch():
    with TestClient(app) as client:
        response = client.post(
            "/detect_extension_mismatch?filename=test.txt",
            files={
                "file": (
                    "test.txt",
                    b"Hello, this is a text file.",
                    "text/plain",
                )
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["filename"] == "test.txt"
        assert data["provided_extension"] == "txt"
        assert "detected_label" in data
        assert "score" in data
        assert "mismatch" in data
        assert "raw_json" in data

        raw_json = data["raw_json"]

        assert isinstance(raw_json, dict)
        assert raw_json["status"] == "ok"
        assert "path" in raw_json
        assert "prediction" in raw_json

        prediction = raw_json["prediction"]

        assert "dl" in prediction
        assert "output" in prediction
        assert "score" in prediction
        assert "overwrite_reason" in prediction

def test_predict_slices():
    with TestClient(app) as client:
        response = client.post(
            "/predict_slices",
            files=[
                (
                    "files",
                    (
                        "test.txt",
                        b"Hello, this is a text file.",
                        "text/plain",
                    ),
                )
            ],
        )

        assert response.status_code == 200

        data = response.json()

        assert "predictions" in data
        assert len(data["predictions"]) == 1

        prediction = data["predictions"][0]

        assert prediction["filename"] == "test.txt"
        assert "label" in prediction
        assert "mime_type" in prediction
        assert "group" in prediction
        assert "description" in prediction
        assert "is_text" in prediction
        assert "score" in prediction
        assert "raw_json" in prediction

        raw_json = prediction["raw_json"]

        assert isinstance(raw_json, dict)
        assert raw_json["status"] == "ok"
        assert "prediction" in raw_json

        magika_prediction = raw_json["prediction"]

        assert "dl" in magika_prediction
        assert "output" in magika_prediction
        assert "score" in magika_prediction
        assert "overwrite_reason" in magika_prediction

def test_predict_batch_slices():
    with TestClient(app) as client:
        response = client.post(
            "/predict_batch_slices",
            data={
                "filenames": "test.txt",
            },
            files=[
                (
                    "heads",
                    (
                        "head.bin",
                        b"Hello ",
                        "application/octet-stream",
                    ),
                ),
                (
                    "mids",
                    (
                        "mid.bin",
                        b"this is ",
                        "application/octet-stream",
                    ),
                ),
                (
                    "tails",
                    (
                        "tail.bin",
                        b"a text file.",
                        "application/octet-stream",
                    ),
                ),
            ],
        )

        assert response.status_code == 200

        data = response.json()

        assert "results" in data
        assert len(data["results"]) == 1

        result = data["results"][0]

        assert result["filename"] == "test.txt"
        assert "label" in result
        assert "mime_type" in result
        assert "score" in result
        assert "raw_json" in result

        raw_json = result["raw_json"]

        assert isinstance(raw_json, dict)
        assert raw_json["status"] == "ok"
        assert "prediction" in raw_json

        magika_prediction = raw_json["prediction"]

        assert "dl" in magika_prediction
        assert "output" in magika_prediction
        assert "score" in magika_prediction
        assert "overwrite_reason" in magika_prediction

def test_predict_batch_slices_rejects_mismatched_lengths():
    with TestClient(app) as client:
        response = client.post(
            "/predict_batch_slices",
            data={
                "filenames": [
                    "test1.txt",
                    "test2.txt",
                ],
            },
            files=[
                (
                    "heads",
                    (
                        "head1.bin",
                        b"Hello ",
                        "application/octet-stream",
                    ),
                ),
                (
                    "mids",
                    (
                        "mid1.bin",
                        b"world",
                        "application/octet-stream",
                    ),
                ),
                (
                    "tails",
                    (
                        "tail1.bin",
                        b"!",
                        "application/octet-stream",
                    ),
                ),
            ],
        )

        assert response.status_code == 400

        data = response.json()

        assert data["detail"] == (
            "Mismatch in length of filenames and slice arrays."
        )

def test_detect_extension_mismatch_requires_file():
    with TestClient(app) as client:
        response = client.post(
            "/detect_extension_mismatch?filename=test.txt"
        )

        assert response.status_code == 422

def test_predict_slices_requires_files():
    with TestClient(app) as client:
        response = client.post("/predict_slices")

        assert response.status_code == 422

def test_get_app_info():
    with TestClient(app) as client:
        response = client.get("/api/info")

        assert response.status_code == 200

        data = response.json()

        assert data["app_version"] == "1.0.3"
        assert data["magika_package_version"] == "1.0.3"

        assert data["model_name"] not in (
            "",
            "unknown",
            "not loaded",
        )

def test_predict_slices_returns_correct_filenames():
    with TestClient(app) as client:
        response = client.post(
            "/predict_slices",
            files=[
                (
                    "files",
                    (
                        "first.txt",
                        b"First text file.",
                        "text/plain",
                    ),
                ),
                (
                    "files",
                    (
                        "second.txt",
                        b"Second text file.",
                        "text/plain",
                    ),
                ),
            ],
        )

        assert response.status_code == 200

        data = response.json()

        predictions = data["predictions"]

        assert len(predictions) == 2
        assert predictions[0]["filename"] == "first.txt"
        assert predictions[1]["filename"] == "second.txt"

def test_detect_extension_mismatch_response_consistency():
    with TestClient(app) as client:
        response = client.post(
            "/detect_extension_mismatch?filename=test.txt",
            files={
                "file": (
                    "test.txt",
                    b"Hello, this is a text file.",
                    "text/plain",
                )
            },
        )

        assert response.status_code == 200

        data = response.json()

        raw_json = data["raw_json"]

        assert isinstance(raw_json, dict)
        assert raw_json["status"] == "ok"
        assert "prediction" in raw_json

        magika_prediction = raw_json["prediction"]

        assert "output" in magika_prediction
        assert "score" in magika_prediction

        # The convenient API fields must match Magika's raw output.
        assert (
            data["detected_label"]
            == magika_prediction["output"]["label"]
        )

        assert data["score"] == magika_prediction["score"]

        assert data["filename"] == "test.txt"
        assert data["provided_extension"] == "txt"