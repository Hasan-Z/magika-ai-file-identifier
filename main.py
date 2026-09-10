import importlib.metadata
import os
import traceback
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from magika import Magika


magika_instance = None
model_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global magika_instance, model_ready

    print("[INFO] Starting Magika initialization...", flush=True)

    try:
        magika_instance = Magika()
        model_ready = True

        print(
            "[INFO] Magika model loaded successfully.",
            flush=True,
        )

        try:
            print(
                f"[INFO] Magika model name: "
                f"{magika_instance.get_model_name()}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[WARNING] Could not get Magika model name: {e}",
                flush=True,
            )

    except Exception as e:
        magika_instance = None
        model_ready = False

        print(
            f"[ERROR] Failed to load Magika model: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        traceback.print_exc()

    yield

    print("[INFO] Shutting down Magika...", flush=True)

    magika_instance = None
    model_ready = False


app = FastAPI(
    title="Magika AI File Identifier",
    description=(
        "FastAPI service for high-performance file type "
        "identification using Google Magika."
    ),
    version="1.0.3",
    lifespan=lifespan,
)


os.makedirs("static", exist_ok=True)
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


@app.get("/", response_class=FileResponse)
async def read_index():
    index_path = os.path.join("static", "index.html")

    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail="Front-end index.html not found in static/ directory.",
        )

    return FileResponse(index_path)


@app.get("/status")
async def get_status():
    return {
        "ready": model_ready,
        "version": "1.0.3",
    }


def build_magika_response(filename: str, res) -> dict:
    """
    Build the application response while preserving the complete
    Magika result untouched inside raw_json.
    """

    output = res.output

    # These fields are convenience fields provided by our API.
    label = str(output.label)
    mime_type = str(output.mime_type)
    group = str(output.group)
    description = str(output.description)
    is_text = bool(output.is_text)
    score = float(res.score)

    # Preserve the complete Magika result without rebuilding it.
    raw_json = res.asdict()

    return {
        "filename": filename,
        "label": label,
        "mime_type": mime_type,
        "group": group,
        "description": description,
        "is_text": is_text,
        "score": score,
        "raw_json": raw_json,
    }


@app.post("/predict_slices")
async def predict_slices(
    files: List[UploadFile] = File(...),
):
    if not model_ready or magika_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Magika model is not ready.",
        )

    results = []

    for file in files:
        content = await file.read()

        res = magika_instance.identify_bytes(content)

        results.append(
            build_magika_response(
                file.filename,
                res,
            )
        )

    return {
        "predictions": results,
    }


@app.post("/predict_batch_slices")
async def predict_batch_slices(
    filenames: List[str] = Form(...),
    heads: List[UploadFile] = File(...),
    mids: List[UploadFile] = File(...),
    tails: List[UploadFile] = File(...),
):
    if not model_ready or magika_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Magika model is not ready.",
        )

    if not (
        len(filenames)
        == len(heads)
        == len(mids)
        == len(tails)
    ):
        raise HTTPException(
            status_code=400,
            detail="Mismatch in length of filenames and slice arrays.",
        )

    batch_results = []

    for idx in range(len(filenames)):
        name = filenames[idx]

        head_bytes = await heads[idx].read()
        mid_bytes = await mids[idx].read()
        tail_bytes = await tails[idx].read()

        combined_bytes = (
            head_bytes
            + mid_bytes
            + tail_bytes
        )

        res = magika_instance.identify_bytes(
            combined_bytes
        )

        batch_results.append(
            build_magika_response(
                name,
                res,
            )
        )

    return {
        "results": batch_results,
    }


@app.post("/detect_extension_mismatch")
async def detect_extension_mismatch(
    filename: str,
    file: UploadFile = File(...),
):
    if not model_ready or magika_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Magika model is not ready.",
        )

    content = await file.read()

    res = magika_instance.identify_bytes(content)

    detected_label = str(res.output.label)

    extracted_ext = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    is_mismatch = (
        extracted_ext != ""
        and extracted_ext != detected_label.lower()
    )

    full_response = build_magika_response(
        filename,
        res,
    )

    return {
        "filename": filename,
        "provided_extension": extracted_ext,
        "detected_label": detected_label,
        "score": full_response["score"],
        "mismatch": is_mismatch,
        "raw_json": full_response["raw_json"],
    }


@app.get("/api/info")
def get_app_info():
    # Installed Python package version.
    package_version = importlib.metadata.version("magika")

    # Actual model loaded by the Magika instance.
    if magika_instance is not None:
        try:
            model_name = magika_instance.get_model_name()
        except AttributeError:
            # Compatibility fallback for older Magika versions.
            model_name = getattr(
                magika_instance,
                "_model_name",
                "unknown",
            )
    else:
        model_name = "not loaded"

    return {
        "app_version": "1.0.3",
        "magika_package_version": package_version,
        "model_name": model_name,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )