import os
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from magika import Magika

magika_instance = None
model_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global magika_instance, model_ready
    try:
        magika_instance = Magika()
        model_ready = True
        print("[INFO] Magika model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load Magika model: {e}")
        model_ready = False
    yield
    magika_instance = None
    model_ready = False


app = FastAPI(
    title="Magika AI File Identifier",
    description="FastAPI service for high-performance file type identification using Google Magika.",
    version="1.0.0",
    lifespan=lifespan,
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


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
    return {"ready": model_ready, "version": "1.0.0"}


def build_magika_response(filename: str, res) -> dict:
    """Formats Magika 1.0+ output attributes into a unified payload."""
    ct_label = getattr(res.output, "ct_label", "unknown")
    score = float(res.score)

    # Extract metadata using Magika helper or fallback lookup
    group = "unknown"
    description = ""
    is_text = False
    mime_type = getattr(res.output, "mime_type", "application/octet-stream")

    if magika_instance is not None:
        try:
            # Magika API method to get detailed info object
            info = magika_instance.get_output_content_type(res)
            group = getattr(info, "group", "unknown")
            description = getattr(info, "description", "")
            is_text = getattr(info, "is_text", False)
            mime_type = getattr(info, "mime_type", mime_type)
        except Exception:
            pass

    raw_details = {
        "filename": filename,
        "score": round(score, 4),
        "ct_label": ct_label,
        "mime_type": mime_type,
        "group": group,
        "description": description,
        "is_text": is_text,
    }

    return {
        "filename": filename,
        "label": ct_label,
        "mime_type": mime_type,
        "group": group,
        "description": description or "N/A",
        "is_text": is_text,
        "score": round(score, 4),
        "raw_json": raw_details,
    }


@app.post("/predict_slices")
async def predict_slices(files: List[UploadFile] = File(...)):
    if not model_ready or magika_instance is None:
        raise HTTPException(status_code=503, detail="Magika model is not ready.")

    results = []
    for file in files:
        content = await file.read()
        res = magika_instance.identify_bytes(content)
        results.append(build_magika_response(file.filename, res))

    return {"predictions": results}


@app.post("/predict_batch_slices")
async def predict_batch_slices(
    filenames: List[str] = Form(...),
    heads: List[UploadFile] = File(...),
    mids: List[UploadFile] = File(...),
    tails: List[UploadFile] = File(...),
):
    if not model_ready or magika_instance is None:
        raise HTTPException(status_code=503, detail="Magika model is not ready.")

    if not (len(filenames) == len(heads) == len(mids) == len(tails)):
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

        combined_bytes = head_bytes + mid_bytes + tail_bytes
        res = magika_instance.identify_bytes(combined_bytes)
        batch_results.append(build_magika_response(name, res))

    return {"results": batch_results}


@app.post("/detect_extension_mismatch")
async def detect_extension_mismatch(
    filename: str,
    file: UploadFile = File(...)
):
    if not model_ready or magika_instance is None:
        raise HTTPException(status_code=503, detail="Magika model is not ready.")

    content = await file.read()
    res = magika_instance.identify_bytes(content)

    detected_label = getattr(res.output, "ct_label", "unknown")
    extracted_ext = filename.split(".")[-1].lower() if "." in filename else ""

    is_mismatch = extracted_ext != "" and extracted_ext != detected_label.lower()
    full_resp = build_magika_response(filename, res)

    return {
        "filename": filename,
        "provided_extension": extracted_ext,
        "detected_label": detected_label,
        "score": full_resp["score"],
        "mismatch": is_mismatch,
        "raw_json": full_resp["raw_json"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)