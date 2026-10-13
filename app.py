"""PesoSight API. Run: uvicorn app:app --host 0.0.0.0 --port 8000"""
import io
import json

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile

from common import load, preprocess

LABELS = json.load(open("labels.json"))
MIN_CONF = 0.80          # below this we refuse to guess: a wrong spoken amount is worse than "try again"
MAX_BYTES = 10_000_000
app = FastAPI(title="PesoSight")
sess = ort.InferenceSession("model_int8.onnx", providers=["CPUExecutionProvider"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):  # plain def: FastAPI runs it in a threadpool, so inference doesn't block the loop
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large")
    try:
        x = preprocess(load(io.BytesIO(data)))
    except OSError:  # includes PIL.UnidentifiedImageError
        raise HTTPException(400, "not a valid image")
    z = sess.run(None, {"input": x})[0][0]
    p = np.exp(z - z.max())
    p /= p.sum()
    i = int(p.argmax())
    return {"denomination": LABELS[i] if p[i] >= MIN_CONF else None, "confidence": round(float(p[i]), 4)}
