# PesoSight

An edge-AI Philippine peso banknote reader for visually impaired users. A client sends a banknote photo to a FastAPI service that runs an INT8-quantized MobileNetV2 through ONNX Runtime and returns the denomination and a confidence score.

Course project for CPE179P (AI on the Edge), Group 5.

> **Status:** academic prototype. It is not a certified assistive device and should not be the only way someone verifies cash.

## How it works

```
Webcam (Windows)  --POST /predict-->  FastAPI (WSL/Linux)  -->  ONNX Runtime (INT8)  -->  JSON
   webcam.py                             app.py                                  {"denomination": "100", "confidence": 0.98}
```

- **Classes:** 20, 50, 100, 500, 1000 pesos (5 classes, 125 photos each).
- **Model:** MobileNetV2, fine-tuned in PyTorch, exported to ONNX, quantized to INT8 (static, per-channel).
- **Safety floor:** if confidence is below 0.80, the API returns `"denomination": null` instead of guessing. A wrong amount is worse than "try again".

## Results

Held-out test set (94 images):

| Model | Accuracy | Size | Median latency | Throughput |
|---|---|---|---|---|
| ONNX FP32 | 0.989 | 8.89 MB | 2.44 ms | 411 img/s |
| ONNX INT8 | 0.989 | 2.61 MB | 2.00 ms | 500 img/s |

INT8 is about 71% smaller with no accuracy loss. Latency is model inference only, measured on CPU.

**Why MobileNetV2 and not V3-Small?** Static INT8 quantization of MobileNetV3-Small collapsed to 57.4% test accuracy (FP32: 98.9%), even after trying percentile calibration and leaving sensitive layers in float. MobileNetV2 quantizes cleanly.

**Caveat:** the dataset is self-collected and the split is random, so near-duplicate photos can inflate the test score. Check with fresh photos in new lighting before trusting it.

## Project layout

```
pesosight/
├── common.py          shared image loading + preprocessing (used by everything)
├── train.py           fine-tune MobileNetV2 -> model.pt, labels.json
├── export.py          ONNX FP32 -> INT8, parity/accuracy/size checks, benchmark
├── app.py             FastAPI service (/health, /predict)
├── test_app.py        end-to-end smoke test
├── webcam.py          webcam client (SPACE = capture, Q = quit)
├── labels.json        class order used by the API
├── model_int8.onnx    trained INT8 model (so you can run the API without training)
├── pyproject.toml     dependencies (managed with uv)
├── uv.lock
└── .python-version
```

Not in the repo: `data/` (the photos), `model.pt`, `model_fp32.onnx`, `.venv/`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The server was developed on Ubuntu under WSL2.

```bash
git clone https://github.com/MarkTofu/pesosight.git
cd pesosight
uv sync
```

## Run the API (no training needed)

The repo includes `model_int8.onnx` and `labels.json`.

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Check http://localhost:8000/health, or try uploads at http://localhost:8000/docs.

```bash
curl -F "file=@some_banknote.jpg" http://localhost:8000/predict
# {"denomination":"100","confidence":0.9871}
```

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `POST /predict` | multipart field `file` (image, max 10 MB). Returns `denomination` (string or `null`) and `confidence`. 400 for non-images, 413 for oversize. |

## Webcam client

WSL cannot access a webcam, so run this in regular Windows (PowerShell) while the server is running:

```powershell
uv run --no-project --with opencv-python --with requests webcam.py
```

`--no-project` stops uv from installing the heavy training dependencies on Windows. Press **SPACE** to capture and **Q** or **Esc** to quit. If you have several cameras, change `VideoCapture(0)` in `webcam.py`.

## Train your own model

1. Put photos in `data/`, one folder per denomination (folder names are the labels):

   ```
   data/
   ├── 20/    ├── 50/    ├── 100/    ├── 500/    └── 1000/
   ```

   JPG or PNG only (convert HEIC first). Use varied lighting, backgrounds and wear, and avoid long runs of near-identical shots. Keep `data/` on the Linux filesystem (not `/mnt/c`) for speed.

2. Train, export and verify:

   ```bash
   uv run python train.py     # prints val accuracy per epoch and final TEST accuracy
   uv run python export.py    # asserts: ONNX parity, INT8 within 2 points, >= 50% smaller
   uv run python test_app.py  # end-to-end smoke test (needs data/)
   ```

`export.py` fails loudly if the INT8 model misses the targets, so a bad model is never served silently.

## Scope and limitations

- Five genuine Philippine banknote denominations only. No coins, counterfeit detection or foreign currency.
- Self-collected data may not cover all lighting and wear conditions. Webcam photos look different from phone photos, so include webcam shots in training if you use the webcam client.
- The original proposal included a Raspberry Pi client with a button, LCD and eSpeak NG. That part was dropped; the webcam client replaces it.

## Authors

Group 5, CPE179P_E01: John Simon Guarin Dominguez, Mark Owen T. Loro, and Joseph Brian B. Pablo.
