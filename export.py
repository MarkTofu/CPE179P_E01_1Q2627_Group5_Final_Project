"""model.pt -> ONNX FP32 -> ONNX INT8, then verify. Run: uv run python export.py
Fails loudly (assert) if parity, accuracy or size targets from the proposal are missed."""
import os
import time

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from torch import nn
from torchvision import models

from common import load, preprocess
from train import splits

FP32, INT8 = "model_fp32.onnx", "model_int8.onnx"
classes, tr, _, te = splits()
items = lambda s: [s.dataset.samples[i] for i in s.indices]  # [(path, label)]
session = lambda p: ort.InferenceSession(p, providers=["CPUExecutionProvider"])

# 1. Export FP32 and check it matches PyTorch on a real image
model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(1280, len(classes))
model.load_state_dict(torch.load("model.pt"))
model.eval()
x = torch.from_numpy(preprocess(load(items(te)[0][0])))
torch.onnx.export(model, x, FP32, input_names=["input"], output_names=["logits"],
                  opset_version=17, dynamo=False)
with torch.no_grad():
    np.testing.assert_allclose(model(x).numpy(), session(FP32).run(None, {"input": x.numpy()})[0], atol=1e-4)
print("parity PyTorch vs ONNX FP32: OK")


# 2. INT8 static quantization, calibrated on 100 training images.
# per_channel is essential for MobileNets (BN-folded depthwise weights vary wildly per channel);
# reduce_range guards against int8 overflow on CPUs without VNNI.
class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter({"input": preprocess(load(p))} for p, _ in items(tr)[:100])

    def get_next(self):
        return next(self.it, None)


quantize_static(FP32, INT8, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                reduce_range=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)


# 3. Benchmark on the held-out test set
def bench(path):
    s, ok, ms = session(path), 0, []
    for p, y in items(te):
        a = preprocess(load(p))
        t = time.perf_counter()
        ok += s.run(None, {"input": a})[0].argmax() == y
        ms.append((time.perf_counter() - t) * 1e3)
    return ok / len(ms), float(np.median(ms)), os.path.getsize(path) / 1e6


print(f"\n{'model':10}{'acc':>7}{'MB':>7}{'median ms':>11}{'img/s':>8}")
(a32, l32, s32), (a8, l8, s8) = bench(FP32), bench(INT8)
for name, a, l, s in [("ONNX FP32", a32, l32, s32), ("ONNX INT8", a8, l8, s8)]:
    print(f"{name:10}{a:7.3f}{s:7.2f}{l:11.2f}{1000 / l:8.0f}")

assert a8 >= a32 - 0.02, f"INT8 lost {100 * (a32 - a8):.1f} points (limit 2)"
assert s8 <= s32 / 2, f"INT8 only {100 * (1 - s8 / s32):.0f}% smaller (need 50%)"
print("\nINT8 checks passed -> serve model_int8.onnx")
