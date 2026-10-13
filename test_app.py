"""End-to-end smoke test (catches label-order bugs). Run: python test_app.py"""
import glob

from fastapi.testclient import TestClient

from app import app

c = TestClient(app)
assert c.get("/health").json() == {"status": "ok"}

for d in ["20", "50", "100", "500", "1000"]:
    for path in sorted(glob.glob(f"data/{d}/*"))[:3]:
        r = c.post("/predict", files={"file": open(path, "rb")}).json()
        print(d, r)
        assert r["denomination"] == d, f"{path} -> {r}"

assert c.post("/predict", files={"file": ("x.txt", b"not an image")}).status_code == 400
print("all good")
