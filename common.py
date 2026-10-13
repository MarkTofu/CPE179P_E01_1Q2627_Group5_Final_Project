"""Single source of truth for image loading + preprocessing.
Training, export and the API all use this, so they cannot drift apart."""
import numpy as np
from PIL import Image, ImageOps

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet stats (MobileNetV3 weights)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(src):
    """Path or file-like -> RGB PIL image. exif_transpose fixes sideways phone photos."""
    return ImageOps.exif_transpose(Image.open(src)).convert("RGB")


def preprocess(img):
    """PIL image -> float32 array shaped (1, 3, 224, 224)."""
    a = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
