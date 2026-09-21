from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image


def read_gray(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image")
    return arr


def write_gray_png(path: str | Path, image: np.ndarray) -> None:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Expected uint8 2-D grayscale image")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(path, format="PNG", optimize=False)
