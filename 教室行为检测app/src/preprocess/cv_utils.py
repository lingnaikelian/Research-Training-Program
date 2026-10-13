"""OpenCV 在 Windows 上不支持中文路径，这里封装两个工具函数。"""
from pathlib import Path
import cv2
import numpy as np


def imread_unicode(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img) -> bool:
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True
