"""
OpenCV 在 Windows 上不支持中文路径，封装两个工具函数。
"""
from pathlib import Path
import cv2
import numpy as np


def imread_unicode(path) -> np.ndarray:
    """替代 cv2.imread，支持中文路径。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img) -> bool:
    """替代 cv2.imwrite，支持中文路径。"""
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True
