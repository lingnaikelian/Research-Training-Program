"""
用 YOLOv8 检测人体，裁剪出每个人的 bbox（加 padding）。

用法：
    python detect_persons.py <frames_dir>

输入：src/data/frames/<video>/frame_0001.jpg ...
输出：src/data/crops/<video>/frame_0001_person_0.jpg ...
"""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from cv_utils import imread_unicode, imwrite_unicode

ROOT = Path(__file__).resolve().parents[2]
CROP_DIR = ROOT / "src" / "data" / "crops"
FRAME_DIR = ROOT / "src" / "data" / "frames"

# 全局只加载一次模型
_model = None
def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")  # nano 版最快，CPU 也能跑
    return _model


def detect_in_dir(frames_dir: Path, pad_ratio: float = 0.1):
    out_dir = CROP_DIR / frames_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model()
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    total_persons = 0

    for fp in frames:
        img = imread_unicode(fp)
        h, w = img.shape[:2]
        # 只检测 person（class 0），conf 0.3
        results = model(img, conf=0.3, classes=[0], verbose=False)
        persons = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # 加 padding
                bw, bh = x2 - x1, y2 - y1
                x1 = max(0, int(x1 - bw * pad_ratio))
                y1 = max(0, int(y1 - bh * pad_ratio))
                x2 = min(w, int(x2 + bw * pad_ratio))
                y2 = min(h, int(y2 + bh * pad_ratio))
                crop = img[y1:y2, x1:x2]
                out_name = f"{fp.stem}_person_{persons}.jpg"
                imwrite_unicode(out_dir / out_name, crop)
                persons += 1
        total_persons += persons

    print(f"[OK] {frames_dir.name}: {len(frames)} 帧, 共裁出 {total_persons} 个人体 -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir", nargs="?",
                    help=f"抽帧目录，默认处理 {FRAME_DIR} 下所有")
    args = ap.parse_args()

    if args.frames_dir:
        detect_in_dir(Path(args.frames_dir))
    else:
        for d in sorted(FRAME_DIR.iterdir()):
            if d.is_dir():
                detect_in_dir(d)
