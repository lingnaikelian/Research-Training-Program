"""多帧拼接：取每帧所有行人的联合区域，按 stride 抽稀后把连续 6 个时刻
拼成 2 行 x 3 列网格，让 VLM 同时看到多人位置关系与动作变化。

输出：src/data/crops_union/<video>/frame_00000_union.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.preprocess.cv_utils import imread_unicode, imwrite_unicode

FRAME_DIR = ROOT / "src" / "data" / "frames"
UNION_DIR = ROOT / "src" / "data" / "crops_union"

GRAY = np.full((300, 300, 3), 180, dtype=np.uint8)  # 缺帧占位
PAD_RATIO = 0.1


def get_union_boxes(video: str, model, conf: float = 0.3):
    """对视频每帧跑 YOLO，返回 {frame_idx: [(x1,y1,x2,y2), ...]}。"""
    frame_dir = FRAME_DIR / video
    result = {}
    for fp in sorted(frame_dir.glob("frame_*.jpg")):
        idx = int(fp.stem.split("_")[1])
        img = imread_unicode(fp)
        r = model(img, conf=conf, classes=[0], verbose=False)[0]
        boxes = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((x1, y1, x2, y2))
        result[idx] = boxes
    return result


def build_union(video: str, stride: int = 4, resize_h: int = 256):
    frame_dir = FRAME_DIR / video
    if not frame_dir.is_dir():
        print(f"[跳过] 无 {video}")
        return
    model = YOLO(ROOT / "src" / "preprocess" / "yolov8n-pose.pt")
    boxes_by_frame = get_union_boxes(video, model)
    if not boxes_by_frame:
        print(f"[跳过] {video} 无检测结果")
        return

    out_dir = UNION_DIR / video
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idxs = sorted(boxes_by_frame)
    anchors = [f for f in frame_idxs if f % stride == 0]
    made = 0
    for f in anchors:
        times = [f - stride * k for k in range(5, -1, -1)]
        tiles = []
        for t in times:
            boxes = boxes_by_frame.get(t)
            if not boxes:
                tiles.append(GRAY)
                continue
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)
            img = imread_unicode(frame_dir / f"frame_{t:05d}.jpg")
            h, w = img.shape[:2]
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - bw * PAD_RATIO))
            y1 = max(0, int(y1 - bh * PAD_RATIO))
            x2 = min(w, int(x2 + bw * PAD_RATIO))
            y2 = min(h, int(y2 + bh * PAD_RATIO))
            tiles.append(img[y1:y2, x1:x2])

        target_h = max(t.shape[0] for t in tiles)
        tiles2 = []
        for t in tiles:
            th, tw = t.shape[:2]
            scale = target_h / th
            tiles2.append(cv2.resize(t, (int(tw * scale), target_h)))
        target_w = max(t.shape[1] for t in tiles2)
        tiles3 = []
        for t in tiles2:
            tw = t.shape[1]
            pad_l = (target_w - tw) // 2
            pad_r = target_w - tw - pad_l
            tiles3.append(cv2.copyMakeBorder(
                t, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT,
                value=(180, 180, 180)))
        r1 = np.hstack(tiles3[:3])
        r2 = np.hstack(tiles3[3:])
        combo = np.vstack([r1, r2])
        if resize_h and combo.shape[0] > resize_h:
            scale = resize_h / combo.shape[0]
            combo = cv2.resize(combo, (int(combo.shape[1] * scale), resize_h))
        out_name = f"frame_{f:05d}_union.jpg"
        imwrite_unicode(out_dir / out_name, combo)
        made += 1

    print(f"[OK] {video}: 生成 {made} 张联合拼接图 -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="视频名；空 = 全部")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--resize", type=int, default=256)
    args = ap.parse_args()

    videos = args.videos or [d.name for d in FRAME_DIR.iterdir() if d.is_dir()]
    for v in videos:
        build_union(v, args.stride, args.resize)
