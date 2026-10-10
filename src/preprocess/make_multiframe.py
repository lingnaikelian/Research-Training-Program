"""
多帧拼接：把同一人连续 3 帧的裁剪图横排拼成一张图，让 VLM 能看出动作变化
（转头、说话、起身走开等时序行为）。

输入：src/data/crops/<video>/frame_00012_person_0.jpg ...
输出：src/data/crops_multi/<video>/frame_00012_person_0.jpg（拼接图，同名）

用法：python make_multiframe.py [video ...] 或 --all
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from cv_utils import imread_unicode, imwrite_unicode

ROOT = Path(__file__).resolve().parents[2]
CROP_DIR = ROOT / "src" / "data" / "crops"
MULTI_DIR = ROOT / "src" / "data" / "crops_multi"

GRAY = np.full((200, 200, 3), 180, dtype=np.uint8)  # 缺帧占位块


def parse_frame_person(name: str):
    """frame_00012_person_0.jpg -> (12, 0)"""
    stem = name[:-4]
    parts = stem.split("_")  # frame, 00012, person, 0
    return int(parts[1]), int(parts[3])


def build_multi(video: str, stride: int = 1, resize_h: int = 0):
    src_dir = CROP_DIR / video
    if not src_dir.is_dir():
        print(f"[跳过] 无 {video}")
        return
    out_dir = MULTI_DIR / video
    out_dir.mkdir(parents=True, exist_ok=True)

    # 建立 frame -> {person_idx: 图} 索引
    frame_map = {}
    for fp in src_dir.glob("frame_*.jpg"):
        f, p = parse_frame_person(fp.name)
        frame_map.setdefault(f, {})[p] = fp

    made = 0
    for f in sorted(frame_map):
        if f % stride != 0:
            continue
        for p in sorted(frame_map[f]):
            # 取本帧、前1帧、前2帧的同 person 图；缺失时用最近的可用帧补足
            tiles = []
            for offset in (2, 1, 0):
                src_fp = frame_map.get(f - offset, {}).get(p)
                if src_fp is not None:
                    tiles.append(imread_unicode(src_fp))
                    continue
                # 缺帧：向前找最近一帧的同 person
                fallback = None
                for k in range(offset - 1, -1, -1):
                    cand = frame_map.get(f - k, {}).get(p)
                    if cand is not None:
                        fallback = imread_unicode(cand)
                        break
                tiles.append(fallback if fallback is not None else GRAY)
            # 统一高度后横排
            h = max(t.shape[0] for t in tiles)
            resized = []
            for t in tiles:
                th, tw = t.shape[:2]
                if th != h:
                    scale = h / th
                    t = cv2.resize(t, (int(tw * scale), h))
                resized.append(t)
            combo = np.hstack(resized)
            if resize_h and combo.shape[0] > resize_h:
                scale = resize_h / combo.shape[0]
                combo = cv2.resize(
                    combo, (int(combo.shape[1] * scale), resize_h))
            out_name = f"frame_{f:05d}_person_{p}.jpg"
            imwrite_unicode(out_dir / out_name, combo)
            made += 1

    print(f"[OK] {video}: 生成 {made} 张拼接图 -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="视频名；空 = 全部")
    ap.add_argument("--stride", type=int, default=1,
                    help="抽稀间隔：每 stride 帧生成一张拼接图（默认1=全生成）")
    ap.add_argument("--resize", type=int, default=0,
                    help="拼接图缩放高度（像素），0=不缩放")
    args = ap.parse_args()

    if args.videos:
        for v in args.videos:
            build_multi(v, args.stride, args.resize)
    else:
        for d in sorted(CROP_DIR.iterdir()):
            if d.is_dir():
                build_multi(d.name, args.stride, args.resize)
