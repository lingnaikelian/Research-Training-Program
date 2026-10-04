"""
视频抽帧：按固定 fps 把视频拆成图片。

用法：
    python extract_frames.py <video_path> [--fps 1]

输出：
    src/data/frames/<视频文件名>/frame_0001.jpg
"""
import argparse
from pathlib import Path
import cv2
from cv_utils import imwrite_unicode

ROOT = Path(__file__).resolve().parents[2]
FRAME_DIR = ROOT / "src" / "data" / "frames"


def extract(video_path: str, fps: float = 1.0):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_dir = FRAME_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / fps))  # 每 step 帧取一张

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            name = f"frame_{saved:05d}.jpg"
            imwrite_unicode(out_dir / name, frame)
            saved += 1
        idx += 1
    cap.release()

    print(f"[OK] {video_path.name}: {total} 帧 @ {src_fps:.1f}fps -> "
          f"抽 {saved} 帧 -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="视频文件路径")
    ap.add_argument("--fps", type=float, default=1.0, help="抽帧频率，默认 1fps")
    args = ap.parse_args()
    extract(args.video, args.fps)
