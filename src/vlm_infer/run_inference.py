"""
批量行为推理：遍历 crops/<video>/ 下的裁剪图，逐张调 Qwen3-VL，
结果追加到 results/json/<video>.jsonl（支持断点续跑）。

用法：
    python run_inference.py <video_name>          # 只跑一个视频
    python run_inference.py --all                 # 跑全部视频
    python run_inference.py <video_name> --limit 10   # 只跑前10张（测试）
"""
import argparse
import json
import sys
import time
from pathlib import Path

from client_qwen_vl import QwenVLClient
from prompts import build_prompt

ROOT = Path(__file__).resolve().parents[2]
CROP_DIR = ROOT / "src" / "data" / "crops"
RESULT_DIR = ROOT / "results" / "json"

_client = None
def get_client():
    global _client
    if _client is None:
        _client = QwenVLClient()
    return _client


def infer_video(video_name: str, limit: int = 0, sleep_sec: float = 0.0,
                crops_dir: Path = CROP_DIR, multiframe: bool = False):
    """对单个视频的裁剪图批量推理。"""
    crop_dir = crops_dir / video_name
    if not crop_dir.is_dir():
        print(f"[FAIL] 找不到裁剪目录: {crop_dir}")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULT_DIR / f"{video_name}.jsonl"

    # 已处理的帧（断点续跑）
    done = set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["crop"])
            except Exception:
                pass

    frames = sorted(crop_dir.glob("frame_*.jpg"))
    if limit:
        frames = frames[:limit]

    client = get_client()
    prompt = build_prompt(multiframe=multiframe)
    ok = skip = fail = 0
    t0 = time.time()

    with out_file.open("a", encoding="utf-8") as f:
        for fp in frames:
            if fp.name in done:
                skip += 1
                continue
            try:
                result = client.analyze_image(str(fp), prompt, expect_json=True)
                record = {
                    "crop": fp.name,
                    "video": video_name,
                    **result,  # sleeping / looking_phone / ... / reason
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  [ERR] {fp.name}: {type(e).__name__}: {e}")
            if sleep_sec:
                time.sleep(sleep_sec)

    cost = time.time() - t0
    print(f"[DONE] {video_name}: 新增 {ok}, 跳过 {skip}, 失败 {fail}, "
          f"耗时 {cost:.1f}s")
    if ok:
        avg = cost / ok
        print(f"       平均 {avg:.2f}s/张，预计每视频(42张)约 {avg*42:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="视频名（crops 下的目录名）")
    ap.add_argument("--all", action="store_true", help="处理全部视频")
    ap.add_argument("--limit", type=int, default=0, help="每视频只跑前 N 张")
    ap.add_argument("--force", action="store_true",
                    help="删除该视频旧结果后重跑（prompt 改动后使用）")
    ap.add_argument("--multiframe", action="store_true",
                    help="使用多帧拼接图（crops_multi/）与多帧 Prompt")
    args = ap.parse_args()

    use_dir = (ROOT / "src" / "data" / "crops_multi") if args.multiframe else CROP_DIR

    if args.force and args.video:
        old = RESULT_DIR / f"{args.video}.jsonl"
        if old.exists():
            old.unlink()
            print(f"[重跑] 已删除旧结果: {old.name}")

    if args.all:
        videos = sorted(d.name for d in use_dir.iterdir() if d.is_dir())
        for v in videos:
            infer_video(v, args.limit, crops_dir=use_dir,
                        multiframe=args.multiframe)
    elif args.video:
        infer_video(args.video, args.limit, crops_dir=use_dir,
                    multiframe=args.multiframe)
    else:
        print("用法: python run_inference.py <视频名> | --all [--limit N]")
        sys.exit(1)
