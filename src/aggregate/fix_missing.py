"""
校验并补全推理结果：对比 crops(或 crops_multi)/<video> 的图片与 results/json/<video>.jsonl，
缺失的帧补写"无行为"记录（reason 标注），保证事件聚合数据完整。

用法：python fix_missing.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "results" / "json"
CROP_DIR = ROOT / "src" / "data" / "crops"
MULTI_DIR = ROOT / "src" / "data" / "crops_multi"

BEHAVIORS = ["sleeping", "looking_phone", "looking_around", "talking", "away"]


def fix_video(video: str, use_multi: bool):
    img_dir = (MULTI_DIR if use_multi else CROP_DIR) / video
    out_file = JSON_DIR / f"{video}.jsonl"
    if not img_dir.is_dir():
        return 0

    # 已记录
    done = set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["crop"])
                except Exception:
                    pass

    missing = [fp.name for fp in sorted(img_dir.glob("*.jpg"))
               if fp.name not in done]
    if not missing:
        return 0

    rec = {"sleeping": 0, "looking_phone": 0, "looking_around": 0,
           "talking": 0, "away": 0,
           "reason": "模型未返回有效JSON，按无行为处理"}
    with out_file.open("a", encoding="utf-8") as f:
        for name in missing:
            r = {"crop": name, "video": video, **rec}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[补全] {video}: 补写 {len(missing)} 条缺失帧")
    return len(missing)


if __name__ == "__main__":
    total = 0
    # 多帧重跑的视频（crops_multi 下）
    if MULTI_DIR.is_dir():
        for d in sorted(MULTI_DIR.iterdir()):
            if d.is_dir():
                total += fix_video(d.name, True)
    # 单帧视频（crops 下未重跑的）
    for d in sorted(CROP_DIR.iterdir()):
        if not d.is_dir():
            continue
        if (MULTI_DIR / d.name).is_dir():
            continue  # 已用多帧重跑
        total += fix_video(d.name, False)
    print(f"共补全 {total} 条")
