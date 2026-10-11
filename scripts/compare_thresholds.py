"""对比事件阈值配置：统一 5s vs 瞬态行为(张望/交谈)降为 3s。

只做内存评估，不写 events 文件。命中口径与 eval_baseline.py 一致：
事件列表中存在预期行为即命中。
"""
import sys
from pathlib import Path

sys.path.insert(0, "src/aggregate")
from events import (frame_level_series, detect_events,
                    detector_away_events)

ROOT = Path.cwd()
JSON_DIR = ROOT / "results" / "json"
UNION_DIR = ROOT / "src" / "data" / "crops_union"

# 前缀 -> 预期行为（英文键，与事件 behavior 字段一致）
PREFIX2BEHAVIOR_EN = {
    "sleeping": "sleeping", "phone": "looking_phone",
    "looking_around": "looking_around", "talking": "talking",
    "leaving": "away",
}
CN = {"sleeping": "睡觉", "looking_phone": "看手机",
      "looking_around": "张望", "talking": "交谈", "away": "离座"}


def run_all(behavior_minframes=None, min_frames=5):
    hits = {}
    for jl in sorted(JSON_DIR.glob("*.jsonl")):
        v = jl.stem
        prefix = v.rsplit("_", 1)[0]
        expect = PREFIX2BEHAVIOR_EN.get(prefix)
        if expect is None:
            continue
        series = frame_level_series(v)
        max_gap = 6 if (UNION_DIR / v).is_dir() else 3
        ev = detect_events(series, min_frames, max_gap=max_gap,
                           behavior_minframes=behavior_minframes)
        if not any(e["behavior"] == "away" for e in ev):
            ev += detector_away_events(v, min_frames)
        hits[expect] = hits.get(expect, 0) + int(
            any(e["behavior"] == expect for e in ev))
    return hits


if __name__ == "__main__":
    configs = [
        ("统一 5 秒（现状）", None),
        ("张望/交谈 3 秒", {"looking_around": 3, "talking": 3}),
        ("仅张望 3 秒", {"looking_around": 3}),
    ]
    for name, bmf in configs:
        h = run_all(bmf)
        line = "  ".join(f"{CN[k]}:{h.get(k, 0)}/5" for k in
                         ["sleeping", "looking_phone", "looking_around",
                          "talking", "away"])
        print(f"{name}: {line} | 合计 {sum(h.values())}/25")
