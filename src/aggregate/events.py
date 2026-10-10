"""
事件聚合：把逐帧行为判断合并成"事件"。

逻辑（帧级聚合）：
1. 读 results/json/<video>.jsonl
2. 每帧把该帧所有 person 的行为合并：某行为被任一 person 触发 -> 该帧标记 1
3. 对每个行为单独扫描时间连续性：连续 >= min_frames 帧（1fps = 秒）记为一个事件
4. 过滤过短事件，输出 results/events/<video>.json + 时间轴图

用法：
    python events.py <video_name>        # 单视频
    python events.py --all               # 全部
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt

# 中文字体（Windows 微软雅黑 / 黑体）
for _f in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "results" / "json"
EVENT_DIR = ROOT / "results" / "events"
FIG_DIR = ROOT / "results" / "report"

BEHAVIORS = ["sleeping", "looking_phone", "looking_around", "talking", "away"]
BEHAVIOR_CN = {
    "sleeping": "睡觉",
    "looking_phone": "看手机",
    "looking_around": "张望",
    "talking": "交谈",
    "away": "离座",
}
COLORS = {
    "sleeping": "#8BC8EA",
    "looking_phone": "#9BBBF4",
    "looking_around": "#94D8C3",
    "talking": "#E4D48F",
    "away": "#EAA7B2",
}


def parse_frame_index(crop_name: str) -> int:
    """frame_00012_person_1.jpg -> 12"""
    try:
        return int(crop_name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def frame_level_series(video: str) -> dict:
    """返回 {behavior: {frame_idx: 0/1}}，帧级合并所有 person。"""
    series = {b: {} for b in BEHAVIORS}
    path = JSON_DIR / f"{video}.jsonl"
    if not path.exists():
        return series
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        idx = parse_frame_index(rec.get("crop", ""))
        if idx < 0:
            continue
        for b in BEHAVIORS:
            v = int(rec.get(b, 0) or 0)
            series[b][idx] = max(series[b].get(idx, 0), v)
    return series


def detect_events(series: dict, min_frames: int = 5, stride: int = 1,
                  max_gap: int = 3) -> list:
    """连续（允许 gap <= max_gap）覆盖 >= min_frames 秒的同标签 -> 事件。

    stride=1 时帧号即秒；stride>1（多帧抽稀）时相邻推理帧间隔为 stride，
    事件按时间跨度判断。
    """
    events = []
    for b in BEHAVIORS:
        frames = sorted(series[b].items())
        run_start = None
        run_sec = 0
        prev_idx = None
        for idx, val in frames:
            if val == 1:
                if run_start is None:
                    run_start = idx
                    run_sec = 1
                else:
                    # 允许推理帧之间的间隔（gap <= max_gap 视为连续）
                    if prev_idx is not None and (idx - prev_idx) <= max_gap:
                        run_sec += (idx - prev_idx)
                    else:
                        # 间隔过大：收尾上一个事件，另起新事件
                        if run_sec >= min_frames:
                            events.append({
                                "behavior": b,
                                "start": run_start,
                                "end": prev_idx,
                                "duration": run_sec,
                                "confidence": round(
                                    run_sec / max(run_sec + 10, 1), 2),
                            })
                        run_start = idx
                        run_sec = 1
                prev_idx = idx
            else:
                if run_start is not None and run_sec >= min_frames:
                    events.append({
                        "behavior": b,
                        "start": run_start,
                        "end": prev_idx,
                        "duration": run_sec,
                        "confidence": round(
                            run_sec / max(run_sec + 10, 1), 2),
                    })
                run_start = None
                run_sec = 0
                prev_idx = None
        if run_start is not None and run_sec >= min_frames:
            events.append({
                "behavior": b,
                "start": run_start,
                "end": prev_idx,
                "duration": run_sec,
                "confidence": round(run_sec / max(run_sec + 10, 1), 2),
            })
    events.sort(key=lambda e: (e["start"], e["behavior"]))
    return events


def plot_timeline(video: str, events: list, max_sec: int = 45):
    """画行为时间轴（横条图）。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    rows = {b: i for i, b in enumerate(BEHAVIORS)}
    for e in events:
        ax.barh(rows[e["behavior"]], e["end"] - e["start"] + 1,
                left=e["start"], height=0.55,
                color=COLORS[e["behavior"]], alpha=0.85,
                edgecolor="white", linewidth=0.5)
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels([BEHAVIOR_CN[b] for b in BEHAVIORS])
    ax.set_xlabel("时间（秒）")
    ax.set_xlim(0, max_sec)
    ax.set_title(f"{video} 行为时间轴")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG_DIR / f"{video}_timeline.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def process(video: str, min_frames: int = 5, max_gap: int = 3):
    series = frame_level_series(video)
    events = detect_events(series, min_frames, max_gap=max_gap)
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    out = EVENT_DIR / f"{video}.json"
    out.write_text(json.dumps(events, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    # 摘要
    summary = "、".join(
        f"{BEHAVIOR_CN[e['behavior']]}{e['duration']}s" for e in events) or "无事件"
    print(f"[OK] {video}: {len(events)} 个事件 -> {summary}")
    if events:
        fig = plot_timeline(video, events)
        print(f"     时间轴: {fig}")
    return events


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="视频名")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-frames", type=int, default=5,
                    help="连续至少几帧算事件，默认5（=5秒）")
    args = ap.parse_args()

    if args.all:
        for v in sorted(d.stem for d in JSON_DIR.glob("*.jsonl")):
            process(v, args.min_frames, max_gap=3)
    elif args.video:
        process(args.video, args.min_frames, max_gap=3)
    else:
        print("用法: python events.py <视频名> | --all")
