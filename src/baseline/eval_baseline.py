"""
VLM vs YOLO规则基线 对比评估。

对每个视频按前缀确定"预期行为"，检查 VLM 事件与 Baseline 事件是否命中该行为，
输出对比表 results/report/vlm_vs_baseline.csv 与柱状图。

用法：
    python eval_baseline.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _f in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

ROOT = Path(__file__).resolve().parents[2]
VLM_DIR = ROOT / "results" / "events"
BASE_DIR = ROOT / "results" / "baseline"
REPORT_DIR = ROOT / "results" / "report"

PREFIX2BEHAVIOR = {
    "sleeping": "sleeping", "phone": "looking_phone",
    "looking_around": "looking_around", "talking": "talking",
    "leaving": "away",
}
BEHAVIOR_CN = {
    "sleeping": "睡觉", "looking_phone": "看手机",
    "looking_around": "张望", "talking": "交谈", "away": "离座",
}


def load_events(path: Path) -> list:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def hit(events: list, behavior: str) -> bool:
    return any(e["behavior"] == behavior for e in events)


def main():
    rows = []
    for vlm_f in sorted(VLM_DIR.glob("*.json")):
        video = vlm_f.stem
        prefix = video.rsplit("_", 1)[0]
        expect = PREFIX2BEHAVIOR.get(prefix)
        if expect is None:
            continue
        vlm_events = load_events(vlm_f)
        base_events = load_events(BASE_DIR / f"{video}.json")
        vlm_hit = hit(vlm_events, expect)
        base_hit = hit(base_events, expect)
        rows.append({
            "video": video, "expect": BEHAVIOR_CN[expect],
            "vlm_hit": vlm_hit, "baseline_hit": base_hit,
            "vlm_events": "、".join(
                f"{BEHAVIOR_CN[e['behavior']]}{e['duration']}s"
                for e in vlm_events) or "无",
            "baseline_events": "、".join(
                f"{BEHAVIOR_CN[e['behavior']]}{e['duration']}s"
                for e in base_events) or "无",
        })

    # 汇总：按行为类别
    summary = {}
    for r in rows:
        b = r["expect"]
        s = summary.setdefault(b, {"vlm": 0, "base": 0, "total": 0})
        s["total"] += 1
        s["vlm"] += int(r["vlm_hit"])
        s["base"] += int(r["baseline_hit"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "vlm_vs_baseline.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "video", "expect", "vlm_hit", "baseline_hit",
            "vlm_events", "baseline_events"])
        w.writeheader()
        w.writerows(rows)
        w.writerow({})
        w.writerow({"video": "== 汇总 =="})
        for b in ["睡觉", "看手机", "张望", "交谈", "离座"]:
            s = summary.get(b, {"vlm": 0, "base": 0, "total": 0})
            w.writerow({
                "video": b,
                "vlm_hit": f"{s['vlm']}/{s['total']}",
                "baseline_hit": f"{s['base']}/{s['total']}",
            })

    # 汇总文本
    print("类别         VLM      Baseline")
    for b in ["睡觉", "看手机", "张望", "交谈", "离座"]:
        s = summary.get(b, {"vlm": 0, "base": 0, "total": 0})
        print(f"{b}   {s['vlm']}/{s['total']}    {s['base']}/{s['total']}")
    total_v = sum(s["vlm"] for s in summary.values())
    total_b = sum(s["base"] for s in summary.values())
    total_n = sum(s["total"] for s in summary.values())
    print(f"合计   {total_v}/{total_n}    {total_b}/{total_n}")

    # 柱状图
    cats = ["睡觉", "看手机", "张望", "交谈", "离座"]
    vlm_v = [summary.get(c, {"vlm": 0}).get("vlm", 0) for c in cats]
    base_v = [summary.get(c, {"base": 0}).get("base", 0) for c in cats]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = range(len(cats))
    w = 0.35
    ax.bar([i - w / 2 for i in x], vlm_v, w, label="VLM 零样本",
           color="#8BC8EA", edgecolor="white")
    ax.bar([i + w / 2 for i in x], base_v, w, label="YOLO-pose 规则基线",
           color="#EAA7B2", edgecolor="white")
    for i in x:
        ax.text(i - w / 2, vlm_v[i] + 0.1, str(vlm_v[i]),
                ha="center", fontsize=10)
        ax.text(i + w / 2, base_v[i] + 0.1, str(base_v[i]),
                ha="center", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylabel("识别出事件的视频数（每类共5）")
    ax.set_ylim(0, 6)
    ax.set_title("VLM 零样本 vs YOLO 规则基线：事件级命中对比")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    png = REPORT_DIR / "vlm_vs_baseline.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"对比表: {csv_path}")
    print(f"对比图: {png}")


if __name__ == "__main__":
    main()
