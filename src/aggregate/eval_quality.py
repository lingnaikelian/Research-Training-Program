"""
识别质量初步分析：统计每个视频各行为的帧命中情况，对照预期行为。

用法：python eval_quality.py
输出：每个视频的"预期行为 vs 识别最多行为"对照表
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "results" / "json"

BEHAVIORS = ["sleeping", "looking_phone", "looking_around", "talking", "away"]
BEHAVIOR_CN = {
    "sleeping": "睡觉", "looking_phone": "看手机",
    "looking_around": "张望", "talking": "交谈", "away": "离座",
}


def expected_behavior(video: str) -> str:
    """视频名前缀即预期行为：sleeping_01 -> sleeping，leaving_01 -> away"""
    prefix = video.rsplit("_", 1)[0]
    return {"leaving": "away", "phone": "looking_phone"}.get(prefix, prefix)


def main():
    rows = []
    for p in sorted(JSON_DIR.glob("*.jsonl")):
        video = p.stem
        expected = expected_behavior(video)
        counts = defaultdict(int)
        total = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            for b in BEHAVIORS:
                if int(rec.get(b, 0) or 0) == 1:
                    counts[b] += 1
        top = max(counts, key=counts.get) if counts else "none"
        top_ratio = counts[top] / total if total else 0
        rows.append((video, expected, total, counts, top, top_ratio))

    header = f"{'视频':<18}{'预期':<8}{'帧数':<6}{'识别最多':<10}{'占比':<7}其他命中"
    print(header)
    ok = 0
    for video, expected, total, counts, top, ratio in rows:
        others = " ".join(
            f"{BEHAVIOR_CN[b]}={counts[b]}" for b in BEHAVIORS
            if counts[b] and b != top
        )
        match = "OK" if top == expected else "??"
        if top == expected:
            ok += 1
        top_cn = BEHAVIOR_CN.get(top, "无行为")
        print(f"{video:<18}{BEHAVIOR_CN[expected]:<8}{total:<6}"
              f"{top_cn:<10}{ratio:.0%}  {match}  {others}")
    print(f"\n按'识别最多的行为=预期行为'计，{ok}/{len(rows)} 个视频匹配")


if __name__ == "__main__":
    main()
