"""列出 25 视频的识别命中情况（只读，不写文件）。"""
import json
from pathlib import Path

EV = Path("results/events")
P2E_CN = {"sleeping": "睡觉", "phone": "看手机", "looking_around": "张望",
          "talking": "交谈", "leaving": "离座"}
CN2EN = {"睡觉": "sleeping", "看手机": "looking_phone", "张望": "looking_around",
         "交谈": "talking", "离座": "away"}

miss = []
for p in sorted(EV.glob("*.json")):
    v = p.stem
    prefix = v.rsplit("_", 1)[0]
    expect_cn = P2E_CN.get(prefix)
    if expect_cn is None:
        continue
    ev = json.loads(p.read_text(encoding="utf-8"))
    hit = any(e["behavior"] == CN2EN[expect_cn] for e in ev)
    desc = "、".join(f"{e['behavior']}{e.get('duration', 0)}s" for e in ev) or "无事件"
    print(f"{'OK' if hit else 'MISS'} {v}  预期[{expect_cn}]  识别[{desc}]")
    if not hit:
        miss.append((v, expect_cn, desc))

print()
print(f"未命中 {len(miss)} 个:")
for v, e, d in miss:
    print(f"  {v}  预期[{e}]  实际[{d}]")
