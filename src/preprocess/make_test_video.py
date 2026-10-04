"""生成一段 10 秒测试视频：蓝背景上移动的红方块，用来验证抽帧脚本。"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "src" / "data" / "raw" / "test_clip.mp4"
out.parent.mkdir(parents=True, exist_ok=True)

fps = 10
size = (640, 400)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(str(out), fourcc, fps, size)

for i in range(100):  # 10 秒
    frame = np.full((400, 640, 3), (200, 200, 200), dtype=np.uint8)
    x = 50 + i * 5
    cv2.rectangle(frame, (x, 150), (x + 80, 250), (40, 40, 220), -1)
    cv2.putText(frame, f"frame {i}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    vw.write(frame)
vw.release()
print(f"[OK] 测试视频已生成: {out}")
