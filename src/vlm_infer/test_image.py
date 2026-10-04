"""
Day 1 图片通路测试：
1. 生成一张带形状和文字的测试图
2. 调 Qwen3-VL 描述图中内容
"""
from pathlib import Path
from PIL import Image, ImageDraw
from client_qwen_vl import QwenVLClient

# --- Step 1: 生成测试图 ---
out_dir = Path(__file__).resolve().parents[2] / "src" / "data"
out_dir.mkdir(parents=True, exist_ok=True)
img_path = out_dir / "test_card.png"

img = Image.new("RGB", (640, 400), color=(100, 149, 237))  # 蓝色背景
draw = ImageDraw.Draw(img)
draw.ellipse([220, 80, 420, 280], fill=(220, 50, 50))       # 红色圆
draw.text((250, 320), "HELLO 2026", fill=(255, 255, 255))    # 白色文字
img.save(img_path)
print(f"[1] 测试图已生成: {img_path}")

# --- Step 2: 调 Qwen3-VL ---
c = QwenVLClient()
print("[2] 发送给 Qwen3-VL...")
prompt = (
    "请观察这张图，然后严格按 JSON 输出："
    '{"background_color": "主背景颜色", "main_shape": "主要形状", '
    '"text": "图中文字内容"}'
)
result = c.analyze_image(str(img_path), prompt, expect_json=True)
print("[3] 模型返回：")
import json
print(json.dumps(result, ensure_ascii=False, indent=2))
