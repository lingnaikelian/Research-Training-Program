"""
千帆平台 Qwen3-VL 客户端封装。

用法：
    from client_qwen_vl import QwenVLClient
    client = QwenVLClient()
    result = client.analyze_image("test.jpg", "这张图里有几个人？")
"""

import os
import base64
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# 加载项目根目录下的 .env
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


class QwenVLClient:
    def __init__(self):
        self.api_key = os.getenv("QIANFAN_API_KEY")
        self.vl_endpoint = os.getenv("QIANFAN_VL_ENDPOINT")
        self.llm_endpoint = os.getenv("QIANFAN_LLM_ENDPOINT")

        if not self.api_key:
            raise RuntimeError("未设置 QIANFAN_API_KEY，请检查 .env 文件")

        # 千帆 v2 API 兼容 OpenAI SDK
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://qianfan.baidubce.com/v2",
        )

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """把本地图片编码为 base64 data URI。"""
        p = Path(image_path)
        ext = p.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "bmp": "bmp", "webp": "webp"}.get(ext, "jpeg")
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:image/{mime};base64,{b64}"

    def analyze_image(self, image_path: str, prompt: str,
                      expect_json: bool = True) -> dict | str:
        """
        传一张本地图片 + 一段 prompt 给 Qwen3-VL，返回模型输出。

        Args:
            image_path: 本地图片路径
            prompt: 给模型的指令
            expect_json: 是否要求模型输出 JSON 并自动解析

        Returns:
            expect_json=True 时返回 dict；否则返回原始字符串
        """
        if not self.vl_endpoint:
            raise RuntimeError(
                "未设置 QIANFAN_VL_ENDPOINT。请先到千帆控制台部署 "
                "Qwen3-VL-32B-Thinking，把 endpoint id (ep-xxxx) 填入 .env"
            )

        data_uri = self._encode_image(image_path)
        resp = self.client.chat.completions.create(
            model=self.vl_endpoint,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.1,  # 行为识别要稳定，调低温度
        )
        text = resp.choices[0].message.content.strip()

        if expect_json:
            return self._parse_json(text)
        return text

    def chat_text(self, prompt: str) -> str:
        """纯文本对话（用 LLM endpoint，Day 6 事件聚合时用）。"""
        if not self.llm_endpoint:
            raise RuntimeError("未设置 QIANFAN_LLM_ENDPOINT")
        resp = self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _parse_json(text: str) -> dict:
        """从模型输出里抠出 JSON 对象，容忍 ```json 包裹和前后废话。"""
        # 去掉 ```json ... ``` 包裹
        if "```" in text:
            parts = text.split("```")
            for seg in parts:
                seg = seg.strip()
                if seg.startswith("json"):
                    seg = seg[4:].strip()
                if seg.startswith("{"):
                    text = seg
                    break
        # 找到第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"模型输出里没有 JSON：{text[:200]}")
        return json.loads(text[start:end + 1])


if __name__ == "__main__":
    # 直接运行此文件做连通性测试
    c = QwenVLClient()
    print(f"[OK] 客户端初始化成功，API Key 已加载")
    print(f"     VL Endpoint: {c.vl_endpoint or '(未设置)'}")
    print(f"     LLM Endpoint: {c.llm_endpoint or '(未设置)'}")
