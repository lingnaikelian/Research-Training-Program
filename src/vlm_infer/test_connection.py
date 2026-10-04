"""打印完整响应结构，定位内容字段。"""
from client_qwen_vl import QwenVLClient

c = QwenVLClient()
resp = c.client.chat.completions.create(
    model=c.vl_endpoint,
    messages=[{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    temperature=0.1,
    max_tokens=500,
)
print("=== 完整响应 ===")
print(resp.model_dump_json(indent=2))
