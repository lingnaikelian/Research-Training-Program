"""启动教室行为智能分析演示应用。

用法：python start.py [--port 8501]

- 检查 .env 是否存在（缺失时提示；浏览结果不需要 .env，上传识别需要）
- 启动 Streamlit，浏览器访问 http://localhost:<port>
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "src" / "app" / "demo.py"


def main():
    ap = argparse.ArgumentParser(description="启动教室行为智能分析演示应用")
    ap.add_argument("--port", type=int, default=8501, help="端口号，默认 8501")
    args = ap.parse_args()

    if not (ROOT / ".env").exists():
        print("[提示] 未找到 .env 文件。")
        print("       - 上传识别前需先复制 .env.example 为 .env，"
              "填入千帆平台的 Base URL / API Key / Endpoint")
        print()

    print(f"[启动] http://localhost:{args.port}")
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(DEMO), "--server.port", str(args.port),
                    "--server.headless", "true"])


if __name__ == "__main__":
    main()
