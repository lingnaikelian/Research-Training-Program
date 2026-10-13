"""环境检测与依赖安装。

用法：python setup.py

- 检查 Python 版本
- 逐项检查运行所需的依赖，缺失的自动安装
- 检查模型权重文件是否存在
- 创建运行所需的目录结构（src/data、results 等）
"""
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 运行 app 所需的依赖（包名 与 import 名不一致的做映射）
DEPENDENCIES = [
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    "opencv-python>=4.8.0",
    "ultralytics>=8.1.0",
    "streamlit>=1.30.0",
    "matplotlib>=3.8.0",
    "pyyaml>=6.0",
]
IMPORT_NAMES = {
    "python-dotenv": "dotenv",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
}

# 运行时需要的目录（上传识别与结果展示会读写）
NEED_DIRS = [
    "src/data/frames",
    "src/data/raw",
    "src/data/crops",
    "src/data/crops_union",
    "results/json",
    "results/events",
    "results/report",
    "results/baseline",
]

WEIGHTS = [
    ("src/preprocess/yolov8n.pt", "src/preprocess/"),
    ("src/preprocess/yolov8n-pose.pt", "src/preprocess/"),
]


def check_python():
    if sys.version_info < (3, 9):
        print(f"[警告] 建议 Python >= 3.9，当前 {sys.version.split()[0]}")
    else:
        print(f"[OK] Python {sys.version.split()[0]}")


def is_installed(pkg: str) -> bool:
    raw = pkg
    for sep in ("==", ">=", "<=", ">", "<", "~=", "!="):
        raw = raw.split(sep)[0]
    raw = raw.strip()
    name = IMPORT_NAMES.get(raw, raw).replace("-", "_")
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def install(pkg: str):
    print(f"[安装] {pkg}")
    subprocess.run([sys.executable, "-m", "pip", "install", pkg],
                   check=False)


def ensure_dirs():
    for d in NEED_DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("[OK] 目录结构就绪")


def check_weights():
    for path, where in WEIGHTS:
        if (ROOT / path).exists():
            print(f"[OK] {path}")
        else:
            print(f"[缺失] {path}（应放在{where}）"
                  f"—— yolov8n.pt 缺失时 ultralytics 首次运行会自动下载；"
                  f"yolov8n-pose.pt 需手动获取")


def main():
    print("== 环境检测 ==")
    check_python()

    print("\n== 依赖检查 ==")
    need = [p for p in DEPENDENCIES if not is_installed(p)]
    if need:
        print(f"以下依赖未安装，开始安装：{', '.join(need)}")
        for p in need:
            install(p)
    else:
        print("全部依赖已安装")

    print("\n== 权重检查 ==")
    check_weights()

    print("\n== 目录检查 ==")
    ensure_dirs()

    print("\n完成。运行 python start.py 启动演示应用。")


if __name__ == "__main__":
    main()
