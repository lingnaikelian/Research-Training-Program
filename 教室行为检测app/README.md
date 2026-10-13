# 教室行为智能分析应用

基于千帆平台 **Qwen3-VL-32B-Thinking** 视觉大模型的教室监控视频行为识别应用，
零样本识别 5 类行为：睡觉 / 看手机 / 张望 / 交谈 / 离座。

## 环境要求

- Windows / Linux / macOS
- Python 3.9 及以上
- 一个千帆平台账号（部署了 Qwen3-VL-32B-Thinking 模型的团队共用）

## 快速开始

```bash
python setup.py    # 检测并安装依赖、检查权重、创建运行目录
python start.py    # 启动应用，浏览器打开 http://localhost:8501
```

`setup.py` 会自动完成环境检测与依赖安装（openai、opencv、ultralytics、streamlit、
matplotlib 等），无需手动处理；`start.py` 负责启动应用，可用 `--port` 指定端口。

## 配置模型接口

上传识别前，把 `.env.example` 复制为 `.env`，填入你们团队在千帆平台的部署信息：

```
QIANFAN_BASE_URL=<千帆私有化部署地址>
QIANFAN_API_KEY=<API Key>
QIANFAN_VL_ENDPOINT=<Qwen3-VL-32B-Thinking 的 endpoint id>
```

## 使用方法

打开应用后：

1. **上传视频识别**：上传一段教室监控视频（建议 10~45 秒、单人或双人入镜、
   视角稳定），点击「开始识别」，等待完整管线运行：
   抽帧（1fps）→ 行人检测 → 多帧拼接 → VLM 零样本推理 → 事件聚合，
   页面展示识别结论、事件明细、行为时间轴与逐帧判断分布。
   识别耗时约 3~6 分钟（多帧图逐张调用大模型，随服务端负载波动）。
2. **查看已有结果**：上传识别完成后，结果会自动存档到 `results/`，
   可在此模式复看（本提交包不含预置结果数据，空目录下会提示先上传识别）。

## 目录结构

```
├── src/
│   ├── preprocess/   # 抽帧、YOLO 行人检测、多帧拼接
│   ├── vlm_infer/    # Qwen3-VL 客户端、Prompt、批量推理
│   ├── aggregate/    # 事件聚合 + 时间轴图
│   └── app/          # Streamlit 应用（demo.py）
├── .env.example      # 模型接口配置模板
├── setup.py          # 环境检测与依赖安装
├── start.py          # 应用启动脚本
└── src/preprocess/   # 两个模型权重均在此：yolov8n.pt（行人检测）、yolov8n-pose.pt（姿态估计）
```

## 代码说明

- `src/preprocess/extract_frames.py`：按 1fps 抽取视频帧
- `src/preprocess/detect_persons.py`：YOLOv8n 行人检测，输出各帧裁剪图
- `src/preprocess/build_union_multi.py`：连续 6 时刻行人联合区域拼成 2×3 网格，
  让 VLM 同时看到头部转动、嘴部开合、手部动作、坐姿转移动等时序信号
- `src/vlm_infer/client_qwen_vl.py`：千帆 Qwen3-VL 接口客户端
- `src/vlm_infer/prompts.py`：5 类行为的零样本判断指令
- `src/vlm_infer/run_inference.py`：对每个时刻的多帧图调用 VLM，输出逐帧判断
- `src/aggregate/events.py`：逐帧判断按时间连续性聚合为事件（持续行为 ≥5s、
  瞬态行为 ≥3s 记为事件，离座补充检测层规则），并生成行为时间轴图
- `src/app/demo.py`：Streamlit 界面（上传识别 / 查看结果）

## 复现实验（可选）

如需对一批视频做批量评估，可按顺序执行：

```bash
python src/preprocess/extract_frames.py <video.mp4> --fps 1
python src/preprocess/detect_persons.py src/data/frames/<name>
python src/preprocess/build_union_multi.py <name> --stride 4 --resize 192
python src/vlm_infer/run_inference.py <name> --union --simple
python src/aggregate/events.py <name>
```

各脚本支持 `--all` 批量处理，推理脚本支持断点续跑。
