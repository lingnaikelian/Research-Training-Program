# 项目进度记录（PROGRESS）

> 下次开始前先读本文件，就能知道做到哪一步、下一步做什么。
> 最后更新：2026-10-04（Day 2 结束）

---

## 项目基本信息

- **项目名**：视觉大模型在智慧校园行为智能分析中的应用研究（方向十）
- **场景**：教室/图书馆，识别 5 类行为（睡觉 / 看手机 / 张望 / 交谈 / 离座）
- **仓库**：git@github.com:lingnaikelian/Research-Training-Program.git
- **本地路径**：`E:\科研训练计划`
- **计划文档**：`docs/project_plan.tex`（十日计划，已编译成 PDF）

## 环境与关键配置

- **Python**：3.13.13（系统 Python，非虚拟环境）
- **已装依赖**：openai 3.24.0、python-dotenv、opencv-python 5.0.0、ultralytics、Pillow
- **千帆平台**：学校私有化内网部署，非百度公有云
  - base_url：`http://10.18.18.35:8080/apis/ais-v2`
  - VL 服务 endpoint：`autogen-dzw7`（Qwen3-VL-32B-Thinking）
  - LLM endpoint：**待部署 DeepSeek 后填**（`.env` 里 `QIANFAN_LLM_ENDPOINT` 留空）
- **API Key**：存在 `.env`（已被 .gitignore 排除，不要提交）

## 关键技术决策（踩过的坑，别重犯）

1. **OpenCV 不支持中文路径**：项目路径 `E:\科研训练计划` 含中文，`cv2.imread/imwrite` 静默失败。
   已封装 `src/preprocess/cv_utils.py` 的 `imread_unicode / imwrite_unicode`，所有 OpenCV 读写必须用这俩。
2. **Qwen3-VL 是 Thinking 模型**：思考过程在 `reasoning_content`，`content` 才是答案。
   `max_tokens` 必须给 ≥2000，否则思考没完就截断，content 会是空字符串。
3. **输出 JSON 要容错**：模型可能用 ```json 包裹、前后有废话，`client_qwen_vl.py` 里 `_parse_json` 已处理。
4. **鉴权**：内网服务访问鉴权关闭，但 OpenAI SDK 仍需传非空 api_key（`.env` 里的 sk-xxx 即可）。

## 已完成

- [x] **Day 1**：千帆客户端 `src/vlm_infer/client_qwen_vl.py`
      - 文本调用 ✅
      - 图片调用 ✅（准确识别颜色/形状/文字）
      - JSON 输出解析 ✅
- [x] **Day 2**：预处理流水线
      - `src/preprocess/extract_frames.py`（按 fps 抽帧）
      - `src/preprocess/detect_persons.py`（YOLOv8n 检测 person + 裁剪加 padding）
      - `src/preprocess/cv_utils.py`（中文路径修复）
      - YOLOv8 自带 bus.jpg 验证：正确识别 4 person

## 当前状态

- 测试视频 `src/data/raw/test_clip.mp4` 已跑通全流程（抽帧 → 检测）
- **真实教室视频尚未开始录制**
- LLM endpoint 尚未部署

## 下一步（Day 3 → Day 4）

### Day 3（用户行动）
用户去空教室录制 25 段视频（5 类行为 × 5 段，每段 30-60 秒），命名：
`sleeping_01.mp4 ... looking_phone_01.mp4 ...` 共 25 个，
放到 `src/data/raw/`。

### Day 4（AI 行动，用户录视频时并行）
1. 写 `src/vlm_infer/prompts.py`：行为规则 prompt 模板，要求模型输出严格 JSON：
   `{sleeping, looking_phone, looking_around, talking, away, reason}`
2. 写 `src/vlm_infer/run_inference.py`：遍历 `crops/` 批量推理，结果存 `results/json/<video>.jsonl`
3. 用户视频到齐后，跑通一段视频验证

### 后续
- Day 5 批量推理
- Day 6 事件聚合 `src/aggregate/events.py`（连续 5 秒同标签才算事件）
- Day 7 YOLO baseline 对比
- Day 8 Streamlit demo
- Day 9-10 报告与收尾

## 常用命令

```powershell
# 抽帧
cd src\preprocess; python extract_frames.py ..\data\raw\<视频名>.mp4 --fps 1

# 人体检测裁剪
python detect_persons.py ..\data\frames\<视频名>

# 测试 VLM 连通性
cd ..\vlm_infer; python test_connection.py

# Git 提交
cd E:\科研训练计划; git add .; git commit -m "feat: DayX - ..."; git push
```
