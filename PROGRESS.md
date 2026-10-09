# 项目进度记录（PROGRESS）

> 下次开始前先读本文件，就能知道做到哪一步、下一步做什么。
> 最后更新：2026-10-09（Day 3-4 进行中，推理未完成）

---

## 项目基本信息

- **项目名**：视觉大模型在智慧校园行为智能分析中的应用研究（方向十）
- **场景**：教室/图书馆，识别 5 类行为（睡觉 / 看手机 / 张望 / 交谈 / 离座）
- **仓库**：git@github.com:lingnaikelian/Research-Training-Program.git
- **本地路径**：`E:\科研训练计划`
- **计划文档**：`docs/project_plan.tex`（十日计划，已编译成 PDF）

## 环境与关键配置

- **Python**：3.13.13（系统 Python，非虚拟环境）
- **已装依赖**：openai 3.24.0、python-dotenv、opencv-python 5.0.0、ultralytics、Pillow、matplotlib
- **千帆平台**：学校私有化内网部署，非百度公有云
  - base_url：`http://10.18.18.35:8080/apis/ais-v2`
  - VL 服务 endpoint：**`qwen3-vl-32b-thinking-gms9`**（2026-10-09 重新部署，已验证文本+图片都通）
  - LLM endpoint：**待部署 DeepSeek 后填**（`.env` 里 `QIANFAN_LLM_ENDPOINT` 留空）
- **API Key**：存在 `.env`（已被 .gitignore 排除，不要提交）

## 关键技术决策（踩过的坑，别重犯）

1. **OpenCV 不支持中文路径**：项目路径 `E:\科研训练计划` 含中文，`cv2.imread/imwrite` 静默失败。
   已封装 `src/preprocess/cv_utils.py` 的 `imread_unicode / imwrite_unicode`，所有 OpenCV 读写必须用这俩。
2. **Qwen3-VL 是 Thinking 模型**：思考过程在 `reasoning_content`，`content` 才是答案。
   `max_tokens` 必须给 ≥2000，否则思考没完就截断，content 会是空字符串。
3. **输出 JSON 要容错**：模型可能用 ```json 包裹、前后有废话，`client_qwen_vl.py` 里 `_parse_json` 已处理。
4. **鉴权**：内网服务访问鉴权关闭，但 OpenAI SDK 仍需传非空 api_key（`.env` 里的 sk-xxx 即可）。
5. **部署的服务会变化**：模型重新部署后 endpoint 会变（autogen-dzw7 → qwen3-vl-32b-thinking-gms9），
   服务停了要手动启动，endpoint 变了要更新 `.env`。
6. **后台推理进程可能被中断**：用 `Start-Process` 独立进程跑推理（PID 29632），输出重定向到
   `results/run_log.txt` / `results/run_err.txt`，不要用 run_in_background 跑长推理。

## 已完成

- [x] **Day 1**：千帆客户端 `src/vlm_infer/client_qwen_vl.py`，文本+图片调用验证通过
- [x] **Day 2**：预处理流水线（抽帧 / YOLOv8 检测 / 中文路径修复）
- [x] **Day 3**：数据录制与预处理
      - 用户 25 段视频已重命名放入 `src/data/raw/`
      - 全部抽帧完成（约 1050 帧 @ 1fps）
      - YOLOv8 人体检测裁剪完成（约 1430 张裁剪图）
- [x] **Day 4（部分）**：Prompt 模板 + 批量推理
      - `prompts.py`：5 类行为规则 prompt
      - `run_inference.py`：批量推理（断点续跑）
      - 测试通过：sleeping_01 前 3 张全对，单张约 6.7s
- [x] **Day 6 前置**：`src/aggregate/events.py` 事件聚合+时间轴脚本（已写好未运行）

## 当前状态（2026-10-09 收尾时）

- **批量推理进行中：18/25 完成**，独立进程 PID 29632 在跑（21:39 重启后），
  输出在 `results/run_log.txt`，错误在 `results/run_err.txt`
- 已完成的 18 个视频结果在 `results/json/<video>.jsonl`
- **明天第一步**：检查进程是否还在 / 是否跑完 25 个
  - 若进程没了：跑 `python src/vlm_infer/run_inference.py --all` 断点续跑（会跳过已完成）
  - 若已完成：检查 25 个 jsonl 齐全
- 注意：`results/` 在 .gitignore 里，推理结果不上传 Git（数据量大），代码才上传

## 下一步（Day 4 收尾 → Day 5/6/7）

1. 等推理全部完成（25/25），检查识别质量
2. 跑事件聚合：`python src/aggregate/events.py --all` → `results/events/*.json` + 时间轴图
3. 分析每类行为识别准确率（对照视频标注：sleeping_01 应识别为睡觉等）
4. Day 7：YOLO baseline 对比（`src/baseline/yolo_rules.py` 待写）
5. Day 8：Streamlit demo（`src/app/demo.py` 待写）
6. Day 9-10：报告与收尾

## 常用命令

```powershell
# 抽帧
cd src\preprocess; python extract_frames.py ..\data\raw\<视频名>.mp4 --fps 1

# 人体检测裁剪（不带参数 = 处理 frames 下所有）
python detect_persons.py

# 批量推理（断点续跑）
cd ..\vlm_infer; python run_inference.py --all
python run_inference.py <视频名> --limit 10     # 测试前10张

# 事件聚合
cd ..\aggregate; python events.py --all

# 测试 VLM 连通性
cd ..\vlm_infer; python test_connection.py

# Git 提交
cd E:\科研训练计划; git add .; git commit -m "feat: DayX - ..."; git push
```
