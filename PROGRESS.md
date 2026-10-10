# 项目进度记录（PROGRESS）

> 下次开始前先读本文件，就能知道做到哪一步、下一步做什么。
> 最后更新：2026-10-10（Day 4-6 完成：单帧+多帧推理、事件聚合、时间轴图）

---

## 项目基本信息

- **项目名**：视觉大模型在智慧校园行为智能分析中的应用研究（方向十）
- **场景**：教室/图书馆，识别 5 类行为（睡觉 / 看手机 / 张望 / 交谈 / 离座）
- **仓库**：git@github.com:lingnaikelian/Research-Training-Program.git
- **本地路径**：`E:\科研训练计划`
- **计划文档**：`docs/project_plan.tex`

## 环境与关键配置

- **Python**：3.13.13（系统 Python）
- **已装依赖**：openai、python-dotenv、opencv-python、ultralytics、Pillow、matplotlib
- **千帆平台**：学校私有化内网
  - base_url：`http://10.18.18.35:8080/apis/ais-v2`
  - VL endpoint：`qwen3-vl-32b-thinking-gms9`
  - LLM endpoint：未部署
- **API Key**：`.env`（.gitignore 排除）

## 关键技术决策（踩过的坑）

1. **OpenCV 中文路径**：用 `cv_utils.py` 的 `imread_unicode/imwrite_unicode`
2. **Thinking 模型**：`max_tokens>=2000`，答案在 `content`，思考在 `reasoning_content`
3. **JSON 容错**：`_parse_json` 处理 ```json 包裹和废话
4. **长推理用 Start-Process 独立进程**，别用 run_in_background（会被中断）
5. **单帧 vs 多帧**：静态行为（看手机/睡觉）单帧足够；时序行为（张望/交谈/离座）
   必须多帧拼接（`make_multiframe.py`，3帧横排 + MULTIFRAME_PROMPT）
6. **重跑多帧前要删旧 jsonl**（断点续跑会跳过已存在的 crop 名）
7. **失败帧处理**：`fix_missing.py` 把模型没输出 JSON 的帧补"无行为"记录
8. **matplotlib 中文**：rcParams 设 Microsoft YaHei

## 数据与流程现状

- 25 段视频（5 行为 × 5 段）在 `src/data/raw/`（已 gitignore）
- 抽帧 1fps → `src/data/frames/`；YOLOv8 裁剪 → `src/data/crops/`
- **推理结果**：`results/json/<video>.jsonl`
  - 看手机/睡觉 10 个：单帧版（效果好）
  - 交谈/张望/离座 15 个：多帧版（stride=3，拼接图在 `crops_multi/`）
- **事件聚合**：`results/events/<video>.json` + `results/report/<video>_timeline.png`

## 最终识别效果（25 视频，事件级命中率）

| 行为 | 识别出事件 | 命中率 | 说明 |
|---|---|---|---|
| 看手机 | phone_01~05 | **5/5 = 100%** | 单帧即完美 |
| 睡觉 | sleeping_01/02/03/05 | **4/5 = 80%** | 04 无事件（可能没裁到人）|
| 张望 | looking_02/03/04/05 | **4/5 = 80%** | 多帧修复后大幅提升（原 0/5）|
| 交谈 | talking_02 | 1/5 = 20% | 仍难：嘴动幅度小/单人裁剪 |
| 离座 | 全部无事件 | 0/5 ❌ | 最大问题，见下方方案 |

## 待解决：离座识别（0/5）

**根因**：人离座后画面变空/人走出画面，YOLO 检测不到 person → 没有裁剪图 → VLM 没机会判 away。
leaving_02~05 只有 3-17 张裁剪图，leaving_01 前段在玩手机被识别为"看手机"。

**方案（检测层规则，不花 API）**：
在 `events.py` 加离座判定：遍历 `crops/<video>/`，按帧统计 person 数量，
若某帧 person 数比前几帧骤降（如从 ≥1 变 0，或明显减少），标记 away 事件。
本质：离座 = "检测目标消失"，属于目标检测层的信号，比让 VLM 看裁剪图更可靠。

## 下一步（Day 7-8）

1. **离座检测层规则**：改 `events.py` 加 person 数骤降判定 → leaving 应能出事件
2. **Day 7 YOLO baseline 对比**：`src/baseline/yolo_rules.py`（待写）
   用 YOLO+姿态规则在同样视频上识别，与 VLM 结果对比准确率
3. **Day 8 Streamlit demo**：`src/app/demo.py`（待写）
   上传视频 → 行为时间轴 → 事件报告
4. **Day 9-10**：技术报告 + PPT + README 完善

## 常用命令

```powershell
# 推理（单帧）
cd src\vlm_infer; python run_inference.py --all
# 推理（多帧重跑，先删旧 jsonl）
python run_inference.py <视频> --multiframe --force
# 生成多帧拼接图
cd ..\preprocess; python make_multiframe.py <视频...> --stride 3 --resize 150
# 补全失败帧
cd ..\aggregate; python fix_missing.py
# 事件聚合+时间轴
python events.py --all
# 识别质量统计
python eval_quality.py
```
