# 项目进度记录（PROGRESS）

> 下次开始前先读本文件，就能知道做到哪一步、下一步做什么。
> 最后更新：2026-10-10（Day 7 完成：离座检测层规则 + YOLO 对比实验 + Streamlit demo）

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
| 离座 | leaving_02~05 | **4/5 = 80%** | 检测层规则救回（原 0/5）|

**合计 18/25 = 72%**。leaving_01（人全程在画面内站立离座）为模型局限。

## 离座检测层规则（已完成）

原理：YOLO person 在视频尾部消失 >=5 帧 = 人走出画面 = 离座信号。
`events.py` 新增 `detector_away_events()`：VLM 未判 away 且裁剪图尾部缺失时自动补离座事件。
leaving_02~05 尾部缺失 27~33 帧 → 全部识别出离座；其余 20 视频全覆盖不误报。

## 对比实验（Day 7 完成）

`src/baseline/yolo_rules.py`：YOLOv8n-pose 姿态规则基线（阈值固定不调优）：
- away：person 消失段；sleeping：关键点置信度<0.55；looking_phone：鼻尖低于肩线 5% 图像高
- looking_around / talking：姿态规则无能力，判 0

| 类别 | VLM 零样本 | YOLO 规则基线 |
|---|---|---|
| 睡觉 | 4/5 | 4/5（事件分裂、有误报）|
| 看手机 | **5/5** | 1/5 |
| 张望 | **4/5** | 0/5 |
| 交谈 | 1/5 | 0/5 |
| 离座 | 4/5 | 4/5 |
| **合计** | **18/25** | **9/25** |

结论：VLM 在看手机/张望/交谈三类语义行为上碾压姿态规则基线；
离座/睡觉两类两者均可做（检测层/姿态信号强），但 VLM 事件质量更高（不分裂、无误报）。
对比表/图：`results/report/vlm_vs_baseline.csv` + `.png`。

## Streamlit Demo（Day 8 完成，已验证可启动）

`src/app/demo.py`：左侧选视频 → 事件摘要表 + 行为时间轴图 + 关键帧证据 + 方案说明。
清新淡雅配色。启动：`streamlit run src/app/demo.py`（已验证 HTTP 200）。

## 下一步（Day 9-10）

1. **技术报告**（`docs/report/`）：实验设置、方法、结果表/图、对比分析、局限与展望
2. **答辩 PPT**：用 ppt skill 做 8-10 页
3. **README 完善**：项目说明、目录结构、复现步骤
4. 最终整体交付：git push + PROGRESS 更新

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
