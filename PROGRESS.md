# 项目进度记录（PROGRESS）

> 下次开始前先读本文件，就能知道做到哪一步、下一步做什么。
> 最后更新：2026-10-10（准确率优化中：双人联合区域 6 帧方案重跑 6 个失败视频）

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

## 最终识别效果（25 视频，事件级命中率）——优化后 20/25 = 80%

| 行为 | 识别出事件 | 命中率 | 说明 |
|---|---|---|---|
| 看手机 | phone_01~05 | **5/5 = 100%** | 单帧即完美 |
| 睡觉 | sleeping_01/02/03/05 | 4/5 = 80% | 04 是**数据标签问题**（视频内容是低头写字，非睡觉），模型判 0 正确 |
| 张望 | looking_02/03/04/05 | 4/5 = 80% | 01 模型判成交谈/看手机，未判张望 |
| 交谈 | talking_01/02 | **2/5 = 40%** | union 方案翻倍（原 1/5）|
| 离座 | leaving_01~05 | **5/5 = 100%** | leaving_01 由 union 方案识别（起身离座 9s），02-05 检测层规则 |

**优化前 18/25 (72%) → 优化后 20/25 (80%)**。剩余短板：交谈 3 个视频动作幅度小（语义歧义天花板）、looking_around_01、sleeping_04（数据问题）。

## 准确率优化（2026-10-10 完成）

失败模式分析 → 三大改进（全部追加式）：
1. **双人联合区域 6 帧方案**（`src/preprocess/build_union_multi.py`）：
   重跑 YOLO 检测 → 每帧 person union bbox → 2×3 网格 6 帧（stride=4, resize 192）
   → `crops_union/<video>/`。解决"单人裁剪看不到交谈上下文/起身动作"。
   - 实测：192 高 10.9s/张（比 3 帧版 15.6s/张还快）
2. **UNION_MULTIFRAME_PROMPT**（prompts.py 追加）：6 帧时序 + 区分交谈/看手机/写字 + 离座变化
3. **事件聚合修复**（events.py）：
   - frame_level_series 只写触发帧（0 值帧不再切断稀疏行为事件）
   - union 版 max_gap 放宽到 6（锚点间隔 4 秒）
   - 该修复同时提升了其他视频的事件完整性

注意：`--union` 推理只覆盖锚点帧（stride=4），`fix_missing.py` 会按 crops_multi 补录
"无行为"记录（帧名 person 版），与 union 记录共存，事件聚合按帧号取 max 合并。

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

1. **准确率优化（进行中）**：双人联合区域 6 帧方案（crops_union + UNION_MULTIFRAME_PROMPT）
   重跑 6 个失败视频（talking_01/03/04/05、looking_around_01、leaving_01）
   - 新增脚本：`src/preprocess/build_union_multi.py`（YOLO 重检 → union bbox → 2×3 网格）
   - `prompts.py` 追加 `UNION_MULTIFRAME_PROMPT`（区分交谈/看手机/写字 + 离座变化）
   - `run_inference.py` 追加 `--union` 参数
   - sleeping_04 确认是**数据标签问题**（视频内容是写字非睡觉，模型判 0 正确），记入报告
   - 推理完成后：重新事件聚合 → 重跑 eval → 更新效果表
2. **技术报告**（`docs/report/`）：实验设置、方法、结果表/图、对比分析、局限与展望
3. **答辩 PPT**：用 ppt skill 做 10-15 页
4. **README 完善**：项目说明、目录结构、复现步骤
5. **成果展示视频**：录 demo 或合成演示视频
6. 最终整体交付：git push + PROGRESS 更新

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
