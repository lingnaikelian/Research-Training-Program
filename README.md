# Research-Training-Program

方向十：视觉大模型在智慧校园行为智能分析中的应用研究

中国石油大学（北京）计算机科学与技术 2023 级本科生科研训练项目。

## 项目目标

聚焦**教室/图书馆场景**，使用百度千帆平台上的 **Qwen3-VL-32B-Thinking** 视觉大模型，
以自然语言描述行为规则的方式零样本识别 5 类行为：

1. 长时间趴桌睡觉
2. 持续看手机
3. 频繁左右回头张望
4. 两人及以上持续交谈
5. 离座长时间未归

## 十日计划

详见 [`docs/project_plan.tex`](docs/project_plan.tex)。

编译方式（任选其一）：

- **Overleaf（推荐）**：新建项目，上传 `docs/project_plan.tex`，点 Recompile 即可；
- **本地**：安装 TeX Live 后执行 `xelatex project_plan.tex`（中文需要 XeLaTeX + ctex）。

## 目录结构

```
├── docs/           # 项目文档（计划、报告）
├── src/            # 源代码
│   ├── preprocess/ # 视频抽帧 + 人体检测
│   ├── vlm_infer/  # 调用 Qwen3-VL
│   ├── aggregate/  # 事件聚合
│   ├── baseline/    # YOLO 对比方法
│   └── app/        # Streamlit 演示
└── results/        # 推理结果与报告
```

## 推进方式

按 `docs/project_plan.tex` 中的 Day 1 → Day 10 逐日执行，
每完成一天任务 commit 一次。
