"""
Streamlit 演示应用：教室行为智能分析（视觉大模型方案）

展示 25 段自录视频的零样本行为识别结果：
- 左侧选择视频，右侧显示事件摘要、行为时间轴、关键帧证据
- 全部读取 results/ 下已有结果，不实时调用 API（演示用）

启动：
    streamlit run src/app/demo.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EVENT_DIR = ROOT / "results" / "events"
REPORT_DIR = ROOT / "results" / "report"
FRAME_DIR = ROOT / "src" / "data" / "frames"

BEHAVIOR_CN = {
    "sleeping": "睡觉", "looking_phone": "看手机",
    "looking_around": "张望", "talking": "交谈", "away": "离座",
}
PREFIX2BEHAVIOR = {
    "sleeping": "睡觉", "phone": "看手机",
    "looking_around": "张望", "talking": "交谈", "leaving": "离座",
}
COLORS = {
    "sleeping": "#8BC8EA", "looking_phone": "#9BBBF4",
    "looking_around": "#94D8C3", "talking": "#E4D48F", "away": "#EAA7B2",
}

# ---------- 页面样式（清新淡雅） ----------
st.set_page_config(page_title="智慧校园行为分析 Demo", page_icon="🎓",
                   layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1 { color: #2E5B7F; }
    h2, h3 { color: #3A6B96; }
    .exp-box {
        background: linear-gradient(135deg, #F0F7FC, #E8F3EC);
        border-radius: 12px; padding: 14px 18px; margin: 8px 0;
    }
    .evt-chip {
        display: inline-block; border-radius: 20px;
        padding: 4px 14px; margin: 2px 6px 2px 0;
        color: #fff; font-size: 14px; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 教室行为智能分析（视觉大模型零样本方案）")
st.caption("Qwen3-VL-32B-Thinking · 千帆私有化部署 · 25 段自录教室视频 · 5 类行为")

# ---------- 数据加载 ----------
videos = sorted(p.stem for p in EVENT_DIR.glob("*.json"))

# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown("### 📼 选择视频")
    video = st.selectbox("视频", videos, label_visibility="collapsed")
    prefix = video.rsplit("_", 1)[0]
    expect = PREFIX2BEHAVIOR.get(prefix, "未知")
    st.markdown(
        f"**预期行为**：<span style='color:#2E5B7F;'>{expect}</span>",
        unsafe_allow_html=True)

    events = json.loads((EVENT_DIR / f"{video}.json").read_text(encoding="utf-8"))
    st.markdown("**识别事件**")
    if events:
        for e in events:
            color = COLORS.get(e["behavior"], "#999")
            dur = e.get("duration", 0)
            st.markdown(
                f"<span class='evt-chip' style='background:{color};'>"
                f"{BEHAVIOR_CN[e['behavior']]} {dur}s</span>",
                unsafe_allow_html=True)
    else:
        st.markdown("无事件")

    st.divider()
    st.markdown("**识别流程**")
    st.markdown("1. 抽帧 (1fps)\n2. YOLOv8 行人检测裁剪\n"
                "3. 多帧拼接（时序行为）\n4. VLM 零样本判断\n5. 事件聚合")

# ---------- 主区 ----------
c1, c2 = st.columns([1.3, 2])

with c1:
    st.markdown("### 📊 事件摘要")
    if events:
        rows = [
            {"行为": BEHAVIOR_CN[e["behavior"]],
             "时段": f"{e['start']}-{e['end']}s",
             "时长": f"{e['duration']}s",
             "置信度": e.get("confidence", "-"),
             "来源": "检测层" if e.get("source") == "detector" else "VLM"}
            for e in events
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("该视频未识别出任何行为事件。")

    st.markdown("### 🖼️ 关键帧证据")
    fdir = FRAME_DIR / video
    if fdir.exists():
        fps = sorted(fdir.glob("frame_*.jpg"))
        sample = [fps[i] for i in
                  range(0, len(fps), max(1, len(fps) // 6))][:6]
        cols = st.columns(3)
        for i, fp in enumerate(sample):
            with cols[i % 3]:
                img = Image.open(str(fp))
                st.image(img, caption=f"t={i * 7:.0f}s 附近", width=180)
    else:
        st.warning("未找到抽帧目录")

with c2:
    st.markdown("### ⏱️ 行为时间轴")
    tpath = REPORT_DIR / f"{video}_timeline.png"
    if tpath.exists():
        st.image(str(tpath), use_container_width=True)
    else:
        # 无事件时画空时间轴
        fig, ax = plt.subplots(figsize=(10, 3.2))
        rows_y = {b: i for i, b in enumerate(BEHAVIOR_CN.keys())}
        ax.set_yticks(list(rows_y.values()))
        ax.set_yticklabels(list(BEHAVIOR_CN.values()))
        ax.set_xlabel("时间（秒）")
        ax.set_xlim(0, 45)
        ax.set_title(f"{video} 行为时间轴（无事件）")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)

st.divider()
with st.expander("📄 技术方案说明"):
    st.markdown("""
**任务**：对教室监控视频进行 5 类行为识别（睡觉 / 看手机 / 张望 / 交谈 / 离座）。

**方案**：采用视觉大模型零样本识别，不训练任何分类器——
1. **预处理**：1fps 抽帧 → YOLOv8n 行人检测，裁剪出每个人；
2. **静态行为**（看手机/睡觉）：单帧裁剪图直接送入 VLM；
3. **时序行为**（张望/交谈/离座）：3 帧横排拼接 + 专用提示词，让 VLM 观察
   头部转动、嘴部开合、坐姿转移动等时序信号；
4. **事件聚合**：对逐帧判断做时间连续性合并（≥5s 记为事件），
   离座补充检测层规则（行人目标消失即判定）。

**模型**：Qwen3-VL-32B-Thinking（学校千帆平台私有化部署）。
**对比实验**：YOLOv8-pose 姿态规则基线，见 `results/report/vlm_vs_baseline.png`。
""")
