"""
Streamlit 演示应用：教室行为智能分析（视觉大模型方案）

两种模式：
1. 查看已有结果：展示 25 段自录视频的零样本行为识别结果（读取 results/）
2. 上传视频识别：上传任意教室监控视频，实时跑完整识别管线（抽帧→检测→
   多帧拼接→VLM 推理→事件聚合），判断 5 类行为

启动：
    streamlit run src/app/demo.py
"""
import json
import subprocess
import sys
import time
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
RAW_DIR = ROOT / "src" / "data" / "raw"
JSON_DIR = ROOT / "results" / "json"
PY = sys.executable

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
st.caption("Qwen3-VL-32B-Thinking · 千帆私有化部署 · 5 类行为：睡觉 / 看手机 / 张望 / 交谈 / 离座")

mode = st.radio("选择模式", ["📁 查看已有结果", "📤 上传视频识别"],
                horizontal=True)


# =====================================================================
# 模式一：查看已有结果（25 段自录视频演示）
# =====================================================================
if mode == "📁 查看已有结果":
    videos = sorted(p.stem for p in EVENT_DIR.glob("*.json"))

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


# =====================================================================
# 模式二：上传视频 → 实时识别
# =====================================================================
else:
    st.markdown("### 📤 上传教室监控视频，实时判断 5 类行为")
    st.caption("建议 10~45 秒、单人或双人入镜、视角稳定。识别耗时约 2~4 分钟"
               "（11 张多帧图逐张调用大模型）。")

    def run_step(name: str, cmd: list, status) -> bool:
        """执行一个管线步骤，返回是否成功。"""
        status.update(label=f"⏳ {name} ...")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=900)
        except subprocess.TimeoutExpired:
            st.error(f"❌ {name} 超时（>15 分钟）")
            return False
        out = (r.stdout or "") + (r.stderr or "")
        tail = [ln for ln in out.splitlines() if "[OK]" in ln or "[DONE]" in ln
                or "[ERR]" in ln or "Error" in ln]
        if r.returncode != 0 and not tail:
            st.error(f"❌ {name} 失败：{out[-600:]}")
            return False
        status.update(label=f"✅ {name} 完成", state="complete",
                      expanded=False)
        return True

    uploaded = st.file_uploader("选择视频文件", type=["mp4", "mov", "avi"])
    if uploaded is None:
        st.info("👆 请先上传一个视频文件（mp4 / mov / avi）")
    else:
        up_id = f"up_{int(time.time())}"
        video_path = RAW_DIR / f"{up_id}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(uploaded.getbuffer())
        st.success(f"已接收 {uploaded.name}（{uploaded.size / 1024 / 1024:.1f} MB）")

        if st.button("🚀 开始识别", type="primary"):
            with st.status("正在运行识别管线 ...", expanded=True) as status:
                steps = [
                    ("抽帧 (1fps)", [PY, "src/preprocess/extract_frames.py",
                                     str(video_path), "--fps", "1"]),
                    ("行人检测裁剪", [PY, "src/preprocess/detect_persons.py",
                                     str(FRAME_DIR / up_id)]),
                    ("多帧拼接", [PY, "src/preprocess/build_union_multi.py",
                                  up_id, "--stride", "4", "--resize", "192"]),
                    ("VLM 零样本推理", [PY, "src/vlm_infer/run_inference.py",
                                       up_id, "--union", "--simple"]),
                    ("事件聚合", [PY, "src/aggregate/events.py", up_id]),
                ]
                ok = True
                for name, cmd in steps:
                    if not run_step(name, cmd, status):
                        ok = False
                        break
                status.update(label="🏁 识别完成" if ok else "⛔ 流程中断",
                              state="complete", expanded=False)

            if ok:
                ev_file = EVENT_DIR / f"{up_id}.json"
                events = (json.loads(ev_file.read_text(encoding="utf-8"))
                          if ev_file.exists() else [])

                # ---- 结论区 ----
                st.markdown("### 🎯 识别结论")
                if events:
                    chips = "".join(
                        f"<span class='evt-chip' style='background:{COLORS[e['behavior']]};'>"
                        f"{BEHAVIOR_CN[e['behavior']]} {e.get('duration', 0)}s"
                        f"（{e['start']}-{e['end']}s）</span>"
                        for e in events)
                    st.markdown(chips, unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='exp-box'><b>判断依据</b>："
                        f"{events[0].get('reason', '—')}</div>",
                        unsafe_allow_html=True)
                else:
                    st.info("未识别出这 5 类行为事件（画面可能为正常静坐/学习）。")

                # ---- 时间轴 ----
                st.markdown("### ⏱️ 行为时间轴")
                tpath = REPORT_DIR / f"{up_id}_timeline.png"
                if tpath.exists():
                    st.image(str(tpath), use_container_width=True)

                # ---- 事件表格 ----
                if events:
                    st.markdown("### 📊 事件明细")
                    rows = [
                        {"行为": BEHAVIOR_CN[e["behavior"]],
                         "时段": f"{e['start']}-{e['end']}s",
                         "时长": f"{e['duration']}s",
                         "来源": "检测层" if e.get("source") == "detector"
                         else "VLM"}
                        for e in events
                    ]
                    st.dataframe(rows, use_container_width=True, hide_index=True)

                # ---- 逐帧判断分布 ----
                st.markdown("### 🧩 模型逐帧判断分布")
                jl = JSON_DIR / f"{up_id}.jsonl"
                if jl.exists():
                    cnt = {b: 0 for b in BEHAVIOR_CN}
                    n = 0
                    for ln in jl.read_text(encoding="utf-8").splitlines():
                        if not ln.strip():
                            continue
                        rec = json.loads(ln)
                        n += 1
                        for b in BEHAVIOR_CN:
                            cnt[b] += int(rec.get(b, 0) or 0)
                    dist = [
                        {"行为": BEHAVIOR_CN[b], "触发帧数": cnt[b],
                         "占比": f"{cnt[b] / max(n, 1) * 100:.0f}%"}
                        for b in BEHAVIOR_CN if cnt[b] > 0
                    ]
                    if dist:
                        st.dataframe(dist, use_container_width=True,
                                     hide_index=True)
                    else:
                        st.caption("模型在各帧均未触发 5 类行为。")

                st.caption(f"本次结果已存档：results/events/{up_id}.json，"
                           f"可在「查看已有结果」模式通过选择该视频复看（重启后仍可查看）。")
