"""Streamlit 演示应用：教室行为智能分析。

两种模式：
1. 查看已有结果：浏览 25 段自录视频的识别结果（results/）
2. 上传视频识别：上传视频后跑完整识别管线（抽帧→检测→拼接→VLM→聚合）

启动：streamlit run src/app/demo.py
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EVENT_DIR = ROOT / "results" / "events"
REPORT_DIR = ROOT / "results" / "report"
FRAME_DIR = ROOT / "src" / "data" / "frames"
RAW_DIR = ROOT / "src" / "data" / "raw"
CROP_DIR = ROOT / "src" / "data" / "crops"
UNION_DIR = ROOT / "src" / "data" / "crops_union"
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


def html_table(headers, rows):
    """自定义浅色表格（不随系统深色主题变黑）。"""
    thead = "".join(f"<th>{h}</th>" for h in headers)
    tbody = "".join(
        f"<tr>{''.join(f'<td>{c}</td>' for c in row)}</tr>" for row in rows)
    return f"<table class='tbl'><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"


def delete_video_data(video: str):
    """删除某视频的全部数据：原始视频、抽帧、裁剪、拼接、推理结果、时间轴。"""
    targets = [
        RAW_DIR / f"{video}.mp4",
        FRAME_DIR / video,
        CROP_DIR / video,
        UNION_DIR / video,
        JSON_DIR / f"{video}.jsonl",
        JSON_DIR / f"{video}.single.jsonl",
        EVENT_DIR / f"{video}.json",
        REPORT_DIR / f"{video}_timeline.png",
        REPORT_DIR / f"{video}.single_timeline.png",
    ]
    removed, failed = [], []
    for t in targets:
        try:
            if t.is_dir():
                shutil.rmtree(t)
                removed.append(t)
            elif t.exists():
                t.unlink()
                removed.append(t)
        except Exception as e:
            failed.append((t, e))
    return removed, failed

st.set_page_config(page_title="教室行为智能分析", layout="wide")

st.markdown("""
<style>
:root { --bg: #F6F8FB; --card: #FFFFFF; --line: #E7EDF3;
        --ink: #263445; --ink2: #6B7A8C; --accent: #3E7BB6; }
html, body, [class*="css"] { font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; }
:root, html, body { color-scheme: light; }  /* 强制浅色渲染，避免系统深色模式把组件变黑 */
.block-container { padding: 1.6rem 2.5rem 3rem 0.8rem; max-width: 1500px; }

/* 侧边栏收窄，减少与主界面的空隙 */
[data-testid="stSidebar"] {
    min-width: 300px !important; max-width: 300px !important;
    width: 300px !important;
}

.stApp {
    background:
        radial-gradient(1100px 420px at 18% -6%, rgba(139,200,234,.16) 0%, rgba(139,200,234,0) 60%),
        radial-gradient(900px 380px at 92% 0%, rgba(148,216,195,.14) 0%, rgba(148,216,195,0) 55%),
        var(--bg);
}
h1, h2, h3 { color: var(--ink); letter-spacing: .3px; }

/* 隐藏 Streamlit 顶部工具条，避免深色 header 横条 */
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none !important; }

.hero {
    background: linear-gradient(120deg, #EAF3FB 0%, #E7F3EC 55%, #FBF4E9 100%);
    border: 1px solid var(--line);
    border-radius: 14px; padding: 12px 24px; margin-bottom: 10px;
    box-shadow: 0 3px 16px rgba(46, 91, 127, .06);
    position: relative; overflow: hidden;
}
.hero::after {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #8BC8EA, #94D8C3, #E4D48F);
}
.hero h1 { margin: 0; font-size: 28px; font-weight: 600; color: #2C5F8A; }
.hero p { margin: 0; font-size: 15px; color: var(--ink2); }

.card {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 14px; padding: 14px 18px; margin: 8px 0 14px;
    box-shadow: 0 4px 18px rgba(46, 91, 127, .06);
    position: relative; overflow: hidden;
}
.card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #8BC8EA, #94D8C3);
}
.card h3 { margin: 0 0 12px; font-size: 18px; font-weight: 600; color: #34506B; }
.card-sub { font-size: 14px; color: var(--ink2); margin-bottom: 10px; }

.stMarkdown p, .stMarkdown li { font-size: 15px; }
.stMarkdown h4 { font-size: 17px; }
.stInfo, .stSuccess, .stWarning, .stError { font-size: 14px; }
.stCaption, .stCaption p { font-size: 13px; color: var(--ink2); }

div[data-testid="stRadio"] > div { gap: 0; }
div[data-testid="stRadio"] label {
    padding: 9px 26px; border: 1px solid var(--line);
    background: #fff; color: var(--ink2); font-size: 16px;
    border-radius: 0; margin: 0;
}
div[data-testid="stRadio"] label:first-of-type { border-radius: 10px 0 0 10px; }
div[data-testid="stRadio"] label:last-of-type { border-radius: 0 10px 10px 0; }
div[data-testid="stRadio"] label p { color: inherit; }
div[data-testid="stRadio"] label:hover { background: #F0F5FA; }
div[data-testid="stRadio"] label:has(input:checked) {
    background: var(--accent); border-color: var(--accent); color: #fff;
}
div[data-testid="stRadio"] label:has(input:checked) p { color: #fff; }
div[data-testid="stRadio"] input { display: none; }
div[data-testid="stRadio"] > div > div > div { padding: 0; }
div[data-testid="stRadio"] > div > div { flex-direction: row; }

.evt-chip {
    display: inline-block; border-radius: 18px;
    padding: 7px 18px; margin: 3px 7px 3px 0;
    color: #fff; font-size: 15px; font-weight: 500;
    box-shadow: 0 2px 6px rgba(38,52,69,.12);
}
.evt-chip span { font-weight: 400; opacity: .9; }

/* 侧边栏：淡蓝渐变过渡 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F1F7FB 0%, #FFFFFF 55%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h3 {
    font-size: 14px; color: #34506B; margin-top: 6px;
    border-left: 3px solid #8BC8EA; padding-left: 8px;
}
[data-testid="stSidebar"] .evt-chip { margin: 4px 6px 4px 0; }

.expect-badge {
    display: inline-block; background: #EDF5FB; color: #2C5F8A;
    border: 1px solid #C9DDF0; border-radius: 8px;
    padding: 4px 14px; font-size: 15px; font-weight: 500;
}

/* 下拉框强制浅色，不随系统深色主题变黑 */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #FFFFFF !important; border-color: var(--line) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: var(--ink) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: var(--ink2) !important;
}
[data-baseweb="popover"] [data-baseweb="menu"] {
    background: #FFFFFF !important; color: var(--ink) !important;
}
[data-baseweb="menu"] li { color: var(--ink) !important; }
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] li[aria-selected="true"] {
    background: #F0F5FA !important; color: var(--ink) !important;
}

/* 自定义浅色表格（替代 st.dataframe，不随系统深色主题变黑） */
table.tbl { width: 100%; border-collapse: collapse; font-size: 15px;
            margin: 2px 0 6px; }
table.tbl th {
    background: linear-gradient(180deg, #F0F5FA, #EAF2F8);
    color: #34506B; text-align: left;
    padding: 11px 16px; font-weight: 600; border-bottom: 1px solid var(--line);
}
table.tbl td { padding: 11px 16px; border-bottom: 1px solid #EEF2F7;
               color: var(--ink); }
table.tbl tbody tr:nth-child(even) { background: #FAFCFE; }
table.tbl tbody tr:hover { background: #F2F7FC; }

[data-testid="stFileUploader"] section {
    border: 1.5px dashed #B9CDE2; border-radius: 12px;
    background: #FBFDFF; padding: 6px;
}
.stButton > button[kind="primary"] {
    background: var(--accent); border: none; border-radius: 10px;
    padding: 8px 28px; font-size: 14px; font-weight: 500;
    box-shadow: 0 2px 8px rgba(62,123,182,.25);
}
.stButton > button[kind="primary"]:hover { background: #35699D; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>教室行为智能分析</h1>
</div>
""", unsafe_allow_html=True)

mode = st.radio("模式", ["查看已有结果", "上传视频识别"], horizontal=True,
                label_visibility="collapsed")


if mode == "查看已有结果":
    videos = sorted(p.stem for p in EVENT_DIR.glob("*.json"))

    with st.sidebar:
        st.markdown("#### 选择视频")
        video = st.selectbox("视频", videos, label_visibility="collapsed")
        prefix = video.rsplit("_", 1)[0]
        expect = PREFIX2BEHAVIOR.get(prefix, "未知")
        st.markdown(
            f"<div class='card-sub'>预期行为 "
            f"<span class='expect-badge'>{expect}</span></div>",
            unsafe_allow_html=True)

        events = json.loads(
            (EVENT_DIR / f"{video}.json").read_text(encoding="utf-8"))
        st.markdown("#### 识别事件")
        if events:
            chips = "".join(
                f"<span class='evt-chip' style='background:{COLORS.get(e['behavior'], '#9AA6B2')};'>"
                f"{BEHAVIOR_CN[e['behavior']]} <span>{e.get('duration', 0)}s</span></span>"
                for e in events)
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown("<div class='card-sub'>无事件</div>",
                        unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 行为图例")
        legend = "".join(
            f"<div style='display:flex;align-items:center;gap:9px;"
            f"margin:6px 0;font-size:15px;color:#6B7A8C;'>"
            f"<span style='width:12px;height:12px;border-radius:50%;"
            f"background:{COLORS[b]};display:inline-block;'></span>"
            f"{BEHAVIOR_CN[b]}</div>" for b in BEHAVIOR_CN)
        st.markdown(legend, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 删除数据")
        del_key = f"del_confirm_{video}"
        if not st.session_state.get(del_key, False):
            if st.button("删除该视频的全部数据", width="stretch"):
                st.session_state[del_key] = True
                st.rerun()
        else:
            st.warning("将一并删除：原始视频、抽帧、裁剪图、拼接图、"
                       "推理结果与时间轴，无法恢复。")
            c1, c2 = st.columns(2)
            if c1.button("确认删除", width="stretch"):
                removed, failed = delete_video_data(video)
                st.session_state[del_key] = False
                if failed:
                    names = "、".join(str(t.relative_to(ROOT)) for t, _ in failed)
                    st.error(f"以下 {len(failed)} 项删除失败：{names}")
                st.success(f"已删除 {len(removed)} 项数据。")
                st.rerun()
            if c2.button("取消", width="stretch"):
                st.session_state[del_key] = False
                st.rerun()

    c1, c2 = st.columns([1.3, 2])

    with c1:
        st.markdown(
            f"<div class='card'><h3>事件摘要</h3>"
            f"<div class='card-sub'>{video}</div>", unsafe_allow_html=True)
        if events:
            rows = [
                [BEHAVIOR_CN[e["behavior"]],
                 f"{e['start']}-{e['end']}s",
                 f"{e['duration']}s",
                 "检测层" if e.get("source") == "detector" else "VLM"]
                for e in events
            ]
            st.markdown(html_table(["行为", "时段", "时长", "来源"], rows),
                        unsafe_allow_html=True)
        else:
            st.info("该视频未识别出任何行为事件。")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3>关键帧证据</h3>",
                    unsafe_allow_html=True)
        fdir = FRAME_DIR / video
        if fdir.exists():
            fps = sorted(fdir.glob("frame_*.jpg"))
            sample = [fps[i] for i in
                      range(0, len(fps), max(1, len(fps) // 6))][:6]
            cols = st.columns(3)
            for i, fp in enumerate(sample):
                with cols[i % 3]:
                    img = Image.open(str(fp))
                    st.image(img, caption=f"t={i * 7}s 附近", width=180)
        else:
            st.warning("未找到抽帧目录")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"<div class='card'><h3>行为时间轴</h3>"
            f"<div class='card-sub'>{video}</div>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    with st.expander("技术方案说明"):
        st.markdown("""
**任务**：对教室监控视频进行 5 类行为识别（睡觉 / 看手机 / 张望 / 交谈 / 离座）。

**方案**：视觉大模型零样本识别，不训练任何分类器——
1. 预处理：1fps 抽帧 → YOLOv8n 行人检测，取画面中所有行人的联合区域；
2. 统一送入 VLM：将连续 6 个时刻的联合区域图拼接为 2×3 网格（多帧拼接），
   让 VLM 观察头部转动、嘴部开合、手部动作、坐姿转移动等时序信号；
3. 事件聚合：逐帧判断按时间连续性合并（持续行为 ≥5s、瞬态行为 ≥3s 记为事件），
   离座补充检测层规则（行人目标消失即判定）。

**模型**：Qwen3-VL-32B-Thinking。
""")

else:
    st.markdown("""
<div class='card'>
  <h3>上传视频识别</h3>
  <div class='card-sub'>上传教室监控视频，实时判断 5 类行为。
  建议 10~45 秒、单人或双人入镜、视角稳定。识别耗时约 3~6 分钟
  （多帧图逐张调用大模型，耗时随视频长度与服务端负载波动）。</div>
</div>""", unsafe_allow_html=True)

    def run_step(name: str, cmd: list, status) -> bool:
        status.update(label=f"{name} ...")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=900)
        except subprocess.TimeoutExpired:
            st.error(f"{name} 超时（>15 分钟）")
            return False
        out = (r.stdout or "") + (r.stderr or "")
        tail = [ln for ln in out.splitlines() if "[OK]" in ln or "[DONE]" in ln
                or "[ERR]" in ln or "Error" in ln]
        if r.returncode != 0 and not tail:
            st.error(f"{name} 失败：{out[-600:]}")
            return False
        status.update(label=f"{name} 完成", state="complete", expanded=False)
        return True

    uploaded = st.file_uploader("选择视频文件", type=["mp4", "mov", "avi"])
    if uploaded is None:
        st.info("请先上传一个视频文件（mp4 / mov / avi）")
    else:
        fid = getattr(uploaded, "file_id", id(uploaded))
        if st.session_state.get("last_upload_id") != fid:
            st.session_state.last_upload_id = fid
            st.session_state.up_id = f"up_{int(time.time())}"
        up_id = st.session_state.up_id
        video_path = RAW_DIR / f"{up_id}.mp4"
        if not video_path.exists():
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(uploaded.getbuffer())
        st.success(f"已接收 {uploaded.name}（{uploaded.size / 1024 / 1024:.1f} MB）")

        orphans = sorted(p.name for p in RAW_DIR.glob("up_*.mp4")
                         if not (EVENT_DIR / f"{p.stem}.json").exists()
                         and p.name != f"{up_id}.mp4")
        if orphans:
            st.divider()
            st.markdown("#### 未识别的孤立上传")
            st.caption("以下文件只有原始视频、没有识别结果，可单独删除：")
            for name in orphans:
                c1, c2 = st.columns([4, 1])
                c1.caption(name)
                if c2.button("删除", key=f"del_orphan_{name}"):
                    (RAW_DIR / name).unlink(missing_ok=True)
                    st.rerun()

        if st.button("开始识别", type="primary"):
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
                status.update(label="识别完成" if ok else "流程中断",
                              state="complete", expanded=False)

            if ok:
                ev_file = EVENT_DIR / f"{up_id}.json"
                events = (json.loads(ev_file.read_text(encoding="utf-8"))
                          if ev_file.exists() else [])

                st.markdown(
                    "<div class='card'><h3>识别结论</h3>",
                    unsafe_allow_html=True)
                if events:
                    chips = "".join(
                        f"<span class='evt-chip' style='background:{COLORS.get(e['behavior'], '#9AA6B2')};'>"
                        f"{BEHAVIOR_CN[e['behavior']]} <span>"
                        f"{e.get('duration', 0)}s · {e['start']}-{e['end']}s"
                        f"</span></span>" for e in events)
                    st.markdown(chips, unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card-sub' style='margin-top:10px;'>"
                        f"判断依据：{events[0].get('reason', '—')}</div>",
                        unsafe_allow_html=True)
                else:
                    st.info("未识别出这 5 类行为事件（画面可能为正常静坐/学习）。")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='card'><h3>行为时间轴</h3>",
                            unsafe_allow_html=True)
                tpath = REPORT_DIR / f"{up_id}_timeline.png"
                if tpath.exists():
                    st.image(str(tpath), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                if events:
                    st.markdown("<div class='card'><h3>事件明细</h3>",
                                unsafe_allow_html=True)
                    rows = [
                        [BEHAVIOR_CN[e["behavior"]],
                         f"{e['start']}-{e['end']}s",
                         f"{e['duration']}s",
                         "检测层" if e.get("source") == "detector" else "VLM"]
                        for e in events
                    ]
                    st.markdown(html_table(["行为", "时段", "时长", "来源"], rows),
                                unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='card'><h3>逐帧判断分布</h3>",
                            unsafe_allow_html=True)
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
                        [BEHAVIOR_CN[b], str(cnt[b]),
                         f"{cnt[b] / max(n, 1) * 100:.0f}%"]
                        for b in BEHAVIOR_CN if cnt[b] > 0
                    ]
                    if dist:
                        st.markdown(html_table(["行为", "触发帧数", "占比"], dist),
                                    unsafe_allow_html=True)
                    else:
                        st.caption("模型在各帧均未触发 5 类行为。")
                st.markdown("</div>", unsafe_allow_html=True)

                st.caption(f"结果已存档：results/events/{up_id}.json，"
                           f"可在「查看已有结果」模式复看（重启后仍可查看）。")
