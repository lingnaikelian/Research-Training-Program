"""
YOLOv8-pose 规则基线（Baseline）：用姿态关键点的简单规则识别教室行为，
与 VLM 零样本方案做对比实验。规则阈值固定，不针对测试集调优。

规则（帧级，对每个 person）：
- away       : YOLO 检测不到 person 且位于视频尾部（目标消失=离座）
- sleeping   : 关键点平均置信度 < 0.55（趴桌时姿态估计质量差）
- looking_phone: 明显低头（nose_y - 肩中_y > 0.12*图像高）
- looking_around / talking : 姿态规则无法区分，基线不识别（记为 0）

输出：results/baseline/<video>.json（事件列表，与 VLM 事件同格式）

用法：
    python yolo_rules.py <video>     # 单视频
    python yolo_rules.py --all       # 全部 25 个
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # 使 from src... 可用
from src.preprocess.cv_utils import imread_unicode
FRAME_DIR = ROOT / "src" / "data" / "frames"
OUT_DIR = ROOT / "results" / "baseline"

BEHAVIORS = ["sleeping", "looking_phone", "looking_around", "talking", "away"]
BEHAVIOR_CN = {
    "sleeping": "睡觉", "looking_phone": "看手机",
    "looking_around": "张望", "talking": "交谈", "away": "离座",
}

KP_CONF_SLEEP = 0.55      # 趴桌时关键点置信度阈值
NOSE_BELOW_SH = 0.05      # 低头：鼻尖低于肩中线 5% 图像高（常识阈值，0.12 过严）


def detect_events_from_series(series: dict, min_frames: int = 5,
                              max_gap: int = 3) -> list:
    """与 events.py 相同的事件聚合逻辑（复制自 detect_events，独立可运行）。"""
    events = []
    for b in BEHAVIORS:
        frames = sorted(series[b].items())
        run_start = None
        run_sec = 0
        prev_idx = None
        for idx, val in frames:
            if val == 1:
                if run_start is None:
                    run_start = idx
                    run_sec = 1
                else:
                    if prev_idx is not None and (idx - prev_idx) <= max_gap:
                        run_sec += (idx - prev_idx)
                    else:
                        if run_sec >= min_frames:
                            events.append({
                                "behavior": b, "start": run_start,
                                "end": prev_idx, "duration": run_sec,
                                "confidence": round(run_sec / max(run_sec + 10, 1), 2),
                            })
                        run_start = idx
                        run_sec = 1
                prev_idx = idx
            else:
                if run_start is not None and run_sec >= min_frames:
                    events.append({
                        "behavior": b, "start": run_start,
                        "end": prev_idx, "duration": run_sec,
                        "confidence": round(run_sec / max(run_sec + 10, 1), 2),
                    })
                run_start = None
                run_sec = 0
                prev_idx = None
        if run_start is not None and run_sec >= min_frames:
            events.append({
                "behavior": b, "start": run_start,
                "end": prev_idx, "duration": run_sec,
                "confidence": round(run_sec / max(run_sec + 10, 1), 2),
            })
    events.sort(key=lambda e: (e["start"], e["behavior"]))
    return events


_model = None
def get_model():
    global _model
    if _model is None:
        _model = YOLO(ROOT / "yolov8n-pose.pt")
    return _model


def frame_level_rules(video: str, min_frames: int = 5) -> dict:
    """返回 {behavior: {frame_idx: 0/1}}，纯 YOLO-pose 规则。"""
    series = {b: {} for b in BEHAVIORS}
    frame_dir = FRAME_DIR / video
    if not frame_dir.exists():
        return series
    fps = sorted(frame_dir.glob("frame_*.jpg"))
    model = get_model()
    seen_person_last = None  # 最后一帧出现 person 的帧号

    # 第一遍：逐帧 pose，记录所有帧的检测状态
    frame_persons = {}  # idx -> [(kp_conf, nose_y, sh_y, img_h)]
    for fp in fps:
        idx = int(fp.stem.split("_")[1])
        img = imread_unicode(fp)
        h = img.shape[0]
        r = model(img, conf=0.3, verbose=False)[0]
        persons = []
        if len(r.boxes):
            kps = r.keypoints.data
            for i in range(len(r.boxes)):
                kc = float(kps[i, :, 2].mean())
                nose = kps[i, 0, :2].tolist()
                shy = float((kps[i, 5, 1] + kps[i, 6, 1]) / 2)
                persons.append((kc, float(nose[1]), shy, h))
                seen_person_last = idx
        frame_persons[idx] = persons

    # 第二遍：判定行为
    total = len(fps)
    for idx, persons in frame_persons.items():
        if not persons:
            # person 缺失：若在尾部且之前出现过 -> away
            if seen_person_last is not None and idx > seen_person_last:
                series["away"][idx] = 1
            continue
        for (kc, nose_y, shy, h) in persons:
            if kc < KP_CONF_SLEEP:
                series["sleeping"][idx] = 1
            elif (nose_y - shy) > NOSE_BELOW_SH * h:
                series["looking_phone"][idx] = 1
            # 其他（张望/交谈/正常）不识别

    # away 用连续缺失段语义：缺失帧已在上面标记，交给事件聚合
    return series


def process(video: str, min_frames: int = 5):
    series = frame_level_rules(video, min_frames)
    events = detect_events_from_series(series, min_frames)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{video}.json"
    out.write_text(json.dumps(events, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    summary = "、".join(
        f"{BEHAVIOR_CN[e['behavior']]}{e['duration']}s" for e in events) or "无事件"
    print(f"[OK] {video}: {len(events)} 个事件 -> {summary}")
    return events


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="视频名")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for d in sorted(FRAME_DIR.iterdir()):
            if d.is_dir():
                process(d.name)
    elif args.video:
        process(args.video)
    else:
        print("用法: python yolo_rules.py <视频名> | --all")
