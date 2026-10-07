"""CP3 — sweep score_thr cho PointPillars: so box dự đoán (results/preds_kitti.json, sinh bởi
src/infer.py) với GT KITTI, đo số box, recall/precision theo lớp và theo khoảng cách.

Chỉ đổi MỘT yếu tố: score_thr. Model, 20 frame, quy tắc ghép box giữ nguyên.
Model chỉ chạy 1 lần (score_thr = 0.1 trong config); các mức cao hơn = lọc lại output.

Quy tắc đánh giá (tự đặt, đơn giản hơn AP chính thức của KITTI):
  - GT được đánh giá: Car, Pedestrian, Cyclist (3 lớp model được train).
  - Box dự đoán được xét: score >= thr. Box KHÔNG ghép được chỉ tính FP nếu tâm nằm trong FOV camera
    (KITTI chỉ gán nhãn vật trong ảnh). Không lọc FOV trước khi ghép: vật bị cắt ở mép ảnh
    (truncated) có tâm box ngoài ảnh nhưng vẫn có GT.
  - Ghép: duyệt box dự đoán theo score giảm dần, ghép với GT CÙNG LỚP chưa được ghép,
    gần nhất, khoảng cách tâm BEV <= --match-dist (mặc định 2 m) -> TP. Không ghép được -> FP.
    GT không được ghép -> FN (bỏ sót).
  - Box dự đoán rơi vào vật "gần giống" (Van với Car, Person_sitting với Pedestrian) không tính FP,
    giống cách KITTI bỏ qua các lớp này.
  - Nhóm khoảng cách theo khoảng cách BEV từ LiDAR tới tâm GT: 0-20, 20-40, >40 m.

Chạy từ gốc repo:
    .venv-det/bin/python -m src.benchmark
    .venv-det/bin/python -m src.benchmark --preds results/preds_kitti_second.json --tag _second
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.boxes import kitti_obj_to_lidar_box
from starter.datasets import load_frame

CLASSES = ["Pedestrian", "Cyclist", "Car"]          # label id 0, 1, 2 của checkpoint
IGNORE_AS = {"Van": "Car", "Person_sitting": "Pedestrian"}  # dự đoán trúng các vật này: không phạt
RANGE_BINS = [(0, 20), (20, 40), (40, np.inf)]


def bin_name(lo, hi) -> str:
    return f"{lo:.0f}-{hi:.0f}m" if np.isfinite(hi) else f">{lo:.0f}m"


def load_gt(data_root: str, frame_ids) -> dict:
    """GT mỗi frame: list (class, box LiDAR 7 số, khoảng cách BEV), tách GT chính và GT bỏ qua."""
    gt = {}
    for fid in frame_ids:
        fr = load_frame(data_root, fid)
        main, ignore = [], []
        for o in fr["labels"]:
            if o.type in CLASSES or o.type in IGNORE_AS:
                box = kitti_obj_to_lidar_box(o, fr["calib"])
                item = (o.type, box, float(np.hypot(box[0], box[1])))
                (main if o.type in CLASSES else ignore).append(item)
        gt[fid] = {"main": main, "ignore": ignore}
    return gt


def match_frame(pred: dict, gt: dict, thr: float, match_dist: float) -> dict:
    """Ghép box dự đoán với GT của 1 frame ở ngưỡng thr."""
    scores = np.array(pred["scores"])
    in_fov = np.array(pred["in_fov"], dtype=bool)
    keep = scores >= thr
    idx = np.where(keep)[0]
    idx = idx[np.argsort(-scores[idx])]                       # score cao xét trước
    boxes, labels = np.array(pred["boxes"]).reshape(-1, 7), np.array(pred["labels"])

    gt_main = gt["main"]
    matched = np.zeros(len(gt_main), dtype=bool)
    tp_scores, fp_scores = [], []
    for i in idx:
        cls = CLASSES[labels[i]]
        best, best_d = -1, match_dist
        for j, (g_cls, g_box, _) in enumerate(gt_main):
            if g_cls != cls or matched[j]:
                continue
            d = np.hypot(*(boxes[i, :2] - g_box[:2]))
            if d <= best_d:
                best, best_d = j, d
        if best >= 0:
            matched[best] = True
            tp_scores.append(scores[i])
            continue
        near_ignore = any(IGNORE_AS[g_cls] == cls and np.hypot(*(boxes[i, :2] - g_box[:2])) <= match_dist
                          for g_cls, g_box, _ in gt["ignore"])
        if not near_ignore and in_fov[i]:
            fp_scores.append(scores[i])
    return {"n_pred": int((keep & in_fov).sum()), "n_pred_all": int(keep.sum()),
            "matched": matched, "tp_scores": tp_scores, "fp_scores": fp_scores}


def main() -> None:
    ap = argparse.ArgumentParser(description="CP3: sweep score_thr, recall/precision theo lớp và khoảng cách")
    ap.add_argument("--preds", default="results/preds_kitti.json", help="output của src/infer.py")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--thrs", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--match-dist", type=float, default=2.0, help="khoảng cách tâm BEV tối đa để ghép (m)")
    ap.add_argument("--pass-thr", type=float, default=0.3, help="ngưỡng dùng cho bảng pass/fail từng frame")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="", help="hậu tố tên file output, vd. _second")
    args = ap.parse_args()
    t = args.tag

    preds = json.load(open(args.preds))["frames"]
    frames = sorted(preds)
    gt = load_gt(args.data_root, frames)
    out, fig_dir = Path(args.out_dir), Path(args.out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows, cls_rows, frame_rows, scores_by_thr = [], [], [], {}
    for thr in args.thrs:
        n_pred = n_pred_all = 0
        tp_s, fp_s = [], []
        hits = []   # (class, range, matched) của mọi GT
        for fid in frames:
            r = match_frame(preds[fid], gt[fid], thr, args.match_dist)
            n_pred += r["n_pred"]; n_pred_all += r["n_pred_all"]
            tp_s += r["tp_scores"]; fp_s += r["fp_scores"]
            hits += [(c, rng, m) for (c, _, rng), m in zip(gt[fid]["main"], r["matched"])]
            if thr == args.pass_thr:
                n_gt, n_tp = len(r["matched"]), int(r["matched"].sum())
                frame_rows.append({"frame": fid, "score_thr": thr, "n_gt": n_gt, "tp": n_tp,
                                   "fn": n_gt - n_tp, "fp": len(r["fp_scores"]),
                                   "pass": n_tp == n_gt and len(r["fp_scores"]) == 0})
        scores_by_thr[thr] = (tp_s, fp_s)
        tp, fp, n_gt = len(tp_s), len(fp_s), len(hits)
        row = {"score_thr": thr, "boxes_per_frame": n_pred_all / len(frames),
               "boxes_in_fov_per_frame": n_pred / len(frames), "n_gt": n_gt, "tp": tp, "fp": fp,
               "fn": n_gt - tp, "recall": tp / n_gt, "precision": tp / max(tp + fp, 1)}
        for cls in CLASSES:
            h = [m for c, _, m in hits if c == cls]
            row[f"recall_{cls}"] = np.mean(h) if h else np.nan
            for lo, hi in RANGE_BINS:
                hb = [m for c, rng, m in hits if c == cls and lo <= rng < hi]
                cls_rows.append({"score_thr": thr, "class": cls, "range": bin_name(lo, hi), "n_gt": len(hb),
                                 "detected": int(sum(hb)), "recall": np.mean(hb) if hb else np.nan})
        sweep_rows.append(row)

    sweep = pd.DataFrame(sweep_rows).round(3)
    by_cls = pd.DataFrame(cls_rows).round(3)
    per_frame = pd.DataFrame(frame_rows)
    sweep.to_csv(out / f"score_thr_sweep{t}.csv", index=False)
    by_cls.to_csv(out / f"recall_by_class_range{t}.csv", index=False)
    per_frame.to_csv(out / f"per_frame_pass_fail_thr{args.pass_thr}{t}.csv", index=False)
    print(sweep.to_string(index=False))
    print()
    print(by_cls.pivot_table(index=["class", "range"], columns="score_thr", values="recall").to_string())
    print(f"\nPass/fail từng frame @ score_thr={args.pass_thr}: "
          f"{int(per_frame['pass'].sum())}/{len(per_frame)} frame pass")

    # Hình 1: recall / precision / số box theo score_thr
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for col, lab in [("recall", "recall (mọi lớp)"), ("precision", "precision")] + \
                    [(f"recall_{c}", f"recall {c}") for c in CLASSES]:
        a1.plot(sweep["score_thr"], sweep[col], "o-", label=lab)
    a1.set_xlabel("score_thr"); a1.set_ylabel("tỉ lệ"); a1.set_ylim(0, 1.05); a1.grid(alpha=.3); a1.legend(fontsize=8)
    a1.set_title("Recall / precision theo score_thr")
    a2.bar(sweep["score_thr"].astype(str), sweep["boxes_in_fov_per_frame"], color="tab:red", alpha=.7,
           label="box dự đoán / frame (trong FOV)")
    a2.axhline(sweep["n_gt"][0] / len(frames), color="g", ls="--", label="GT / frame")
    a2.set_xlabel("score_thr"); a2.set_title("Số box mỗi frame"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / f"score_thr_sweep{t}.png", dpi=120); plt.close(fig)

    # Hình 2: recall theo khoảng cách cho từng lớp, mỗi cột một score_thr
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        d = by_cls[by_cls["class"] == cls]
        bins = [bin_name(lo, hi) for lo, hi in RANGE_BINS]
        w = 0.8 / len(args.thrs)
        for k, thr in enumerate(args.thrs):
            dd = d[d["score_thr"] == thr].set_index("range").reindex(bins)
            ax.bar(np.arange(3) + k * w, dd["recall"].fillna(0), w, label=f"thr {thr}")
        ns = d[d["score_thr"] == args.thrs[0]].set_index("range").reindex(bins)["n_gt"]
        ax.set_xticks(np.arange(3) + w); ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bins, ns)])
        ax.set_title(f"Recall {cls} theo khoảng cách"); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("recall"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / f"recall_by_range{t}.png", dpi=120); plt.close(fig)

    # Hình 3: histogram score của box đúng (TP) và sai (FP) ở ngưỡng thấp nhất
    tp_s, fp_s = scores_by_thr[min(args.thrs)]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(args.thrs), 1, 19)
    ax.hist([tp_s, fp_s], bins=bins, stacked=True, color=["tab:green", "tab:red"],
            label=[f"TP (n={len(tp_s)})", f"FP (n={len(fp_s)})"])
    for thr in args.thrs[1:]:
        ax.axvline(thr, color="k", ls="--", lw=.8)
    ax.set_xlabel("score"); ax.set_ylabel("số box"); ax.legend()
    ax.set_title("Phân bố score: box đúng vs box sai (trong FOV)")
    fig.tight_layout(); fig.savefig(fig_dir / f"score_hist{t}.png", dpi=120); plt.close(fig)
    print(f"-> {out}/score_thr_sweep{t}.csv, recall_by_class_range{t}.csv, per_frame_pass_fail_thr{args.pass_thr}{t}.csv, "
          f"figures/score_thr_sweep{t}.png, recall_by_range{t}.png, score_hist{t}.png")


if __name__ == "__main__":
    main()
