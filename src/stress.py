"""Bonus B2 — stress test: làm suy giảm point cloud rồi chạy lại detector, đo recall/precision.

Hai loại suy giảm (hàm có sẵn trong starter/perturb.py), mỗi loại 3 mức + mức gốc:
  - random_dropout(keep_ratio = 0.7 / 0.5 / 0.3): giả lập LiDAR thưa hơn, mất điểm (mưa, bụi, phản xạ kém)
  - gaussian_noise(sigma_xyz = 0.02 / 0.05 / 0.10 m): giả lập sai số đo khoảng cách
Mỗi lần chỉ đổi MỘT yếu tố; model, 20 frame, GT, score_thr, quy tắc ghép giữ nguyên.
seed cố định (=0) cho mọi phép ngẫu nhiên -> chạy lại ra cùng số.

Chạy từ gốc repo (cần GPU):
    .venv-det/bin/python -m src.stress
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark import load_gt, match_frame
from src.boxes import points_in_box
from src.infer import MODELS, default_config, in_camera_fov, run_model
from starter.datasets import list_frames, load_frame
from starter.perturb import gaussian_noise, random_dropout

CONFIGS = [("gốc", "none", 0.0)] + \
          [("random_dropout", "keep_ratio", k) for k in (0.7, 0.5, 0.3)] + \
          [("gaussian_noise", "sigma_xyz_m", s) for s in (0.02, 0.05, 0.10)]


def perturb(points, kind, level, seed):
    if kind == "random_dropout":
        return random_dropout(points, keep_ratio=level, seed=seed)
    if kind == "gaussian_noise":
        return gaussian_noise(points, sigma_xyz_m=level, seed=seed)
    return points


def main() -> None:
    ap = argparse.ArgumentParser(description="B2: stress test random_dropout + gaussian_noise")
    ap.add_argument("--model", choices=list(MODELS), default="pointpillars")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--match-dist", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    from mmdet3d.apis import init_model
    model = init_model(default_config(args.model), MODELS[args.model][1], device="cuda:0")
    frames = list_frames(args.data_root)
    gt = load_gt(args.data_root, frames)
    data = {}
    for fid in frames:
        fr = load_frame(args.data_root, fid)
        data[fid] = (fr["points"][np.isfinite(fr["points"]).all(axis=1)], fr["calib"], fr["image"].shape)

    rows = []
    for kind, param, level in CONFIGS:
        tp = fp = n_gt = n_pts = 0
        pts_on_obj = []
        for fid in frames:
            pts0, calib, shape = data[fid]
            pts = perturb(pts0, kind, level, args.seed)
            pred = run_model(model, pts)
            pred["in_fov"] = in_camera_fov(pred["boxes"], calib, shape)
            r = match_frame({k: v.tolist() for k, v in pred.items()}, gt[fid], args.thr, args.match_dist)
            tp += int(r["matched"].sum()); fp += len(r["fp_scores"]); n_gt += len(r["matched"])
            n_pts += len(pts)
            pts_on_obj += [int(points_in_box(pts, box, 0.1).sum()) for _, box, _ in gt[fid]["main"]]
        rows.append({"perturbation": kind, "param": param, "level": level, "points_per_frame": n_pts / len(frames),
                     "median_points_per_gt": float(np.median(pts_on_obj)), "tp": tp, "fp": fp, "fn": n_gt - tp,
                     "recall": tp / n_gt, "precision": tp / max(tp + fp, 1)})
        print(f"{kind:15s} {param}={level:<5} recall={tp / n_gt:.3f} precision={tp / max(tp + fp, 1):.3f} "
              f"FP={fp} median điểm/GT={np.median(pts_on_obj):.0f}")

    df = pd.DataFrame(rows).round(3)
    out = Path(args.out_dir)
    df.to_csv(out / "stress_test.csv", index=False)

    base = df[df["perturbation"] == "gốc"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, kind, xlab in [(axes[0], "random_dropout", "tỉ lệ điểm giữ lại (keep_ratio)"),
                           (axes[1], "gaussian_noise", "độ lệch chuẩn nhiễu xyz (m)")]:
        d = df[df["perturbation"] == kind]
        x = [1.0 if kind == "random_dropout" else 0.0] + list(d["level"])
        for col, c in [("recall", "tab:blue"), ("precision", "tab:orange")]:
            ax.plot(x, [base[col]] + list(d[col]), "o-", color=c, label=col)
        ax.set_xlabel(xlab); ax.set_ylim(0, 1.05); ax.grid(alpha=.3); ax.legend()
        ax.set_title(f"{kind} — PointPillars, score_thr={args.thr}")
        if kind == "random_dropout":
            ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(out / "figures" / "stress_test.png", dpi=120); plt.close(fig)
    print(f"-> {out}/stress_test.csv, {out}/figures/stress_test.png")


if __name__ == "__main__":
    main()
