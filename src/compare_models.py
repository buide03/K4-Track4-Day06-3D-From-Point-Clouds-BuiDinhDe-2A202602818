"""Bonus B1 — so sánh PointPillars và SECOND (cùng 20 frame kitti_mini, cùng quy tắc ghép, cùng metric).

Cần chạy trước:
    .venv-det/bin/python -m src.infer --model pointpillars
    .venv-det/bin/python -m src.infer --model second
    .venv-det/bin/python -m src.benchmark
    .venv-det/bin/python -m src.benchmark --preds results/preds_kitti_second.json --tag _second
Rồi:
    .venv-det/bin/python -m src.compare_models

Latency: nạp CẢ HAI model trong cùng một tiến trình và đo xen kẽ (frame 1 model A, frame 1 model B, ...)
để hai model chịu cùng điều kiện GPU (nhiệt độ, xung nhịp, tải nền). Bỏ --warmup lần đầu mỗi model.
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

from src.benchmark import CLASSES, load_gt, match_frame
from starter.datasets import list_frames, load_frame

NAMES = {"pointpillars": "PointPillars", "second": "SECOND"}


def recall_by_occlusion(preds, gt, data_root, thr, match_dist) -> dict:
    hit, tot = {}, {}
    for fid in sorted(preds):
        occ = [o.occluded for o in load_frame(data_root, fid)["labels"] if o.type in CLASSES]  # thứ tự như load_gt
        m = match_frame(preds[fid], gt[fid], thr, match_dist)["matched"]
        for o, ok in zip(occ, m):
            key = "occluded 0-1" if o <= 1 else ("occluded 2" if o == 2 else "occluded 3")
            hit[key] = hit.get(key, 0) + int(ok); tot[key] = tot.get(key, 0) + 1
    return {k: (hit[k], tot[k]) for k in sorted(tot)}


def measure_latency(data_root, warmup, runs) -> pd.DataFrame:
    import time
    import torch
    from mmdet3d.apis import inference_detector, init_model
    from src.infer import MODELS, default_config
    models = {k: init_model(default_config(k), MODELS[k][1], device="cuda:0") for k in NAMES}
    rows = []
    for fid in list_frames(data_root):
        p = load_frame(data_root, fid)["points"]
        p = p[np.isfinite(p).all(axis=1)].astype(np.float32)
        for k in range(warmup + runs):
            for name, model in models.items():          # xen kẽ 2 model
                torch.cuda.synchronize(); t0 = time.perf_counter()
                inference_detector(model, p)
                torch.cuda.synchronize()
                if k >= warmup:
                    rows.append({"model": NAMES[name], "frame": fid, "run": k - warmup,
                                 "ms": (time.perf_counter() - t0) * 1000})
    return pd.DataFrame(rows), torch.cuda.get_device_name(0)


def main() -> None:
    ap = argparse.ArgumentParser(description="B1: so sánh PointPillars vs SECOND (độ chính xác + latency)")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--thr", type=float, default=0.3, help="ngưỡng dùng cho bảng theo mức che khuất")
    ap.add_argument("--match-dist", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=20, help="số lần đo mỗi frame mỗi model")
    ap.add_argument("--skip-latency", action="store_true", help="bỏ đo latency (không cần GPU)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)

    # 1) Độ chính xác: ghép bảng sweep của 2 model (sinh bởi src/benchmark.py)
    acc = pd.concat([pd.read_csv(out / "score_thr_sweep.csv").assign(model="PointPillars"),
                     pd.read_csv(out / "score_thr_sweep_second.csv").assign(model="SECOND")])
    cols = ["model", "score_thr", "boxes_in_fov_per_frame", "tp", "fp", "fn", "recall", "precision",
            "recall_Car", "recall_Pedestrian", "recall_Cyclist"]
    acc = acc[cols].sort_values(["score_thr", "model"])
    acc.to_csv(out / "model_comparison.csv", index=False)
    print(acc.to_string(index=False)); print()

    # 2) Theo mức che khuất
    occ_rows = []
    for key, fname in [("pointpillars", "preds_kitti.json"), ("second", "preds_kitti_second.json")]:
        preds = json.load(open(out / fname))["frames"]
        gt = load_gt(args.data_root, sorted(preds))
        for level, (h, n) in recall_by_occlusion(preds, gt, args.data_root, args.thr, args.match_dist).items():
            occ_rows.append({"model": NAMES[key], "score_thr": args.thr, "level": level, "detected": h,
                             "n_gt": n, "recall": round(h / n, 3)})
    occ = pd.DataFrame(occ_rows)
    occ.to_csv(out / "model_comparison_occlusion.csv", index=False)
    print(occ.pivot_table(index="level", columns="model", values="recall").to_string()); print()

    # 3) Latency đo xen kẽ
    lat = None
    if not args.skip_latency:
        runs, gpu = measure_latency(args.data_root, args.warmup, args.runs)
        lat = runs.groupby("model")["ms"].agg(n_runs="size", p50_ms=lambda x: np.percentile(x, 50),
                                              p95_ms=lambda x: np.percentile(x, 95), max_ms="max").round(2)
        lat["gpu"] = gpu
        lat.reset_index().to_csv(out / "model_comparison_latency.csv", index=False)
        print(lat.to_string()); print()

    # Hình: precision-recall theo score_thr + latency
    fig, axes = plt.subplots(1, 3 if lat is not None else 2, figsize=(15, 4.2))
    for m, c in [("PointPillars", "tab:blue"), ("SECOND", "tab:orange")]:
        d = acc[acc["model"] == m]
        axes[0].plot(d["recall"], d["precision"], "o-", color=c, label=m)
        for _, r in d.iterrows():
            axes[0].annotate(f"{r['score_thr']}", (r["recall"], r["precision"]), textcoords="offset points",
                             xytext=(4, 4), fontsize=8, color=c)
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision"); axes[0].grid(alpha=.3); axes[0].legend()
    axes[0].set_title("Precision-recall theo score_thr (nhãn = ngưỡng)")
    d = acc[acc["score_thr"] == args.thr].set_index("model")
    x = np.arange(3); w = 0.38
    for i, (m, c) in enumerate([("PointPillars", "tab:blue"), ("SECOND", "tab:orange")]):
        axes[1].bar(x + i * w, [d.loc[m, f"recall_{k}"] for k in ["Car", "Pedestrian", "Cyclist"]], w, color=c, label=m)
    axes[1].set_xticks(x + w / 2); axes[1].set_xticklabels(["Car (n=72)", "Pedestrian (n=18)", "Cyclist (n=6)"])
    axes[1].set_ylim(0, 1.05); axes[1].set_title(f"Recall theo lớp @ score_thr={args.thr}"); axes[1].legend()
    if lat is not None:
        lt = lat.reset_index()
        axes[2].bar(np.arange(2) - 0.2, lt["p50_ms"], 0.4, label="p50", color="0.6")
        axes[2].bar(np.arange(2) + 0.2, lt["p95_ms"], 0.4, label="p95", color="0.3")
        axes[2].set_xticks(range(2)); axes[2].set_xticklabels(lt["model"]); axes[2].set_ylabel("ms")
        axes[2].set_title(f"Latency (20 frame × {args.runs} lần, đo xen kẽ)\n{lat['gpu'].iloc[0]}", fontsize=9)
        axes[2].legend()
    fig.tight_layout(); fig.savefig(out / "figures" / "model_comparison.png", dpi=120); plt.close(fig)
    print(f"-> {out}/model_comparison*.csv, {out}/figures/model_comparison.png")


if __name__ == "__main__":
    main()
