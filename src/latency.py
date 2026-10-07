"""CP3 — đo latency PointPillars trên GPU.

Cách đo:
  - Đo thời gian 1 lần gọi inference_detector (tiền xử lý + voxelize + mạng + NMS),
    điểm LiDAR đã nạp sẵn trong RAM (không tính đọc file).
  - torch.cuda.synchronize() trước khi bấm và trước khi dừng đồng hồ: GPU chạy bất đồng bộ,
    không synchronize thì đo thiếu.
  - Bỏ --warmup lần chạy đầu (khởi tạo CUDA/cuDNN chậm), rồi chạy --runs lần, báo p50/p95.
  - Chế độ 1: lặp trên CÙNG 1 frame (--frame). Chế độ 2: mỗi frame của data-root chạy --runs lần.

Chạy từ gốc repo:
    .venv-det/bin/python -m src.latency
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.infer import DEFAULT_CKPT, default_config
from starter.datasets import list_frames, load_frame


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def time_runs(model, points: np.ndarray, warmup: int, runs: int) -> list[float]:
    """Trả về list thời gian (ms) của `runs` lần chạy, sau khi bỏ `warmup` lần đầu."""
    from mmdet3d.apis import inference_detector
    times = []
    for k in range(warmup + runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inference_detector(model, points)
        torch.cuda.synchronize()
        if k >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return times


def main() -> None:
    ap = argparse.ArgumentParser(description="CP3: latency p50/p95 của PointPillars trên GPU")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame", default="000011", help="frame dùng cho chế độ lặp trên cùng 1 frame")
    ap.add_argument("--warmup", type=int, default=1, help="số lần chạy đầu bị bỏ")
    ap.add_argument("--runs", type=int, default=30, help="số lần đo (>= 20)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    from mmdet3d.apis import init_model
    torch.manual_seed(0); np.random.seed(0)
    model = init_model(args.config or default_config(), args.checkpoint, device="cuda:0")
    hw = {"gpu": torch.cuda.get_device_name(0), "cpu": cpu_name(), "torch": torch.__version__}

    def load(fid):
        p = load_frame(args.data_root, fid)["points"]
        return p[np.isfinite(p).all(axis=1)].astype(np.float32)

    rows, summary = [], []
    # Chế độ 1: cùng 1 frame
    pts = load(args.frame)
    t = time_runs(model, pts, args.warmup, args.runs)
    rows += [{"mode": "same_frame", "frame": args.frame, "n_points": len(pts), "run": i, "ms": v}
             for i, v in enumerate(t)]
    summary.append({"mode": f"same_frame ({args.frame})", "n_runs": len(t)})
    # Chế độ 2: mọi frame (model đã nóng từ chế độ 1, vẫn bỏ warmup mỗi frame cho đồng nhất)
    for fid in list_frames(args.data_root):
        pts = load(fid)
        t = time_runs(model, pts, args.warmup, args.runs)
        rows += [{"mode": "all_frames", "frame": fid, "n_points": len(pts), "run": i, "ms": v}
                 for i, v in enumerate(t)]
    summary.append({"mode": f"all_frames ({len(list_frames(args.data_root))} frame)", "n_runs": None})

    df = pd.DataFrame(rows)
    for s, mode in zip(summary, ["same_frame", "all_frames"]):
        ms = df[df["mode"] == mode]["ms"]
        s.update({"n_runs": len(ms), "warmup_dropped_per_frame": args.warmup,
                  "p50_ms": round(float(np.percentile(ms, 50)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2),
                  "mean_ms": round(float(ms.mean()), 2), "max_ms": round(float(ms.max()), 2), **hw})
    out = Path(args.out_dir)
    df.round(3).to_csv(out / "latency_runs.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "latency.csv", index=False)
    print(pd.DataFrame(summary).drop(columns=["cpu", "torch"]).to_string(index=False))
    print(f"CPU: {hw['cpu']} | torch {hw['torch']}")
    print(f"-> {out}/latency.csv, {out}/latency_runs.csv")


if __name__ == "__main__":
    main()
