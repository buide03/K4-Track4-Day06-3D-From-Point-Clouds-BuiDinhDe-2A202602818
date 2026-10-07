"""CP4 — phân tích failure case của PointPillars (đọc results/preds_kitti.json, không chạy lại model).

1. Thống kê "khi nào model sai" trên cả 20 frame: recall theo mức che khuất (occluded trong label_2)
   và theo số điểm LiDAR nằm trong box GT -> results/failure_recall_by_occlusion.csv, ..._by_points.csv
2. fail_01: frame 000049 — xe bị che khuất (occluded >= 2) không được phát hiện   [lớp Model]
3. fail_02: frame 000011 — lọc FOV TRƯỚC khi ghép làm mất xe bị cắt ở mép ảnh     [lớp Metric]

Chạy từ gốc repo:
    .venv-det/bin/python -m src.failure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark import CLASSES, load_gt, match_frame
from src.boxes import cam_to_velo, lidar_box_corners, points_in_box
from starter.datasets import load_frame

OCC_NAME = {0: "0 không che", 1: "1 che một phần", 2: "2 che phần lớn", 3: "3 không rõ"}
PTS_BINS = [(0, 50), (50, 200), (200, np.inf)]


def gt_details(data_root, fid, pred, gt, thr, match_dist):
    """Mỗi GT chính của frame: object label, box LiDAR, số điểm LiDAR, có được phát hiện không."""
    fr = load_frame(data_root, fid)
    pts = fr["points"][np.isfinite(fr["points"][:, :3]).all(axis=1)]
    objs = [o for o in fr["labels"] if o.type in CLASSES]   # cùng thứ tự với load_gt
    matched = match_frame(pred, gt, thr, match_dist)["matched"]
    rows = []
    for o, (cls, box, rng), m in zip(objs, gt["main"], matched):
        rows.append({"frame": fid, "class": cls, "range_m": rng, "occluded": o.occluded,
                     "truncated": o.truncated, "n_points": int(points_in_box(pts, box, 0.1).sum()),
                     "detected": bool(m), "obj": o, "box": box})
    return fr, pts, rows


def bev_panel(ax, pts, rows, pred, thr, xlim, ylim, title):
    p = pts[(pts[:, 0] > xlim[0]) & (pts[:, 0] < xlim[1]) & (-pts[:, 1] > ylim[0]) & (-pts[:, 1] < ylim[1])]
    ax.scatter(-p[:, 1], p[:, 0], s=0.4, c="0.35")
    for r in rows:
        c = lidar_box_corners(r["box"])[[0, 1, 2, 3, 0]]
        col = "tab:green" if r["detected"] else "tab:red"
        ax.plot(-c[:, 1], c[:, 0], color=col, lw=2)
        if not r["detected"]:
            ax.text(-r["box"][1] + 1.2, r["box"][0], f"occ={r['occluded']}\n{r['n_points']} điểm",
                    color="tab:red", fontsize=7, va="center")
    for box, s in zip(pred["boxes"], pred["scores"]):
        if s >= thr:
            c = lidar_box_corners(np.array(box))[[0, 1, 2, 3, 0]]
            ax.plot(-c[:, 1], c[:, 0], color="tab:orange", lw=1, ls="--")
    ax.plot([], [], color="tab:green", lw=2, label="GT được phát hiện")
    ax.plot([], [], color="tab:red", lw=2, label="GT bị bỏ sót")
    ax.plot([], [], color="tab:orange", ls="--", label=f"dự đoán score>={thr}")
    ax.set_xlim(*ylim); ax.set_ylim(*xlim); ax.set_aspect("equal")
    ax.set_xlabel("-y LiDAR (m)"); ax.set_ylabel("x LiDAR (m), phía trước")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7, loc="lower left")


def image_panel(ax, fr, rows, title):
    img = fr["image"].copy()
    for r in rows:
        x1, y1, x2, y2 = (int(v) for v in r["obj"].bbox)
        col = (0, 170, 0) if r["detected"] else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        if not r["detected"]:
            cv2.putText(img, f"MISS occ={r['occluded']}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    ax.imshow(img[:, :, ::-1]); ax.axis("off"); ax.set_title(title, fontsize=10)


def fov_rays(calib, image_w, depth=40.0):
    """2 tia biên trái/phải của FOV camera, đổi sang LiDAR frame (để vẽ trên BEV)."""
    P = calib.P2
    fx, cx = P[0, 0], P[0, 2]
    ends = np.array([[(u - cx) * depth / fx, 0.0, depth] for u in (0, image_w)])
    origin = cam_to_velo(np.zeros((1, 3)), calib)[0]
    return origin, cam_to_velo(ends, calib)


def main() -> None:
    ap = argparse.ArgumentParser(description="CP4: thống kê failure + ảnh fail_01, fail_02")
    ap.add_argument("--preds", default="results/preds_kitti.json")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--match-dist", type=float, default=2.0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    preds = json.load(open(args.preds))["frames"]
    frames = sorted(preds)
    gt = load_gt(args.data_root, frames)
    out, fig_dir = Path(args.out_dir), Path(args.out_dir) / "figures"

    # 1) Khi nào model sai: recall theo mức che khuất và theo số điểm LiDAR (cả 20 frame)
    cache, all_rows = {}, []
    for fid in frames:
        cache[fid] = gt_details(args.data_root, fid, preds[fid], gt[fid], args.thr, args.match_dist)
        all_rows += cache[fid][2]
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("obj", "box")} for r in all_rows])
    by_occ = df.groupby("occluded").agg(n_gt=("detected", "size"), detected=("detected", "sum"),
                                        recall=("detected", "mean"), median_points=("n_points", "median")).reset_index()
    by_occ["occluded"] = by_occ["occluded"].map(OCC_NAME)
    df["points_bin"] = pd.cut(df["n_points"], [b[0] for b in PTS_BINS] + [np.inf], right=False,
                              labels=["<50", "50-199", ">=200"])
    by_pts = df.groupby("points_bin", observed=False).agg(n_gt=("detected", "size"), detected=("detected", "sum"),
                                                          recall=("detected", "mean")).reset_index()
    by_occ.round(3).to_csv(out / "failure_recall_by_occlusion.csv", index=False)
    by_pts.round(3).to_csv(out / "failure_recall_by_points.csv", index=False)
    df.drop(columns="points_bin").round(3).to_csv(out / "failure_gt_details.csv", index=False)
    print(f"score_thr={args.thr}, cả {len(frames)} frame, {len(df)} GT (Car/Ped/Cyc)")
    print(by_occ.round(3).to_string(index=False)); print()
    print(by_pts.round(3).to_string(index=False)); print()

    # 2) fail_01 — frame 000049, xe bị che khuất
    fid = "000049"
    fr, pts, rows = cache[fid]
    missed = [r for r in rows if not r["detected"]]
    print(f"[fail_01] {fid}: {len(rows)} GT, bỏ sót {len(missed)}:")
    for r in missed:
        print(f"   {r['class']} {r['range_m']:.1f} m, occluded={r['occluded']}, {r['n_points']} điểm LiDAR")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.9, 1]})
    image_panel(a1, fr, rows, f"KITTI {fid}: xanh = phát hiện, đỏ = BỎ SÓT (score_thr={args.thr})")
    bev_panel(a2, pts, rows, preds[fid], args.thr, (0, 40), (-20, 20), "BEV: xe bị che phía sau xe gần")
    fig.suptitle("fail_01 — PointPillars bỏ sót xe bị che khuất: ít điểm LiDAR, chỉ thấy một phần thân xe", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / "fail_01_occluded_cars_000049.png", dpi=110); plt.close(fig)

    # 3) fail_02 — lọc FOV trước khi ghép (cách đo cũ) so với sau khi sửa
    def recall_car_near(prefilter: bool) -> tuple[int, int]:
        hit = tot = 0
        for f in frames:
            p = preds[f]
            if prefilter:   # cách đo SAI ban đầu: bỏ box có tâm ngoài FOV trước khi ghép
                keep = [i for i, ok in enumerate(p["in_fov"]) if ok]
                p = {k: [p[k][i] for i in keep] for k in ("boxes", "scores", "labels", "in_fov")}
            m = match_frame(p, gt[f], args.thr, args.match_dist)["matched"]
            for (c, _, rng), ok in zip(gt[f]["main"], m):
                if c == "Car" and rng < 20:
                    tot += 1; hit += int(ok)
        return hit, tot
    old, new = recall_car_near(True), recall_car_near(False)
    pd.DataFrame([{"cach_do": "loc FOV truoc khi ghep (sai)", "car_0_20m_detected": old[0], "n_gt": old[1],
                   "recall": round(old[0] / old[1], 3)},
                  {"cach_do": "ghep moi box, FOV chi de dem FP (da sua)", "car_0_20m_detected": new[0], "n_gt": new[1],
                   "recall": round(new[0] / new[1], 3)}]).to_csv(out / "failure_fov_filter.csv", index=False)
    print(f"[fail_02] recall Car 0-20 m @ {args.thr}: lọc FOV trước khi ghép {old[0]}/{old[1]} "
          f"= {old[0]/old[1]:.3f}  ->  đã sửa {new[0]}/{new[1]} = {new[0]/new[1]:.3f}")

    fid = "000011"
    fr, pts, rows = cache[fid]
    P = preds[fid]
    car = min((r for r in rows if r["class"] == "Car"), key=lambda r: r["range_m"])   # xe gần, bị cắt mép ảnh
    d = [np.hypot(b[0] - car["box"][0], b[1] - car["box"][1]) for b in P["boxes"]]
    k = int(np.argmin(d))
    print(f"[fail_02] {fid}: Car {car['range_m']:.1f} m truncated={car['obj'].truncated:.2f}; dự đoán gần nhất "
          f"score={P['scores'][k]:.2f}, cách tâm GT {d[k]:.2f} m, in_fov={P['in_fov'][k]}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.9, 1]})
    img = fr["image"].copy()
    x1, y1, x2, y2 = (int(v) for v in car["obj"].bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(img, f"Car truncated={car['obj'].truncated:.2f}", (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
    a1.imshow(img[:, :, ::-1]); a1.axis("off")
    a1.set_title(f"KITTI {fid}: xe sát mép ảnh, phần lớn thân xe nằm ngoài khung hình", fontsize=10)
    q = pts[(pts[:, 0] > -3) & (pts[:, 0] < 15) & (pts[:, 1] > -2) & (pts[:, 1] < 12)]
    a2.scatter(-q[:, 1], q[:, 0], s=0.4, c="0.35")
    gc = lidar_box_corners(car["box"])[[0, 1, 2, 3, 0]]
    pc = lidar_box_corners(np.array(P["boxes"][k]))[[0, 1, 2, 3, 0]]
    a2.plot(-gc[:, 1], gc[:, 0], color="tab:green", lw=2, label="GT Car")
    a2.plot(-pc[:, 1], pc[:, 0], color="tab:orange", lw=1.5, ls="--", label=f"dự đoán Car score {P['scores'][k]:.2f}")
    a2.plot(-P["boxes"][k][1], P["boxes"][k][0], "x", color="tab:orange", ms=10, mew=2, label="tâm box dự đoán")
    o, ends = fov_rays(fr["calib"], fr["image"].shape[1])
    for e in ends:
        a2.plot([-o[1], -e[1]], [o[0], e[0]], "b--", lw=1)
    a2.plot([], [], "b--", label="biên FOV camera")
    a2.set_xlim(-12, 2); a2.set_ylim(-3, 15); a2.set_aspect("equal"); a2.legend(fontsize=7, loc="lower left")
    a2.set_xlabel("-y LiDAR (m)"); a2.set_ylabel("x LiDAR (m)")
    a2.set_title("Tâm box dự đoán nằm NGOÀI FOV -> bị loại trước khi ghép", fontsize=10)
    fig.suptitle(f"fail_02 — lỗi cách đo: lọc FOV trước khi ghép làm recall Car 0-20 m tụt "
                 f"{new[0]/new[1]:.2f} -> {old[0]/old[1]:.2f} dù model đoán đúng", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / "fail_02_truncated_car_fov_filter_000011.png", dpi=110); plt.close(fig)
    print(f"-> {fig_dir}/fail_01_occluded_cars_000049.png, fail_02_truncated_car_fov_filter_000011.png, "
          f"{out}/failure_*.csv")


if __name__ == "__main__":
    main()
