"""Chạy detector 3D có sẵn (mặc định PointPillars KITTI-3class; SECOND qua --config/--checkpoint) trên frame KITTI,
lưu box dự đoán ra JSON và vẽ ảnh BEV (điểm LiDAR + box GT xanh + box dự đoán đỏ).

Cách gọi model tham khảo demo/pcd_demo.py của MMDetection3D
(https://github.com/open-mmlab/mmdetection3d, Apache-2.0): init_model + inference_detector.

Chạy trong môi trường detector, từ gốc repo:
    .venv-det/bin/python -m src.infer --data-root data/kitti_mini
    .venv-det/bin/python -m src.infer --data-root data/kitti_mini --frames 000011 000004
    .venv-det/bin/python -m src.infer --model second    # SECOND, ghi ra preds_kitti_second.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # vẽ ra file, không cần cửa sổ (WSL)
import matplotlib.pyplot as plt
import numpy as np

from src.boxes import kitti_obj_to_lidar_box, lidar_box_corners
from starter.datasets import list_frames, load_frame
from starter.projection import project_velo_to_image

# Hai model KITTI-3class có sẵn trong model zoo MMDetection3D (link tải trong metafile.yml của từng thư mục config)
MODELS = {
    "pointpillars": ("pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py",
                     "checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"),
    "second": ("second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py",
               "checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth"),
}
DEFAULT_CKPT = MODELS["pointpillars"][1]
CLASSES = ["Pedestrian", "Cyclist", "Car"]  # thứ tự label id của cả 2 checkpoint


def default_config(model: str = "pointpillars") -> str:
    import mmdet3d
    return os.path.join(os.path.dirname(mmdet3d.__file__), ".mim", "configs", MODELS[model][0])


def run_model(model, points: np.ndarray) -> dict:
    """Chạy 1 frame. Trả về box (K, 7) [x, y, z, dx, dy, dz, yaw], score (K,), label (K,)."""
    from mmdet3d.apis import inference_detector
    res, _ = inference_detector(model, points.astype(np.float32))
    p = res.pred_instances_3d
    return {"boxes": p.bboxes_3d.tensor.cpu().numpy(),
            "scores": p.scores_3d.cpu().numpy(),
            "labels": p.labels_3d.cpu().numpy()}


def in_camera_fov(boxes: np.ndarray, calib, image_shape) -> np.ndarray:
    """True nếu tâm box chiếu được vào ảnh camera. GT KITTI chỉ gán nhãn vật trong ảnh,
    nên box ngoài FOV không thể so với GT (không tính là false positive)."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    centers = boxes[:, :3] + np.c_[np.zeros((len(boxes), 2)), boxes[:, 5] / 2]  # tâm đáy -> tâm box
    _, _, mask = project_velo_to_image(centers, calib, image_shape)
    return mask


def draw_bev(points, gt_boxes, gt_types, pred, out_path: Path, title: str, score_thr: float,
             pc_range, model_name: str = "PointPillars") -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    keep = (points[:, 0] > -5) & (points[:, 0] < 75) & (np.abs(points[:, 1]) < 42)
    p = points[keep]
    # BEV: trục ngang = y (đảo dấu để bên trái xe nằm bên trái hình), trục dọc = x (phía trước)
    ax.scatter(-p[:, 1], p[:, 0], s=0.2, c=p[:, 2], cmap="gray", vmin=-2.5, vmax=1)
    x0, y0, _, x1, y1, _ = pc_range
    ax.plot([-y0, -y1, -y1, -y0, -y0], [x0, x0, x1, x1, x0], "b--", lw=0.8, label="point_cloud_range")
    for box, t in zip(gt_boxes, gt_types):
        c = lidar_box_corners(box)[[0, 1, 2, 3, 0]]
        ax.plot(-c[:, 1], c[:, 0], "g-", lw=1.5)
        ax.text(-box[1], box[0] + 2.5, t, color="g", fontsize=7, ha="center")
    for box, s, lab in zip(pred["boxes"], pred["scores"], pred["labels"]):
        if s < score_thr:
            continue
        c = lidar_box_corners(box)[[0, 1, 2, 3, 0]]
        ax.plot(-c[:, 1], c[:, 0], "r-", lw=1)
        # đoạn thẳng từ tâm tới mặt trước = hướng yaw
        ax.plot([-box[1], -box[1] - np.sin(box[6]) * box[3] / 2],
                [box[0], box[0] + np.cos(box[6]) * box[3] / 2], "r-", lw=1)
        ax.text(-box[1], box[0] - 2.5, f"{CLASSES[lab][:3]} {s:.2f}", color="r", fontsize=6, ha="center")
    ax.plot([], [], "g-", label="GT (label_2)")
    ax.plot([], [], "r-", label=f"{model_name} score >= {score_thr}")
    ax.set_xlim(-42, 42); ax.set_ylim(-5, 75); ax.set_aspect("equal")
    ax.set_xlabel("-y LiDAR (m), trái <-> phải"); ax.set_ylabel("x LiDAR (m), phía trước")
    ax.set_title(title); ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chạy detector 3D có sẵn trên KITTI: lưu dự đoán + vẽ BEV")
    ap.add_argument("--model", choices=list(MODELS), default="pointpillars")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frames", nargs="*", help="mặc định: mọi frame trong data-root")
    ap.add_argument("--config", default=None, help="mặc định: config KITTI-3class của --model trong gói mmdet3d")
    ap.add_argument("--checkpoint", default=None, help="mặc định: checkpoint của --model trong checkpoints/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vis-thr", type=float, default=0.3, help="ngưỡng score khi vẽ BEV")
    ap.add_argument("--out", default=None, help="mặc định: results/preds_kitti.json (pointpillars) "
                                                "hoặc results/preds_kitti_<model>.json")
    ap.add_argument("--fig-dir", default=None, help="mặc định: results/figures/bev (pointpillars) "
                                                    "hoặc results/figures/bev_<model>")
    args = ap.parse_args()
    suffix = "" if args.model == "pointpillars" else f"_{args.model}"
    args.config = args.config or default_config(args.model)
    args.checkpoint = args.checkpoint or MODELS[args.model][1]
    args.out = args.out or f"results/preds_kitti{suffix}.json"
    args.fig_dir = args.fig_dir or f"results/figures/bev{suffix}"
    name = {"pointpillars": "PointPillars", "second": "SECOND"}[args.model]

    from mmdet3d.apis import init_model
    model = init_model(args.config, args.checkpoint, device=args.device)
    pc_range = model.cfg.point_cloud_range  # đọc từ config, mỗi model một vùng khác nhau
    frames = args.frames or list_frames(args.data_root)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    all_preds = {}
    for fid in frames:
        fr = load_frame(args.data_root, fid)
        pts = fr["points"][np.isfinite(fr["points"]).all(axis=1)]  # bỏ NaN trước khi vào model
        pred = run_model(model, pts)
        pred["in_fov"] = in_camera_fov(pred["boxes"], fr["calib"], fr["image"].shape)

        gt = [o for o in fr["labels"] if o.type != "DontCare"]
        gt_boxes = [kitti_obj_to_lidar_box(o, fr["calib"]) for o in gt]
        draw_bev(pts, gt_boxes, [o.type for o in gt], pred, Path(args.fig_dir) / f"bev_{fid}.png",
                 f"KITTI {fid} — {name} KITTI-3class", args.vis_thr, pc_range, name)

        all_preds[fid] = {k: v.tolist() for k, v in pred.items()}  # numpy -> list để ghi JSON
        n_vis = int((pred["scores"] >= args.vis_thr).sum())
        print(f"{fid}: {len(pred['scores'])} box (score>=0.1), {n_vis} box score>={args.vis_thr}, "
              f"{int(pred['in_fov'].sum())} box trong FOV camera, GT={len(gt)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    meta = {"config": os.path.basename(args.config), "checkpoint": args.checkpoint,
            "classes": CLASSES, "box_format": "x y z(bottom) dx dy dz yaw, LiDAR frame"}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "frames": all_preds}, f)
    print(f"-> {args.out}, ảnh BEV trong {args.fig_dir}/")


if __name__ == "__main__":
    main()
