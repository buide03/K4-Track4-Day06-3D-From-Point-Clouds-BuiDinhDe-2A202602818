"""Đổi box GT KITTI (camera frame) sang LiDAR frame để so với output của PointPillars.

Quy ước box LiDAR dùng ở đây giống MMDetection3D (LiDARInstance3DBoxes):
    [x, y, z, dx, dy, dz, yaw]
    - (x, y, z): tâm ĐÁY box trong velodyne frame (x trước, y trái, z lên)
    - dx = length (dọc hướng xe), dy = width, dz = height (mét)
    - yaw: góc quay quanh trục z, 0 = hướng theo +x (phía trước)
Công thức đổi yaw tham khảo CameraInstance3DBoxes.convert_to() của MMDetection3D
(https://github.com/open-mmlab/mmdetection3d, Apache-2.0).

Tự kiểm: python -m src.boxes --data-root data/kitti_mini --frame 000011
"""
from __future__ import annotations

import argparse

import numpy as np

from starter.datasets import load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import box3d_corners_cam


def cam_to_velo(points_cam: np.ndarray, calib: KittiCalib) -> np.ndarray:
    """Phép ngược của velo_to_cam: (N, 3) rectified camera -> (N, 3) velodyne."""
    T_velo_cam = np.linalg.inv(calib.T_cam_velo)  # 4x4, đảo chiều biến đổi
    pts_h = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    return (pts_h @ T_velo_cam.T)[:, :3]


def kitti_obj_to_lidar_box(obj: KittiObject, calib: KittiCalib) -> np.ndarray:
    """KittiObject (camera frame) -> box LiDAR [x, y, z, dx, dy, dz, yaw]."""
    h, w, l = obj.dimensions
    # location của KITTI đã là tâm đáy -> đổi điểm này sang LiDAR là ra tâm đáy
    x, y, z = cam_to_velo(obj.location.reshape(1, 3), calib)[0]
    # rotation_y quay quanh trục y-xuống của camera; yaw quay quanh trục z-lên của LiDAR.
    # Đổi trục làm đổi dấu, và rotation_y = 0 nghĩa là xe hướng sang phải (= -y LiDAR) -> lệch -90°.
    yaw = -obj.rotation_y - np.pi / 2
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi  # đưa về [-pi, pi)
    return np.array([x, y, z, l, w, h, yaw])


def lidar_box_corners(box: np.ndarray) -> np.ndarray:
    """8 góc (8, 3) của box LiDAR: 4 góc đáy rồi 4 góc nóc. 4 góc đầu dùng để vẽ BEV."""
    x, y, z, dx, dy, dz, yaw = box
    cx = np.array([dx, dx, -dx, -dx]) / 2
    cy = np.array([dy, -dy, -dy, dy]) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    bx, by = x + c * cx - s * cy, y + s * cx + c * cy  # xoay quanh z rồi dịch
    bottom = np.stack([bx, by, np.full(4, z)], axis=1)
    top = bottom + [0, 0, dz]
    return np.vstack([bottom, top])


def points_in_box(points: np.ndarray, box: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Mask (N,) các điểm LiDAR nằm trong box: đưa điểm về hệ toạ độ của box rồi so kích thước.

    LiDAR chỉ chạm BỀ MẶT vật, nên nhiều điểm nằm đúng trên mặt box; sai số calib vài mm
    đủ đẩy chúng ra ngoài. `margin` (mét) nới box mỗi phía để không đếm thiếu."""
    x, y, z, dx, dy, dz, yaw = box
    d = points[:, :3] - [x, y, z]
    c, s = np.cos(-yaw), np.sin(-yaw)
    lx, ly = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]
    m = margin
    return ((np.abs(lx) <= dx / 2 + m) & (np.abs(ly) <= dy / 2 + m)
            & (d[:, 2] >= -m) & (d[:, 2] <= dz + m))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tự kiểm phép đổi box GT camera -> LiDAR")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame", default="000011")
    ap.add_argument("--margin", type=float, default=0.1, help="nới box khi đếm điểm (mét)")
    args = ap.parse_args()

    fr = load_frame(args.data_root, args.frame)
    calib, pts = fr["calib"], fr["points"]
    pts = pts[np.isfinite(pts[:, :3]).all(axis=1)]
    print(f"{'type':<12}{'x':>7}{'y':>7}{'z':>7}{'yaw':>7}{'range':>7}{'corner_err':>12}{'n_pts':>7}")
    for obj in fr["labels"]:
        if obj.type == "DontCare":
            continue
        box = kitti_obj_to_lidar_box(obj, calib)
        # Kiểm tra 1: 8 góc tính trong camera rồi đổi sang LiDAR phải trùng 8 góc tính từ box LiDAR
        ref = cam_to_velo(box3d_corners_cam(obj), calib)
        mine = lidar_box_corners(box)
        # mỗi góc ref phải có một góc mine trùng nó (không phụ thuộc thứ tự góc)
        err = np.linalg.norm(ref[:, None] - mine[None], axis=2).min(axis=1).max()
        # Kiểm tra 2: box GT phải chứa điểm LiDAR (trừ vật quá xa hoặc bị che hết)
        n = int(points_in_box(pts, box, args.margin).sum())
        rng = float(np.hypot(box[0], box[1]))
        print(f"{obj.type:<12}{box[0]:7.1f}{box[1]:7.1f}{box[2]:7.2f}{np.degrees(box[6]):7.0f}"
              f"{rng:7.1f}{err:12.1e}{n:7d}")


if __name__ == "__main__":
    main()
