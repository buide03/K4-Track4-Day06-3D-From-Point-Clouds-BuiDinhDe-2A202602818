# CLAUDE.md — Day 6 Lab, Topic B (3D detector baseline)

Individual lab. Student: Bui Dinh De, MSSV 2A202602818. Deadline extended to 12:00 2026-10-08 (UTC+7) (originally 23:59 2026-10-07).
Details live in README.md, TOPICS.md (§B), CHECKPOINTS.md, RUBRIC.md, SUBMISSION.md, RULES.md, data/README.md — read on demand, don't duplicate.

## Hard rules
- All own code in `src/`. In `starter/` only edit the 2 `TODO(CP2)` funcs in `projection.py` (`velo_to_cam`, `cam_to_image`).
- Never modify `data/`. Never commit checkpoints (`.pth/.pt/.ckpt`, keep in `checkpoints/`), extra data, archives, files >20 MB, `.env`, keys.
- No fabricated numbers/images. Everything reproducible from commands in REPORT §5, relative paths, fixed seeds.
- Commit per checkpoint as `CPx: <short desc>`; only commit/push when asked.
- Keep code simple and explainable (student is quizzed on every line). Credit open-source snippets (e.g. MMDet3D demo) in a header comment.

## Env
- WSL2, RTX 3050 Laptop 4 GB.
- `.venv` (Py 3.12): requirements.txt. `.venv-det` (Py 3.10, gitignored): torch 2.1.2+cu118, mmcv 2.1.0, mmdet 3.2.0, mmdet3d 1.4.0, spconv-cu118 (needed for SECOND weights), numpy<2. Torch wheels cached in `~/wheels/`.
- Model: PointPillars KITTI-3class. Config from the installed package: `<mmdet3d>/.mim/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py`; checkpoint `checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth` (URL in that dir's metafile.yml). Label ids: 0 Pedestrian, 1 Cyclist, 2 Car.
- KITTI GT only covers the camera FOV → restrict predictions to the FOV before counting FPs.

## Data (`starter.datasets.load_frame(root, id)` → points (N,4), calib P2/R0_rect/Tr_velo_to_cam, image BGR, labels in CAMERA frame)
- `data/kitti_mini`: 20 frames, 64-beam, LiDAR x-fwd/y-left. Main dataset for PointPillars-KITTI. Groups listed in data/README.md §3.1.
- `data/nuscenes_mini_subset`: 80 keyframes, 32-beam, 5 cols, axes x-right/y-fwd, no sweeps → axis/intensity mismatch with KITTI models (good Geometry/Preprocess failure).
- `data/synthetic`: 5 debug frames. CP2 sanity: point (10,0,0) → z_cam≈9.73, uv≈(614,175); NaN-safe.

## Topic B targets
- Basic: inference ≥5 frames + BEV/3D box images. Good: explain config (`point_cloud_range`, `voxel_size`, `class_names`, `score_thr`, NMS, sweeps, test pipeline); latency p50/p95 (drop warm-up, ≥20 runs, `cuda.synchronize`), box count, score histogram, box range, per-frame pass/fail vs GT. Advanced: 2 models or 2 configs.
- Failures to look for: missed ped/cyclist, FP near ego, wrong yaw, far objects dropped, slow.
- Bonus allowed: B1, B2, B4, B5, B6 (NOT B3 — latency is core). Max +10, total ≤100.

## Deliverables (checked by `python tools/check_submission.py`)
- `results/<snake_case>.csv`, `results/figures/*.png`, `results/figures/fail_NN_<desc>.png`.
- `report/REPORT.md`: headings `## 1. Claim`, `## 2. Evidence`, `## 3. Failure case`, `## 4. Khuyến nghị`, `## 5. Cách chạy lại`, `## 6. Khai báo sử dụng AI`; no `[ĐIỀN...]` left; valid `**MSSV:**`; 3–8 lines per section. Failure must name debug layer (I/O/Geometry/Time/Preprocess/Model/Metric) + runtime detection idea.

## Open items
- Repo name should be `BuiDinhDe-2A202602818-Track4-Day21` per SUBMISSION — confirm/rename.
- Bonus B1/B2/B4 committed (see git log). After pushing, submit the new `git rev-parse HEAD` on LMS before 12:00 2026-10-08.
