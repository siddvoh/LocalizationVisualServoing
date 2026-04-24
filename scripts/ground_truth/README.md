# Ground Truth via xArm TCP Corner Probing

Produces an external ground-truth pose for each target object in the robot
base frame, replacing "FoundationPose's own output as the reference" in
the localization error metric (which was flagged as a limitation in
`submissions/final/final.tex`).

## Why this is defensible

The only sensor involved is the xArm's joint encoders. No camera, no depth,
no pose-estimation network. That makes these numbers an *independent*
reference for evaluating all three pipelines (IBVS, DINOv2+SAM2+DepthAnythingV2,
FoundationPose).

## How it works

Your xArm 7 reports its TCP pose in the base frame to ~0.1 mm repeatability
via forward kinematics. Position the TCP directly above a known corner of
the object and press a key; the script records `arm.get_position()`. Repeat
for 4 corners of the top face. Solve Horn's method (rigid-body least-squares
fit) to recover the 6-DoF object pose in the base frame.

**Accuracy:** 2-5 mm, dominated by how precisely you can eyeball "TCP is ~1 mm
above this corner." Arm repeatability and SVD math contribute <1 mm each.

## Prerequisites

- xArm Python SDK installed (already available in the `foundationpose` conda env).
- Arm reachable at `192.168.1.241`.
- Object placed at P1 (tape mark), top face up, will not move during probing.
- `numpy` and `pyyaml` (standard in the env).

## One-time TCP decision

The "sight point" you eyeball over each corner is whatever the arm considers
its TCP. Two options:

1. **Leave the TCP at the flange (default).** Eyeball "flange center is above
   the corner." There is a constant flange-to-corner offset; Horn's method
   absorbs it into the translation output **as long as the arm orientation is
   held fixed across all 4 corners** (tool pointing straight down is ideal).
2. **Set the TCP to a sharper feature on the camera/gripper** that you can
   sight more precisely. Run `arm.set_tcp_offset([dx, dy, dz, 0, 0, 0])` once,
   or configure it in xArm Studio, before running the script. Recommended if
   you want sub-mm absolute accuracy on the recorded translation.

## Usage

```bash
cd ~/localization_for_visual_servoing
conda activate foundationpose
python ground_truth/record_corners.py --object cheez_it_box
```

First run per object: the script prompts for its dimensions (L, W, H in mm)
and caches them in `objects.yaml`. Subsequent runs reuse the cached values.

### The interactive flow

1. Place the box at P1, top face up. Long edge can point in any direction.
2. For each of 4 top-face corners (any consistent clockwise or CCW order):
   - In xArm Studio, jog the TCP to hover ~1 mm above the corner.
   - Keep the arm orientation fixed (tool pointing straight down).
   - Press ENTER in the script terminal to record.
3. Press `r` + ENTER to redo the last corner if you slipped.
4. The script prints per-corner residuals and edge-length consistency, then
   writes the output JSON. If the RMS residual or max edge error exceeds
   5 mm, you're offered a redo.

## Output

Written to `ground_truth/poses/<object>_<timestamp>.json` plus a
`<object>_latest.json` mirror for easy lookup.

Key fields:
- `centroid_base_mm` — object centroid in base frame (mm). **Use this as
  the ground-truth target for translation-error evaluation.**
- `pose_base_frame.translation_mm` / `rotation_matrix` — full 6-DoF pose
  (body frame → base frame).
- `base_corners_mm` — raw TCP readings at the 4 corners (lets you re-run
  the fit with different conventions if needed).
- `fit_residual_rms_mm` — quality indicator. <2 mm is excellent, <5 mm is
  fine, >5 mm means you probably mis-sighted a corner.

## Integrating with analyze_csvs.py

Replace the FoundationPose-as-reference path with something like:

```python
import json, numpy as np

gt = json.load(open(f"ground_truth/poses/{object_name}_latest.json"))
gt_centroid_mm = np.array(gt["centroid_base_mm"])

# Pipeline CSV columns assumed to be object estimate in base frame (mm).
pipeline_xyz_mm = pipeline_df[["obj_x_mm", "obj_y_mm", "obj_z_mm"]].to_numpy()
err_xyz_mm = pipeline_xyz_mm - gt_centroid_mm
err_norm_mm = np.linalg.norm(err_xyz_mm, axis=1)

final_err_mm = err_norm_mm[-100:].mean()   # last N frames after arm stopped
```

Note that the pipelines currently report object position in *camera frame*
(mm), not base frame. You'll need to multiply by `T_base_flange @ T_flange_camera`
(the hand-eye calibration) inside `analyze_csvs.py` before comparing against
the ground truth centroid. If that transform isn't already plumbed through,
the post-probe analysis has two steps: (1) this script, (2) a small
`ground_truth/transform_pipeline_output.py` that does the frame conversion.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `xArm get_version failed, code=...` | arm in error state or xArm Studio holding exclusive lock | clear errors in xArm Studio, close any active motion panels, retry |
| RMS residual > 5 mm | one corner mis-sighted, or arm orientation drifted between corners | accept the redo prompt; keep tool pointed straight down the whole time |
| Edge lengths don't match nominal | box isn't the size in `objects.yaml`, or probed wrong points | re-measure with calipers, edit `objects.yaml`, re-run |
| Box shifted mid-probing | table bump, curious lab visitor, cat | abort, re-place at P1, restart |

## Re-probing between trials

If the object is moved between trials (e.g., swapping objects at P1), re-run
the script for each (object, placement) pair. Each run produces a fresh
timestamped JSON; `<object>_latest.json` always points to the most recent.

If the object stays at P1 across multiple trials of the same pipeline but
different start poses, you only need to probe once — the ground truth is a
property of the object placement, not of the pipeline under test.
