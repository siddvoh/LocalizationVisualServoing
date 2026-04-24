# Localization for Visual Servoing

FoundationPose vs. EKF-based stereo pose tracking for manipulator visual servoing.

**Authors**

- Akanksha Singal, Carnegie Mellon University (`asingal2@andrew.cmu.edu`)
- Sidd Vohra, Carnegie Mellon University (`svohra@andrew.cmu.edu`)

Both authors contributed equally to this work.

## Overview

We compare three perception back-ends inside a shared visual servoing loop on real hardware (xArm 7 + ZED Mini, eye-in-hand):

- **Pipeline A (IBVS).** DINOv2 + SAM2 centroid driving a calibrated image Jacobian.
- **Pipeline B (EKF + DINOv2 / SAM2 / Depth Anything V2).** Monocular centroid plus periodic metric depth, fused through an EKF.
- **Pipeline C (EKF + FoundationPose).** FoundationPose 6-DoF pose per frame, translation fused via a direct-3D update.

The same PBVS controller, EKF core, camera, robot, and per-frame CSV logger are shared across pipelines so observed differences are attributable to the perception back-end.

## Repository layout

```
scripts/     Source for the three pipelines and the evaluation tooling
data/        Reference photos, reference masks, per-trial CSV logs
results/     Post-hoc ground-truth evaluation outputs
```
