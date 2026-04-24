#!/usr/bin/env python3
# DINOv2+SAM2 visual servoing, with EKF/FP/IBVS pipelines switchable via --mode
import csv
import json
import logging
import time
import sys
import threading
import os
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

THIRD_PARTY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third-party")
)

# auto-persisted depth calibration
DEPTH_SCALE_STATE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments",
                 "depth_scale.json")
)


def _save_depth_scale_state(scale, offset, depth_cal_m, d_rel_median=None,
                            reference=None, source="keypress_c"):
    """atomic tmp+rename; failures logged, not raised"""
    payload = {
        "scale": float(scale),
        "offset": float(offset),
        "depth_cal_m": float(depth_cal_m),
        "d_rel_median": float(d_rel_median) if d_rel_median is not None else None,
        "reference": reference,
        "source": source,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        os.makedirs(os.path.dirname(DEPTH_SCALE_STATE_PATH), exist_ok=True)
        tmp = DEPTH_SCALE_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, DEPTH_SCALE_STATE_PATH)
        logger.info("Depth calibration saved to %s", DEPTH_SCALE_STATE_PATH)
    except OSError as exc:
        logger.warning("Could not save depth calibration to %s: %s",
                       DEPTH_SCALE_STATE_PATH, exc)


def _load_depth_scale_state(path=DEPTH_SCALE_STATE_PATH):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if "scale" not in data:
            logger.warning("Depth-scale file %s is missing 'scale'; ignoring.",
                           path)
            return None
        return data
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read depth calibration from %s: %s",
                       path, exc)
        return None

from dinov2_match_segment import (
    extract_patch_features,
    compute_resnet_similarity,
    compute_similarity_map,
    combine_similarity_maps,
    similarity_to_bbox,
    refine_with_sam2,
    select_best_cc_by_similarity,
)

from negative_weighing import (
    MaskTracker,
    RobotController,
    ZedUndistorter,
    _robust_centroid,
    _mask_iou,
    _get_sam2,
    _get_depth_model,
    ROBOT_IP,
    MIN_RECORDING_DIM, ZED_RESOLUTION,
    NEG_POINT_COUNT,
    IOU_DRIFT_THRESH,
    PYZED_AVAILABLE,
    VS_RATE, VS_APPROACH, VS_SPEED, VS_MVACC,
)

try:
    import pyzed.sl as sl
except ImportError:
    sl = None

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    DEVICE = "cpu"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "EKF"))
from ekf_servo import PoseEKF, PBVSController, CameraIntrinsics, DepthScaler

DINOV2_THRESHOLD_PCT = 93
DINOV2_COLOR_WEIGHT  = 0.7
DINOV2_REDETECT_INTERVAL = 30
# reject masks > this frac of frame (anything big is probably desk/wall)
DINOV2_MASK_MAX_FRAC = 0.35
# on redetect-with-anchor, require strong sim AND centroid not teleporting
DINOV2_SIM_MIN_REDETECT       = 0.78
DINOV2_MAX_CENTROID_JUMP_PX   = 250
PATCH_SIZE = 14

# R_cam_to_robot maps cam coords -> robot base coords
CAM_ROT_PRESETS = {
    "identity": np.eye(3),
    # ZED Mini eye-in-hand on xArm looking forward
    "zed_forward": np.array([
        [0.0,  0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]),
}


class ReferenceModel:
    """one-time DINOv2/color feature cache for the reference image"""

    def __init__(self, ref_path: str):
        ref_full = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
        if ref_full is None:
            raise FileNotFoundError(f"Cannot load reference: {ref_path}")

        if ref_full.shape[2] == 4:
            self.ref_bgr = ref_full[:, :, :3]
            self.ref_alpha = ref_full[:, :, 3]
            fg_count = np.count_nonzero(self.ref_alpha > 128)
            logger.info(f"Reference: {self.ref_bgr.shape[1]}x{self.ref_bgr.shape[0]}, "
                        f"alpha present, fg pixels: {fg_count}")
        else:
            self.ref_bgr = ref_full
            self.ref_alpha = np.ones(ref_full.shape[:2], dtype=np.uint8) * 255
            logger.info("Reference: no alpha channel, treating entire image as object")

        logger.info("Extracting DINOv2 reference features...")
        self.ref_features, self.ref_proc_h, self.ref_proc_w = \
            extract_patch_features(self.ref_bgr, PATCH_SIZE)
        logger.info(f"  Reference features: {self.ref_features.shape}")

        ref_alpha_resized = cv2.resize(
            self.ref_alpha,
            (self.ref_proc_w, self.ref_proc_h),
            interpolation=cv2.INTER_NEAREST,
        )
        rh_patches = self.ref_proc_h // PATCH_SIZE
        rw_patches = self.ref_proc_w // PATCH_SIZE
        self.ref_mask_patches = np.zeros((rh_patches, rw_patches), dtype=np.uint8)
        for py in range(rh_patches):
            for px in range(rw_patches):
                patch_region = ref_alpha_resized[
                    py * PATCH_SIZE:(py + 1) * PATCH_SIZE,
                    px * PATCH_SIZE:(px + 1) * PATCH_SIZE,
                ]
                if np.mean(patch_region > 128) > 0.5:
                    self.ref_mask_patches[py, px] = 1

        fg_patches = np.count_nonzero(self.ref_mask_patches)
        logger.info(f"  Foreground patches: {fg_patches}/{self.ref_mask_patches.size}")

    def detect_in_scene(self, scene_bgr: np.ndarray):
        """returns (bbox, sam_mask, sam_score, sim_upscaled, sam_logits, sim_mean)"""
        sh, sw = scene_bgr.shape[:2]

        scene_features, scene_proc_h, scene_proc_w = \
            extract_patch_features(scene_bgr, PATCH_SIZE)

        dinov2_sim = compute_similarity_map(
            self.ref_features, self.ref_mask_patches, scene_features)

        resnet_sim = compute_resnet_similarity(
            self.ref_bgr, self.ref_alpha, scene_bgr,
            scene_proc_h, scene_proc_w, PATCH_SIZE)

        sim_map = combine_similarity_maps(
            dinov2_sim, resnet_sim,
            alpha=1.0 - DINOV2_COLOR_WEIGHT)

        bbox, _, sim_upscaled, sim_mean = similarity_to_bbox(
            sim_map, sh, sw, PATCH_SIZE, DINOV2_THRESHOLD_PCT)

        logger.info("DINOv2 bbox: %s, component mean-sim: %.3f",
                    bbox, sim_mean)

        if bbox is None:
            return None, None, 0.0, sim_upscaled, None, sim_mean

        sam_mask, sam_score, sam_logits = refine_with_sam2(
            scene_bgr, bbox, select_topmost=False, return_logits=True)

        # split multi-instance masks, keep best CC by DINOv2 sim
        cc_pick = select_best_cc_by_similarity(sam_mask, sim_upscaled)
        if cc_pick is not None and cc_pick["n_candidates"] > 1:
            logger.info(
                "SAM2 multi-CC filter: %d qualifying CCs → kept 1 "
                "(mean_sim=%.3f area=%d y_min=%d%s); dropped sims=%s",
                cc_pick["n_candidates"], cc_pick["mean_sim"],
                cc_pick["area"], cc_pick["y_min"],
                " via topmost-tiebreak" if cc_pick["tiebreak_used"] else "",
                [round(s, 3) for s in cc_pick["other_mean_sims"]])
            bbox = cc_pick["bbox"]
            sam_mask, sam_score, sam_logits = refine_with_sam2(
                scene_bgr, bbox, select_topmost=False, return_logits=True)

        mask_area = np.count_nonzero(sam_mask)
        logger.info("SAM2 mask: score=%.3f, area=%dpx (%.1f%%)",
                     sam_score, mask_area, 100.0 * mask_area / (sh * sw))

        return bbox, sam_mask, sam_score, sim_upscaled, sam_logits, sim_mean


_MAX_REDETECT_DEPTH = 1


def run_dinov2_pipeline(
    image_bgr: np.ndarray,
    ref_model: ReferenceModel,
    tracker: MaskTracker,
    _redetect_depth: int = 0,
) -> dict:
    """detect -> sam2 -> centroid; prev_logits > anchor_logits > fresh"""
    res = dict(mask_np=None, best_centroid=None, bbox=None, mode="track")
    h, w = image_bgr.shape[:2]

    no_prior = tracker.prev_logits is None and not tracker.anchor_locked
    periodic = tracker.frame_count > 0 and tracker.frame_count % DINOV2_REDETECT_INTERVAL == 0
    need_detection = no_prior or periodic

    if need_detection:
        logger.info("--- Running DINOv2 detection ---")
        res["mode"] = "detect"
        bbox, sam_mask, sam_score, sim_upscaled, sam_logits, sim_mean = \
            ref_model.detect_in_scene(image_bgr)
        res["bbox"] = bbox

        mask_frac = (
            np.count_nonzero(sam_mask) / float(h * w)
            if sam_mask is not None else 0.0
        )

        candidate_centroid = (
            _robust_centroid(sam_mask) if sam_mask is not None else None
        )

        # guards only apply once anchor is locked
        reject_reason = None
        if sam_mask is None or sam_score <= 0.5:
            reject_reason = "low SAM2 score or no mask"
        elif mask_frac >= DINOV2_MASK_MAX_FRAC:
            reject_reason = (
                f"oversized mask ({100.0 * mask_frac:.1f}% of frame, "
                f"max={100.0 * DINOV2_MASK_MAX_FRAC:.0f}%) — likely "
                f"background/desk")
        elif tracker.anchor_locked:
            # sim floor; -1.0 is the peak-fallback path
            if sim_mean < DINOV2_SIM_MIN_REDETECT:
                reject_reason = (
                    f"weak DINOv2 similarity ({sim_mean:.3f} < "
                    f"{DINOV2_SIM_MIN_REDETECT:.2f}) — not confident "
                    f"enough to overwrite locked anchor")
            # teleport check vs last known centroid
            elif (
                tracker.prev_centroid is not None
                and candidate_centroid is not None
            ):
                dx = candidate_centroid[0] - tracker.prev_centroid[0]
                dy = candidate_centroid[1] - tracker.prev_centroid[1]
                jump = float(np.hypot(dx, dy))
                if jump > DINOV2_MAX_CENTROID_JUMP_PX:
                    reject_reason = (
                        f"centroid teleport ({jump:.0f}px > "
                        f"{DINOV2_MAX_CENTROID_JUMP_PX}px) from "
                        f"{tracker.prev_centroid} to "
                        f"{candidate_centroid} — likely a spurious "
                        f"match on background")

        if reject_reason is None:
            res["mask_np"] = sam_mask
            res["best_centroid"] = candidate_centroid

            tracker.update(sam_mask, sam_logits, candidate_centroid)
            logger.info(
                f"DINOv2 detection done. Centroid: {candidate_centroid}, "
                f"sim_mean={sim_mean:.3f}, "
                f"anchor_locked: {tracker.anchor_locked}")
        else:
            logger.info("DINOv2 detection: rejected — %s", reject_reason)
            # don't wipe tracker on reject; keep propagating from anchor

        return res

    # sam2 propagation path

    def _try_redetect(reason: str):
        if _redetect_depth < _MAX_REDETECT_DEPTH:
            logger.info("SAM2: %s, triggering re-detection", reason)
            tracker.reset(keep_anchor=True)
            return run_dinov2_pipeline(
                image_bgr, ref_model, tracker,
                _redetect_depth=_redetect_depth + 1)
        logger.warning("Max re-detection depth reached (%s)", reason)
        return None

    try:
        pred = _get_sam2()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pred.set_image(rgb)

        point_coords = []
        point_labels = []

        pos_pts = tracker.sample_positive_points(h, w, n=1)
        if pos_pts is not None:
            point_coords.extend(pos_pts)
            point_labels.extend([1] * len(pos_pts))

        neg_pts = tracker.sample_negative_points(h, w, NEG_POINT_COUNT)
        if neg_pts is not None:
            point_coords.extend(neg_pts)
            point_labels.extend([0] * len(neg_pts))
            logger.debug("SAM2 propagation: +%d pos, +%d neg points",
                         len(pos_pts) if pos_pts is not None else 0,
                         len(neg_pts))

        sam_kwargs = dict(
            multimask_output=not tracker.anchor_locked,
            return_logits=True,
        )

        if point_coords:
            sam_kwargs["point_coords"] = np.array(point_coords, dtype=np.float32)
            sam_kwargs["point_labels"] = np.array(point_labels, dtype=np.int32)

        if tracker.prev_logits is not None:
            sam_kwargs["mask_input"] = tracker.prev_logits
        elif tracker.anchor_logits is not None:
            sam_kwargs["mask_input"] = tracker.anchor_logits
            logger.info("SAM2: using anchor logits as prior")

        masks, scores_sam, logits = pred.predict(**sam_kwargs)

        if masks is not None and len(masks) > 0:
            best_idx = int(np.argmax(scores_sam))
            mask_out = (masks[best_idx] > 0).astype(np.uint8) * 255
            tracker_logits = logits[best_idx:best_idx + 1]

            if tracker.prev_mask is not None:
                iou = _mask_iou(mask_out, tracker.prev_mask)
                if iou < IOU_DRIFT_THRESH:
                    fallback = _try_redetect(
                        f"IoU={iou:.2f} < {IOU_DRIFT_THRESH}")
                    if fallback is not None:
                        return fallback

            res["mask_np"] = mask_out
            centroid = _robust_centroid(mask_out)
            res["best_centroid"] = centroid

            tracker.update(mask_out, tracker_logits, centroid)
            logger.debug("SAM2 propagation: score=%.3f, centroid=%s",
                         scores_sam[best_idx], centroid)
        else:
            fallback = _try_redetect("no mask returned")
            if fallback is not None:
                return fallback
            tracker.update(None, None, None)

    except Exception as e:
        logger.exception("SAM2 propagation failed: %s", e)
        tracker.update(None, None, None)

    return res


class DINOv2CameraStreamer(threading.Thread):
    """capture -> segmentation -> overlay -> optional servo"""

    def __init__(self, cam_index: int, stop_event: threading.Event,
                 robot: RobotController,
                 ref_model: ReferenceModel,
                 ekf: PoseEKF,
                 pbvs: PBVSController,
                 depth_scaler: DepthScaler,
                 use_pyzed: bool = True):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.stop_event = stop_event
        self.robot = robot
        self.ref_model = ref_model
        self._use_pyzed = use_pyzed and PYZED_AVAILABLE

        self._latest_left = None
        self._frame_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self._result: dict = {}
        self._models_ready = threading.Event()
        self._tracker = MaskTracker()
        self._undistorter = None

        self._ekf = ekf
        self._pbvs = pbvs
        self._depth_scaler = depth_scaler
        # last commanded camera-frame velocity (for ekf egomotion)
        self._last_servo_vel_cam = np.zeros(3)

        # metric depth target when 'c' is pressed; overridden by --depth-cal-m
        self.depth_cal_m = 0.30

        # ekf | ibvs | foundationpose, set from --mode
        self.pipeline_mode = "ekf"

        self._fp = None
        self._fp_redetect_interval = 60
        self._fp_meas_noise = 0.008
        self._fp_frame_count = 0
        self._latest_depth_m = None

        self._last_robot_pos = None

        self.run_tag = ""

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._out_prefix_str = (args.out_prefix + "_") if getattr(args, "out_prefix", "") else ""
        self._csv_path = os.path.abspath(f"metrics_{self._out_prefix_str}{ts}.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp", "frame", "dt_ms", "mode", "pipeline", "run_tag",
            "centroid_u", "centroid_v",
            "ekf_x", "ekf_y", "ekf_z",
            "ekf_vx", "ekf_vy", "ekf_vz",
            "ekf_uncertainty",
            "depth_metric",
            "servo_dx_mm", "servo_dy_mm", "servo_dz_mm",
            "err_m", "iter_time_ms",
            "robot_x_mm", "robot_y_mm", "robot_z_mm",
            # raw FP translation before EKF smoothing (empty in non-FP rows)
            "fp_raw_x", "fp_raw_y", "fp_raw_z",
            # ibvs-only pixel error from image centre
            "err_px",
        ])
        self._frame_idx = 0
        self._last_servo_cmd = (0.0, 0.0, 0.0, 0.0)  # dx, dy, dz, err
        self._last_ibvs_err_px = 0.0
        self._reset_event = threading.Event()
        self._seg_thread = None
        # one-shot sign-sanity log, re-armed on full reset
        self._pbvs_sign_logged = False

        # safety / auto-exit (None = disabled)
        self._z_floor_mm = None

        self._auto_exit_converge_sec = None
        self._auto_exit_lost_sec = None
        self._auto_exit_max_sec = None
        # depth-based stop; works even for IBVS which has no err_m
        self._stop_depth_m = None
        self._t_servo_enabled_at = None
        self._t_last_err_above_dz = None
        self._t_last_centroid_seen = None
        self._auto_exit_reason = None

    def _get_frame(self):
        with self._frame_lock:
            if self._latest_left is not None:
                return self._latest_left.copy()
            return None

    def _full_reset(self):
        self._reset_event.set()

    def _handle_key(self, key: int) -> bool:
        if key == ord("q"):
            self.stop_event.set()
            return True
        if key == ord("v") and self.robot is not None:
            self.robot.enabled = not self.robot.enabled
            logger.info("Servo: %s", "ON" if self.robot.enabled else "OFF")
        elif key == ord("r"):
            self._tracker.reset(keep_anchor=True)
            logger.info("Tracker soft-reset (anchor kept)")
        elif key == ord("R"):
            self._full_reset()
        elif key == ord("c"):
            self._calibrate_depth()
        return False

    def _calibrate_depth(self):
        """one-point calibration: median masked rel-depth -> known metric Z"""
        with self.data_lock:
            res = dict(self._result)
        depth_rel = res.get("depth_np")
        mask_np = res.get("mask_np")
        if depth_rel is None or mask_np is None:
            logger.warning(
                "Depth calibration: no depth/mask available yet. "
                "Hold still and press 'c' again after a detection.")
            return
        m = mask_np > 0
        if not np.any(m):
            logger.warning("Depth calibration: empty mask, aborting.")
            return
        d_rel_median = float(np.percentile(depth_rel[m], 50))
        if abs(d_rel_median) < 1e-6:
            logger.warning(
                "Depth calibration: relative depth near zero (d=%.6f), "
                "aborting.", d_rel_median)
            return
        self._depth_scaler.calibrate(d_rel_median, self.depth_cal_m)
        logger.info(
            "Depth calibrated at Z=%.3fm (d_rel=%.4f) -> scale=%.6f",
            self.depth_cal_m, d_rel_median, self._depth_scaler.scale)
        _save_depth_scale_state(
            scale=self._depth_scaler.scale,
            offset=self._depth_scaler.offset,
            depth_cal_m=self.depth_cal_m,
            d_rel_median=d_rel_median,
            reference=getattr(self, "reference_path", None),
            source="keypress_c",
        )
        self._full_reset()

    def _run_calibration(self):
        logger.info("Calibration thread: waiting for models...")
        self._models_ready.wait()
        logger.info("Calibration thread: starting Y/Z Jacobian calibration.")
        self.robot.calibrate(self._get_frame)

    def _init_foundationpose(self, mesh_path, mesh_extents_m, fp_repo_dir,
                             weights_dir, est_refine_iter, track_refine_iter,
                             redetect_interval, meas_noise):
        from foundationpose_wrapper import (
            FoundationPoseWrapper, FoundationPoseUnavailable)
        try:
            self._fp = FoundationPoseWrapper(
                mesh_path=mesh_path,
                mesh_extents_m=mesh_extents_m,
                fp_repo_dir=fp_repo_dir,
                weights_dir=weights_dir,
                est_refine_iter=est_refine_iter,
                track_refine_iter=track_refine_iter,
            )
            self._fp.setup()
        except FoundationPoseUnavailable as exc:
            logger.error("FoundationPose unavailable, falling back to EKF "
                         "mode: %s", exc)
            self.pipeline_mode = "ekf"
            self._fp = None
            return
        self._fp_redetect_interval = int(redetect_interval)
        self._fp_meas_noise = float(meas_noise)
        logger.info("FoundationPose initialised "
                    "(est_iter=%d, track_iter=%d, redetect=%d frames).",
                    est_refine_iter, track_refine_iter, redetect_interval)

    def _camera_intrinsics_matrix(self) -> np.ndarray:
        cam = self._ekf.cam
        return np.array([
            [cam.fx, 0.0,    cam.cx],
            [0.0,    cam.fy, cam.cy],
            [0.0,    0.0,    1.0],
        ], dtype=np.float64)

    def _run_foundationpose_step(self, frame_bgr: np.ndarray,
                                 depth_m: np.ndarray) -> dict:
        """first frame: SAM2 mask + FP.register. subsequent: FP.track_one."""
        res = {"mode": "track", "mask_np": None, "best_centroid": None,
               "bbox": None, "fp_pose": None, "fp_position_cam": None}

        if self._fp is None:
            return res
        if depth_m is None:
            logger.warning("FP: no depth map available this frame, skipping.")
            return res

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        K = self._camera_intrinsics_matrix()

        need_register = (
            not self._fp.is_registered
            or self._fp_frame_count >= self._fp_redetect_interval)

        if need_register:
            # Reuse the existing DINOv2+SAM2 path to get a bool mask
            logger.info("FP: running DINOv2+SAM2 to get registration mask")
            dino_res = run_dinov2_pipeline(
                frame_bgr, self.ref_model, self._tracker)
            mask_np = dino_res.get("mask_np")
            res["bbox"] = dino_res.get("bbox")
            res["mask_np"] = mask_np
            if mask_np is None:
                logger.warning("FP: no mask produced, skipping registration.")
                return res
            ob_mask = mask_np > 0
            try:
                pose = self._fp.register(
                    rgb=rgb, depth_m=depth_m, ob_mask=ob_mask, K=K)
            except Exception as e:
                logger.exception("FP register failed: %s", e)
                return res
            self._fp_frame_count = 0
            res["mode"] = "detect"
        else:
            try:
                pose = self._fp.track(rgb=rgb, depth_m=depth_m, K=K)
            except Exception as e:
                logger.exception("FP track failed: %s", e)
                self._fp.reset()
                return res
            self._fp_frame_count += 1

        res["fp_pose"] = pose
        t_cam = pose[:3, 3].astype(np.float64)
        res["fp_position_cam"] = t_cam
        # Project the object origin to image coordinates for the overlay.
        if t_cam[2] > 1e-4:
            u = self._ekf.cam.fx * t_cam[0] / t_cam[2] + self._ekf.cam.cx
            v = self._ekf.cam.fy * t_cam[1] / t_cam[2] + self._ekf.cam.cy
            res["best_centroid"] = (int(round(u)), int(round(v)))
        return res

    def _refresh_fk(self):
        """Read FK from the robot if we don't already have it for this tick.

        The PBVS path caches FK whenever it successfully commands a move.
        The IBVS path doesn't expose its FK read, so we poll here at low
        rate (governed by VS_RATE in the servo loop) to keep a fresh
        end-effector position in the CSV.
        """
        if self.robot is None or self.robot._arm is None:
            return
        try:
            with self.robot._lock:
                pos = self.robot._get_pos()
            if pos is not None:
                self._last_robot_pos = pos
        except Exception:
            pass

    def _seg_loop(self):
        last_time = time.time()
        depth_run_counter = 0
        last_fk_time = 0.0

        while not self.stop_event.is_set():
            if self._reset_event.is_set():
                self._tracker.reset(keep_anchor=False)
                self._ekf.reset()
                self._last_servo_vel_cam = np.zeros(3)
                self._pbvs_sign_logged = False
                self._reset_event.clear()
                logger.info("Tracker + EKF FULL reset")

            with self._frame_lock:
                frame = self._latest_left.copy() if self._latest_left is not None else None
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                iter_start = time.time()
                dt = iter_start - last_time
                last_time = iter_start

                if self.pipeline_mode == "foundationpose" and self._fp is not None:
                    depth_m_frame = self._latest_depth_m
                    res = self._run_foundationpose_step(frame, depth_m_frame)
                else:
                    res = run_dinov2_pipeline(frame, self.ref_model, self._tracker)

                # carry last depth map so 'c' calibration doesn't race the 5-frame stride
                with self.data_lock:
                    prev_depth_np = (self._result.get("depth_np")
                                     if self._result else None)
                if prev_depth_np is not None:
                    res["depth_np"] = prev_depth_np

                robot_vel_cam = (self._last_servo_vel_cam
                                 if np.any(self._last_servo_vel_cam) else None)
                self._ekf.predict(dt, robot_vel_cam=robot_vel_cam)

                centroid = res.get("best_centroid")

                if self.pipeline_mode == "foundationpose":
                    fp_pos = res.get("fp_position_cam")
                    if fp_pos is not None and fp_pos[2] > 0.05:
                        self._ekf.update_3d_position(
                            float(fp_pos[0]), float(fp_pos[1]),
                            float(fp_pos[2]),
                            meas_noise_pos=self._fp_meas_noise)
                        res["depth_metric"] = float(fp_pos[2])
                else:
                    if centroid is not None:
                        u, v = float(centroid[0]), float(centroid[1])

                        # depth every 5 frames
                        depth_metric = None
                        depth_run_counter += 1
                        if depth_run_counter % 5 == 0 or not self._ekf.is_initialised:
                            dm = _get_depth_model()
                            if dm is not None:
                                mask_np = res.get("mask_np")
                                try:
                                    depth_rel = dm.infer_image(frame).astype(np.float32)
                                    res["depth_np"] = depth_rel
                                    if mask_np is not None:
                                        depth_metric = self._depth_scaler.estimate_object_depth(
                                            depth_rel, mask_np, percentile=50)
                                except Exception as e:
                                    logger.warning("Depth inference failed: %s", e)

                        if depth_metric is not None and depth_metric > 0.05:
                            self._ekf.update(u, v, depth_metric)
                            res["depth_metric"] = depth_metric
                        elif self._ekf.is_initialised:
                            self._ekf.update_2d_only(u, v)
                            res["depth_metric"] = None
                        else:
                            default_z = self._pbvs.target_depth
                            logger.info(f"EKF: seeding with default depth {default_z:.2f}m")
                            self._ekf.update(u, v, default_z)
                            res["depth_metric"] = None

                if self._ekf.is_initialised:
                    res["ekf_position"] = self._ekf.position
                    res["ekf_velocity"] = self._ekf.velocity
                    res["ekf_pixel"] = self._ekf.predicted_pixel()
                    res["ekf_uncertainty"] = self._ekf.position_uncertainty()

                with self.data_lock:
                    self._result = res

                if not self._models_ready.is_set():
                    logger.info("Models ready, calibration may now proceed.")
                    self._models_ready.set()

                # ekf/fp -> pbvs, ibvs -> raw centroid
                if self.robot is not None:
                    ekf_pos = res.get("ekf_position")
                    use_pbvs = (
                        self.pipeline_mode in ("ekf", "foundationpose")
                        and ekf_pos is not None)
                    if use_pbvs:
                        self._servo_step_pbvs(ekf_pos)
                    else:
                        raw_centroid = res.get("best_centroid")
                        if raw_centroid is not None:
                            self.robot.servo_step(raw_centroid, frame.shape)
                        # mirror ibvs command into logger state; err_m stays 0
                        ibvs_cmd = getattr(self.robot,
                                           "_last_ibvs_cmd",
                                           (0.0, 0.0, 0.0, 0.0))
                        dx_i, dy_i, dz_i, err_px_i = ibvs_cmd
                        self._last_servo_cmd = (dx_i, dy_i, dz_i, 0.0)
                        self._last_ibvs_err_px = float(err_px_i)

                # refresh FK so IBVS trajectories log
                if iter_start - last_fk_time > VS_RATE:
                    self._refresh_fk()
                    last_fk_time = iter_start

                self._eval_auto_exit(centroid_present=(centroid is not None))

                iter_ms = (time.time() - iter_start) * 1000.0
                self._frame_idx += 1
                ekf_pos = res.get("ekf_position")
                ekf_vel = res.get("ekf_velocity")
                dx, dy, dz, err = self._last_servo_cmd

                def _fmt_vec(vec, fmt=".6f"):
                    if vec is not None:
                        return [f"{v:{fmt}}" for v in vec[:3]]
                    return ["", "", ""]

                def _fmt_opt(key, fmt=None):
                    val = res.get(key)
                    if val is None:
                        return ""
                    return f"{val:{fmt}}" if fmt else str(val)

                rx = ry = rz = ""
                if self._last_robot_pos is not None:
                    rx = f"{self._last_robot_pos[0]:.2f}"
                    ry = f"{self._last_robot_pos[1]:.2f}"
                    rz = f"{self._last_robot_pos[2]:.2f}"

                # Raw FoundationPose translation pre-filter, or empty
                fp_raw = res.get("fp_position_cam")
                fp_raw_cols = _fmt_vec(fp_raw)

                # err_px only for ibvs rows
                err_px_col = (f"{self._last_ibvs_err_px:.2f}"
                              if self.pipeline_mode == "ibvs" else "")

                self._csv_writer.writerow([
                    f"{iter_start:.6f}",
                    self._frame_idx,
                    f"{dt * 1000.0:.1f}",
                    res.get("mode", "track"),
                    self.pipeline_mode,
                    self.run_tag,
                    centroid[0] if centroid else "",
                    centroid[1] if centroid else "",
                    *_fmt_vec(ekf_pos),
                    *_fmt_vec(ekf_vel),
                    _fmt_opt("ekf_uncertainty"),
                    _fmt_opt("depth_metric"),
                    f"{dx:.2f}", f"{dy:.2f}", f"{dz:.2f}",
                    f"{err:.6f}",
                    f"{iter_ms:.1f}",
                    rx, ry, rz,
                    *fp_raw_cols,
                    err_px_col,
                ])
                self._csv_file.flush()

            except Exception:
                logger.exception("Pipeline error")
                time.sleep(1)

        self._csv_file.close()
        logger.info("Metrics saved: %s", self._csv_path)

    def _eval_auto_exit(self, centroid_present: bool):
        """sets stop_event on converge/lost/max-time, no-op if disabled"""
        if self.stop_event.is_set():
            return
        if self.robot is None or not getattr(self.robot, "enabled", False):
            self._t_servo_enabled_at = None
            return

        now = time.time()
        if self._t_servo_enabled_at is None:
            self._t_servo_enabled_at = now
            self._t_last_err_above_dz = now
            self._t_last_centroid_seen = now

        if centroid_present:
            self._t_last_centroid_seen = now

        # depth-based converge override works even for ibvs
        if self._stop_depth_m is not None:
            if self._ekf.is_initialised:
                ekf_z_m = float(self._ekf.position[2])
                if ekf_z_m > self._stop_depth_m:
                    self._t_last_err_above_dz = now
            else:
                self._t_last_err_above_dz = now

        if (self._auto_exit_max_sec is not None
                and now - self._t_servo_enabled_at >= self._auto_exit_max_sec):
            self._auto_exit_reason = (
                f"max-time timeout ({self._auto_exit_max_sec:.1f}s) reached")
        elif (self._auto_exit_lost_sec is not None
              and self._t_last_centroid_seen is not None
              and now - self._t_last_centroid_seen >= self._auto_exit_lost_sec):
            dt = now - self._t_last_centroid_seen
            self._auto_exit_reason = (
                f"object out of view: no centroid for {dt:.1f}s "
                f">= {self._auto_exit_lost_sec:.1f}s")
        elif (self._auto_exit_converge_sec is not None
              and self._t_last_err_above_dz is not None
              and now - self._t_last_err_above_dz
              >= self._auto_exit_converge_sec):
            dt = now - self._t_last_err_above_dz
            if self._stop_depth_m is not None:
                ekf_z_m = (float(self._ekf.position[2])
                           if self._ekf.is_initialised else float("nan"))
                self._auto_exit_reason = (
                    f"converged (depth): ekf_z={ekf_z_m:.3f}m <= "
                    f"stop_depth={self._stop_depth_m:.3f}m for {dt:.1f}s "
                    f">= {self._auto_exit_converge_sec:.1f}s")
            else:
                self._auto_exit_reason = (
                    f"converged (camera-frame): err_m < dead_zone "
                    f"for {dt:.1f}s >= {self._auto_exit_converge_sec:.1f}s")

        if self._auto_exit_reason is not None:
            logger.info("AUTO-EXIT tripped: %s. Stopping servo loop.",
                        self._auto_exit_reason)
            self.stop_event.set()

    def _servo_step_pbvs(self, ekf_position: np.ndarray):
        """Position-based servo step using EKF-filtered 3D pose."""
        if self.robot._arm is None or not self.robot.enabled:
            return

        now = time.time()
        # First real servo step per enable: arm the auto-exit timers
        # (converge, lost, max-sec) from here rather than program start,
        # so perception-only warmup time doesn't burn the budget.
        if self._t_servo_enabled_at is None:
            self._t_servo_enabled_at = now
            self._t_last_err_above_dz = now
            self._t_last_centroid_seen = now

        if now - self.robot._last_t < VS_RATE:
            return
        self.robot._last_t = now

        v_robot, v_cam, err_m = self._pbvs.compute_velocity(ekf_position)
        # Update convergence timer: the "time since error last exceeded
        # the dead zone" is what the camera-frame converge auto-exit
        # tests against. Skipped when --stop-depth-m is in use, since
        # _eval_auto_exit drives the timer from depth instead.
        if (self._stop_depth_m is None
                and err_m >= self._pbvs.dead_zone_m):
            self._t_last_err_above_dz = now
        if err_m < self._pbvs.dead_zone_m:
            self._last_servo_vel_cam = np.zeros(3)
            self._last_servo_cmd = (0.0, 0.0, 0.0, err_m)
            return

        dt_step = VS_RATE
        dx_mm = float(v_robot[0]) * dt_step * 1000.0
        dy_mm = float(v_robot[1]) * dt_step * 1000.0
        dz_mm = float(v_robot[2]) * dt_step * 1000.0

        # Constant approach along robot X (added after PBVS velocity)
        dx_mm += VS_APPROACH

        # One-shot sign sanity log on the first real servo step per session.
        # one-shot sign-trace dump on first call
        if not self._pbvs_sign_logged:
            ex, ey, ez = float(ekf_position[0] - 0.0), \
                         float(ekf_position[1] - 0.0), \
                         float(ekf_position[2] - self._pbvs.target_depth)
            logger.info(
                "PBVS sign trace (first step): "
                "obj_cam=(%.3f,%.3f,%.3f)m  err_cam=(%+.3f,%+.3f,%+.3f)m  "
                "v_cam=(%+.3f,%+.3f,%+.3f)m/s  v_robot=(%+.3f,%+.3f,%+.3f)m/s  "
                "delta_mm=(%+.1f,%+.1f,%+.1f) (incl. approach=%+.1f)",
                ekf_position[0], ekf_position[1], ekf_position[2],
                ex, ey, ez,
                v_cam[0], v_cam[1], v_cam[2],
                v_robot[0], v_robot[1], v_robot[2],
                dx_mm, dy_mm, dz_mm, VS_APPROACH)
            logger.info(
                "  Expected signs (zed_forward mount): "
                "obj-left  (err_x<0) -> dy_mm>0  |  "
                "obj-above (err_y<0) -> dz_mm>0  |  "
                "obj-far   (err_z>0) -> dx_mm>0")
            self._pbvs_sign_logged = True

        # Store camera-frame velocity for EKF egomotion compensation.
        # The EKF predict step expects velocity in camera coords, so we
        # convert the approach component (robot X) through R_cam_to_robot.T
        # and sum with the PBVS camera-frame velocity.
        v_approach_robot = np.array(
            [VS_APPROACH / (dt_step * 1000.0), 0.0, 0.0])
        v_approach_cam = self._pbvs.robot_vel_to_cam(v_approach_robot)
        self._last_servo_vel_cam = v_cam + v_approach_cam
        self._last_servo_cmd = (dx_mm, dy_mm, dz_mm, err_m)

        with self.robot._lock:
            try:
                pos = self.robot._get_pos()
                if pos is None:
                    return
                self._last_robot_pos = pos

                # Z floor safety clamp: never command the end-effector
                # below self._z_floor_mm. If the controller would, absorb
                # the illegal delta into dz_mm so the CSV log reflects
                # what was actually sent to the arm.
                tgt_z = pos[2] + dz_mm
                if self._z_floor_mm is not None and tgt_z < self._z_floor_mm:
                    clamped = self._z_floor_mm
                    if not getattr(self, "_z_floor_warned", False):
                        logger.warning(
                            "Z floor engaged: clamped commanded z "
                            "%.1f -> %.1f mm (floor=%.1f).",
                            tgt_z, clamped, self._z_floor_mm)
                        self._z_floor_warned = True
                    dz_mm = clamped - pos[2]
                    tgt_z = clamped
                    self._last_servo_cmd = (dx_mm, dy_mm, dz_mm, err_m)

                self.robot._arm.set_position(
                    x=pos[0] + dx_mm, y=pos[1] + dy_mm,
                    z=tgt_z,
                    roll=pos[3], pitch=pos[4], yaw=pos[5],
                    speed=VS_SPEED, mvacc=VS_MVACC, wait=True)
                logger.info(
                    f"PBVS: pos=({ekf_position[0]:.3f},"
                    f"{ekf_position[1]:.3f},{ekf_position[2]:.3f})m  "
                    f"err={err_m:.4f}m  "
                    f"delta=({dx_mm:+.1f},{dy_mm:+.1f},{dz_mm:+.1f})mm")
            except Exception as e:
                logger.error("PBVS step failed: %s", e)

    def run(self):
        if self._use_pyzed:
            self._run_pyzed()
        else:
            self._run_opencv()

    def _run_pyzed(self):
        cam = sl.Camera()

        init_params = sl.InitParameters()
        res_map = {
            "HD2K":   sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720":  sl.RESOLUTION.HD720,
            "VGA":    sl.RESOLUTION.VGA,
        }
        init_params.camera_resolution = res_map.get(ZED_RESOLUTION,
                                                     sl.RESOLUTION.HD720)
        init_params.camera_fps = 30
        if self.pipeline_mode == "foundationpose":
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL
            init_params.coordinate_units = sl.UNIT.METER
        else:
            init_params.depth_mode = sl.DEPTH_MODE.NONE

        err = cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            logger.info(f"PyZED: camera open failed ({err}) — falling back to OpenCV")
            self._run_opencv()
            return

        try:
            cal = cam.get_camera_information().calibration_parameters.left_cam
            self._ekf.cam.update(cal.fx, cal.fy, cal.cx, cal.cy)
            logger.info(
                "ZED intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                cal.fx, cal.fy, cal.cx, cal.cy)
        except Exception as e:
            logger.warning(
                "Failed to read ZED intrinsics, keeping defaults: %s", e)

        ts = time.strftime("%Y%m%d_%H%M%S")
        anno_path = os.path.abspath(f"vs_dinov2_{self._out_prefix_str}{ts}.mp4")
        video_writer = None

        mat = sl.Mat()
        depth_mat = sl.Mat() if self.pipeline_mode == "foundationpose" else None
        runtime_params = sl.RuntimeParameters()

        self._seg_thread = threading.Thread(target=self._seg_loop, daemon=True)
        self._seg_thread.start()
        if self.robot is not None and self.robot._arm is not None:
            threading.Thread(target=self._run_calibration, daemon=True).start()

        win = "DINOv2 Servo  |  [v] servo  [c] cal-depth  [r] reset  [q] quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        try:
            while not self.stop_event.is_set():
                if cam.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.01)
                    continue

                cam.retrieve_image(mat, sl.VIEW.LEFT)
                frame_bgra = mat.get_data()
                frame = frame_bgra[:, :, :3].copy()

                # zero out nan/inf depth for FP validity check
                if depth_mat is not None:
                    cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                    d = np.asarray(depth_mat.get_data(), dtype=np.float32)
                    d = np.where(np.isfinite(d), d, 0.0)
                    self._latest_depth_m = d

                with self._frame_lock:
                    self._latest_left = frame.copy()
                with self.data_lock:
                    res = dict(self._result)

                rendered = self._render(frame, res)
                rendered_write = _ensure_min_dim(rendered)

                if video_writer is None:
                    rh, rw = rendered_write.shape[:2]
                    video_writer = cv2.VideoWriter(
                        anno_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, (rw, rh))
                    if video_writer.isOpened():
                        logger.info(f"Recording → {anno_path}  ({rw}x{rh})")
                    else:
                        video_writer.release()
                        video_writer = None

                if video_writer is not None:
                    video_writer.write(rendered_write)
                cv2.imshow(win, rendered)

                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key):
                    break
        finally:
            cam.close()
            if video_writer is not None:
                video_writer.release()
                logger.info("Recording saved: %s", anno_path)
            if self._seg_thread is not None:
                self._seg_thread.join(timeout=3)
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _run_opencv(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            logger.info(f"Failed to open camera {self.cam_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self._seg_thread = threading.Thread(target=self._seg_loop, daemon=True)
        self._seg_thread.start()
        if self.robot is not None and self.robot._arm is not None:
            threading.Thread(target=self._run_calibration, daemon=True).start()

        win = "DINOv2 Servo  |  [v] servo  [c] cal-depth  [r] reset  [q] quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        video_writer = None
        ts = time.strftime("%Y%m%d_%H%M%S")

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                fh, fw = frame.shape[:2]
                left = frame[:, :fw // 2].copy()

                if self._undistorter is None:
                    lh, lw = left.shape[:2]
                    self._undistorter = ZedUndistorter.from_frame_size(lw, lh)
                left = self._undistorter.undistort(left)

                with self._frame_lock:
                    self._latest_left = left.copy()
                with self.data_lock:
                    res = dict(self._result)

                rendered = self._render(left, res)
                rendered_write = _ensure_min_dim(rendered)

                if video_writer is None:
                    rh, rw = rendered_write.shape[:2]
                    video_path = os.path.abspath(f"vs_dinov2_{self._out_prefix_str}{ts}.mp4")
                    video_writer = cv2.VideoWriter(
                        video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, (rw, rh))
                    if video_writer.isOpened():
                        logger.info("Recording: %s (%dx%d)", video_path, rw, rh)
                    else:
                        video_writer.release()
                        video_writer = None

                if video_writer is not None:
                    video_writer.write(rendered_write)
                cv2.imshow(win, rendered)

                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key):
                    break
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if self._seg_thread is not None:
                self._seg_thread.join(timeout=3)
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _render(self, left: np.ndarray, res: dict) -> np.ndarray:
        display = left.copy()
        h, w = display.shape[:2]

        mask_np = res.get("mask_np")
        if mask_np is not None and mask_np.shape[:2] != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        if mask_np is not None:
            overlay = np.zeros_like(display)
            overlay[:, :, 1] = mask_np
            display = cv2.addWeighted(display, 0.72, overlay, 0.28, 0)
            cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, cnts, -1, (0, 255, 0), 2)

        bbox = res.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        centroid = res.get("best_centroid")
        if centroid is not None:
            cx, cy = centroid
            cv2.drawMarker(display, (cx, cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            ic_x, ic_y = w // 2, h // 2
            cv2.drawMarker(display, (ic_x, ic_y), (255, 0, 0),
                           cv2.MARKER_CROSS, 15, 1)
            cv2.line(display, (ic_x, ic_y), (cx, cy), (255, 255, 0), 1)
            err = np.hypot(cx - ic_x, cy - ic_y)
            cv2.putText(display, f"err={err:.0f}px", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        ekf_pixel = res.get("ekf_pixel")
        if ekf_pixel is not None:
            eu, ev = ekf_pixel
            ekf_unc = res.get("ekf_uncertainty")
            if ekf_unc is not None:
                radius = max(5, int(ekf_unc * 500))
                cv2.circle(display, (eu, ev), radius,
                           (255, 100, 100), 1, cv2.LINE_AA)
            cv2.drawMarker(display, (eu, ev), (255, 50, 50),
                           cv2.MARKER_TILTED_CROSS, 18, 2)

        status_lines = [
            f"pipeline: {self.pipeline_mode.upper()}",
            f"anchor: {'LOCKED' if self._tracker.anchor_locked else 'no'}",
            f"frames: {self._tracker.frame_count}",
        ]
        if self.pipeline_mode == "foundationpose":
            fp_registered = bool(self._fp is not None and self._fp.is_registered)
            status_lines.append(
                f"FP: {'REG' if fp_registered else 'NOT-REG'} "
                f"({self._fp_frame_count}/{self._fp_redetect_interval})")
            fp_raw = res.get("fp_position_cam")
            if fp_raw is not None:
                status_lines.append(
                    f"FP raw: X={fp_raw[0]:.3f} Y={fp_raw[1]:.3f} "
                    f"Z={fp_raw[2]:.3f}m")
        if self.robot is not None:
            status_lines.append(f"servo: {'ON' if self.robot.enabled else 'OFF'}")
            status_lines.append(f"cal: {self.robot.cal_status}")

        ekf_pos = res.get("ekf_position")
        if ekf_pos is not None:
            status_lines.append(
                f"EKF: X={ekf_pos[0]:.3f} Y={ekf_pos[1]:.3f} Z={ekf_pos[2]:.3f}m")
        depth_m = res.get("depth_metric")
        if depth_m is not None:
            status_lines.append(f"depth: {depth_m:.3f}m")
        ekf_unc = res.get("ekf_uncertainty")
        if ekf_unc is not None:
            status_lines.append(f"unc: {ekf_unc:.4f}")

        for i, line in enumerate(status_lines):
            cv2.putText(display, line, (10, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display


def _ensure_min_dim(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    short = min(h, w)
    if short < MIN_RECORDING_DIM:
        scale = MIN_RECORDING_DIM / short
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LINEAR)
    return img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DINOv2-guided Visual Servoing")
    parser.add_argument("--scene-source", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--reference", "-r", required=True,
                        help="Path to reference image (RGBA with transparent bg)")
    parser.add_argument("--no-pyzed", action="store_true",
                        help="Force OpenCV capture even if PyZED is available")
    parser.add_argument("--no-robot", action="store_true",
                        help="Disable robot connection (vision-only mode)")
    parser.add_argument("--threshold-pct", type=float, default=DINOV2_THRESHOLD_PCT,
                        help=f"Similarity threshold percentile (default: {DINOV2_THRESHOLD_PCT})")
    parser.add_argument("--color-weight", type=float, default=DINOV2_COLOR_WEIGHT,
                        help=f"Color similarity weight (default: {DINOV2_COLOR_WEIGHT})")
    parser.add_argument("--redetect-interval", type=int, default=DINOV2_REDETECT_INTERVAL,
                        help=f"Re-run DINOv2 every N frames (default: {DINOV2_REDETECT_INTERVAL})")
    parser.add_argument("--cam-to-robot", choices=list(CAM_ROT_PRESETS.keys()),
                        default="identity",
                        help="Camera-to-robot rotation preset "
                             "(default: identity; use 'zed_forward' for a "
                             "typical eye-in-hand ZED Mini mount)")
    parser.add_argument("--depth-cal-m", type=float, default=0.30,
                        help="Known metric depth used when calibrating "
                             "DepthScaler with keypress 'c' (default: 0.30 m)")
    parser.add_argument("--mode",
                        choices=["ekf", "ibvs", "foundationpose"],
                        default="ekf",
                        help="Control pipeline: 'ekf' (DINOv2+SAM2 -> EKF-PBVS, "
                             "default), 'ibvs' (force raw IBVS baseline), "
                             "or 'foundationpose' (FoundationPose 6-DoF -> "
                             "EKF-PBVS).")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Free-form tag written to every CSV row "
                             "(e.g. 'cheezit_pose1_trial2').")
    parser.add_argument("--out-prefix", type=str, default="",
                        help="Prefix for output filenames (CSV and video). "
                             "E.g. 'cheez_it_box_ekf'.")
    # FoundationPose-specific flags (ignored unless --mode foundationpose)
    parser.add_argument("--fp-mesh", type=str, default=None,
                        help="Path to OBJ/PLY CAD mesh (metres). Required "
                             "for --mode foundationpose unless --fp-box is "
                             "given.")
    parser.add_argument("--fp-box", type=float, nargs=3, default=None,
                        metavar=("W", "H", "D"),
                        help="Build a procedural box mesh with these "
                             "extents in metres (width height depth).")
    parser.add_argument("--fp-repo-dir", type=str, default=None,
                        help="Path to the NVLabs/FoundationPose repo "
                             "(overrides $FOUNDATIONPOSE_ROOT).")
    parser.add_argument("--fp-weights-dir", type=str, default=None,
                        help="Path to the FoundationPose weights directory.")
    parser.add_argument("--fp-est-iter", type=int, default=5,
                        help="Refinement iterations for first-frame register.")
    parser.add_argument("--fp-track-iter", type=int, default=2,
                        help="Refinement iterations for per-frame tracking.")
    parser.add_argument("--fp-redetect-interval", type=int, default=60,
                        help="Re-run registration every N frames (0 = never).")
    parser.add_argument("--fp-meas-noise", type=float, default=0.008,
                        help="EKF measurement stddev for FP 3D position (m).")
    # EKF tuning knobs
    parser.add_argument("--process-noise-pos", type=float, default=0.01,
                        help="EKF process-noise stddev on position (m).")
    parser.add_argument("--process-noise-vel", type=float, default=0.1,
                        help="EKF process-noise stddev on velocity (m/s).")
    parser.add_argument("--meas-noise-uv", type=float, default=6.0,
                        help="EKF measurement-noise stddev on pixel centroid "
                             "(px; only used in 'ekf' mode).")
    parser.add_argument("--meas-noise-z", type=float, default=0.10,
                        help="EKF measurement-noise stddev on monocular depth "
                             "(m; only used in 'ekf' mode).")
    # Depth calibration persistence
    parser.add_argument("--depth-scale", type=float, default=None,
                        help="Pre-calibrated DepthScaler scale factor. Lets "
                             "batch runs skip the per-trial 'c' keypress.")
    parser.add_argument("--depth-offset", type=float, default=0.0,
                        help="Pre-calibrated DepthScaler offset (default 0).")
    # PBVS tuning (optional sweeps)
    parser.add_argument("--pbvs-gain", type=float, default=0.4,
                        help="PBVS proportional gain lambda.")
    parser.add_argument("--target-depth", type=float, default=0.35,
                        help="PBVS target depth Z* in metres.")
    parser.add_argument("--pbvs-max-vel", type=float, default=0.015,
                        help="PBVS per-axis velocity clamp in m/s "
                             "(default matches the previously hardcoded "
                             "value, so omitting this flag preserves old "
                             "behavior).")
    parser.add_argument("--pbvs-dead-zone", type=float, default=0.008,
                        help="PBVS 3D position dead zone in metres "
                             "(default matches the previously hardcoded "
                             "value).")
    # Safety / auto-termination knobs (default: disabled, preserves behavior)
    parser.add_argument("--z-floor-mm", type=float, default=None,
                        help="Robot-base-frame Z floor in mm. Every PBVS "
                             "set_position command clamps its z so the "
                             "end-effector never goes below this height. "
                             "Disabled by default.")
    parser.add_argument("--auto-exit-converge-sec", type=float, default=None,
                        help="If the PBVS 3D error stays inside the "
                             "dead-zone continuously for this many "
                             "seconds, the servo loop shuts itself down "
                             "cleanly and the process exits. Disabled by "
                             "default (use 'q' key instead).")
    parser.add_argument("--auto-exit-lost-sec", type=float, default=None,
                        help="If no centroid is produced by perception "
                             "for this many consecutive seconds (object "
                             "out of view / mask lost), the servo loop "
                             "shuts itself down. Disabled by default.")
    parser.add_argument("--auto-exit-max-sec", type=float, default=None,
                        help="Wall-clock timeout from servo-enable to "
                             "auto-shutdown. Safety net so a non-converging "
                             "trial cannot run forever in a batch. "
                             "Disabled by default.")
    parser.add_argument("--stop-depth-m", type=float, default=None,
                        help="Simple depth-based stop. When the "
                             "EKF-filtered camera-frame Z (object "
                             "depth, metres) stays <= this value "
                             "continuously for --auto-exit-converge-sec "
                             "seconds, the servo loop shuts itself "
                             "down cleanly. This replaces the default "
                             "camera-frame PBVS dead-zone test when "
                             "set, which is important for IBVS (no "
                             "err_m at all) and for EKF runs where a "
                             "bad depth scale prevents err_m from ever "
                             "reaching the dead-zone. Disabled by "
                             "default.")
    args = parser.parse_args()

    # Apply CLI overrides
    DINOV2_THRESHOLD_PCT = args.threshold_pct
    DINOV2_COLOR_WEIGHT = args.color_weight
    DINOV2_REDETECT_INTERVAL = args.redetect_interval

    logger.info(f"DINOv2 threshold: {DINOV2_THRESHOLD_PCT}%")
    logger.info(f"Color weight: {DINOV2_COLOR_WEIGHT}")
    logger.info(f"Re-detect interval: {DINOV2_REDETECT_INTERVAL} frames")
    logger.info(f"PyZED: {'available' if PYZED_AVAILABLE else 'not available'}"
                f"{' (disabled)' if args.no_pyzed else ''}")

    # Build reference model (loads DINOv2, extracts features)
    ref_model = ReferenceModel(args.reference)

    # Robot
    robot = RobotController(ROBOT_IP)
    if not args.no_robot:
        robot.connect()
    else:
        logger.info("Robot disabled (--no-robot)")

    # EKF + PBVS setup
    # Camera intrinsics: approximate ZED Mini at 720p
    cam_intrinsics = CameraIntrinsics(fx=700.0, fy=700.0, cx=640.0, cy=360.0)
    ekf = PoseEKF(
        cam_intrinsics,
        process_noise_pos=args.process_noise_pos,
        process_noise_vel=args.process_noise_vel,
        meas_noise_uv=args.meas_noise_uv,
        meas_noise_z=args.meas_noise_z,
    )
    R_cam_to_robot = CAM_ROT_PRESETS[args.cam_to_robot]
    pbvs = PBVSController(
        gain=args.pbvs_gain,
        target_depth=args.target_depth,
        max_vel=args.pbvs_max_vel,
        dead_zone_m=args.pbvs_dead_zone,
        R_cam_to_robot=R_cam_to_robot,
    )
    # Depth scaler precedence:
    #   1. Explicit --depth-scale on the CLI wins.
    #   2. Otherwise try DEPTH_SCALE_STATE_PATH (written on last 'c').
    #   3. Otherwise start uncalibrated and warn.
    if args.depth_scale is not None:
        depth_scaler = DepthScaler(default_scale=args.depth_scale,
                                   default_offset=args.depth_offset)
        depth_scaler.calibrated = True
        logger.info("Depth scaler pre-calibrated from CLI: "
                    "scale=%.6f offset=%.4f",
                    args.depth_scale, args.depth_offset)
    else:
        saved = _load_depth_scale_state()
        if saved is not None:
            depth_scaler = DepthScaler(
                default_scale=saved["scale"],
                default_offset=saved.get("offset", 0.0),
            )
            depth_scaler.calibrated = True
            logger.info(
                "Depth scaler pre-calibrated from %s: scale=%.6f "
                "offset=%.4f (captured %s at Z=%.3fm, ref=%s)",
                DEPTH_SCALE_STATE_PATH,
                saved["scale"], saved.get("offset", 0.0),
                saved.get("saved_at", "?"),
                saved.get("depth_cal_m", float("nan")),
                saved.get("reference", "?"))
        else:
            depth_scaler = DepthScaler(default_scale=0.001)
            logger.warning(
                "Depth scaler is UNCALIBRATED (default_scale=0.001). Place "
                "the object at ~%.2fm and press 'c' to calibrate, pass "
                "--depth-scale <value>, or run "
                "experiments/calibrate_depth.sh to populate %s.",
                args.depth_cal_m, DEPTH_SCALE_STATE_PATH)
    logger.info(
        "EKF + PBVS initialised (cam_to_robot=%s, Q_pos=%.4f Q_vel=%.3f "
        "R_uv=%.2f R_z=%.4f gain=%.2f target_z=%.2fm)",
        args.cam_to_robot, args.process_noise_pos, args.process_noise_vel,
        args.meas_noise_uv, args.meas_noise_z, args.pbvs_gain,
        args.target_depth)

    # Start camera streamer
    stop_ev = threading.Event()
    cam_thread = DINOv2CameraStreamer(
        cam_index=args.scene_source,
        stop_event=stop_ev,
        robot=robot,
        ref_model=ref_model,
        ekf=ekf,
        pbvs=pbvs,
        depth_scaler=depth_scaler,
        use_pyzed=not args.no_pyzed,
    )
    cam_thread.depth_cal_m = args.depth_cal_m
    cam_thread.pipeline_mode = args.mode
    cam_thread.run_tag = args.run_tag
    cam_thread.reference_path = args.reference
    cam_thread._z_floor_mm = args.z_floor_mm
    cam_thread._auto_exit_converge_sec = args.auto_exit_converge_sec
    cam_thread._auto_exit_lost_sec = args.auto_exit_lost_sec
    cam_thread._auto_exit_max_sec = args.auto_exit_max_sec
    cam_thread._stop_depth_m = args.stop_depth_m
    if any(x is not None for x in (args.z_floor_mm,
                                   args.auto_exit_converge_sec,
                                   args.auto_exit_lost_sec,
                                   args.auto_exit_max_sec,
                                   args.stop_depth_m)):
        logger.info(
            "Safety/auto-exit: z_floor_mm=%s  converge_sec=%s  "
            "lost_sec=%s  max_sec=%s  stop_depth_m=%s",
            args.z_floor_mm, args.auto_exit_converge_sec,
            args.auto_exit_lost_sec, args.auto_exit_max_sec,
            args.stop_depth_m)

    if args.mode == "foundationpose":
        if args.fp_mesh is None and args.fp_box is None:
            parser.error("--mode foundationpose requires --fp-mesh or --fp-box")
        cam_thread._init_foundationpose(
            mesh_path=args.fp_mesh,
            mesh_extents_m=tuple(args.fp_box) if args.fp_box else None,
            fp_repo_dir=args.fp_repo_dir,
            weights_dir=args.fp_weights_dir,
            est_refine_iter=args.fp_est_iter,
            track_refine_iter=args.fp_track_iter,
            redetect_interval=args.fp_redetect_interval,
            meas_noise=args.fp_meas_noise,
        )

    cam_thread.start()

    try:
        cam_thread.join()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        robot.stop()
        stop_ev.set()
        cam_thread.join(timeout=2)
        logger.info("Done.")
