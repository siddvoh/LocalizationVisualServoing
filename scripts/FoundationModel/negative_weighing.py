#!/usr/bin/env python3
import time
import sys
import threading
import os
import tempfile


import cv2


import numpy as np

THIRD_PARTY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third-party")
)

PROMPT = "Yellow box"

DETECTOR = os.environ.get("DETECTOR", "gdino").lower()  # gdino | owlv2

GDINO_CONFIG  = os.path.join(THIRD_PARTY_ROOT, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GDINO_WEIGHTS = os.path.join(THIRD_PARTY_ROOT, "GroundingDINO/weights/groundingdino_swint_ogc.pth")
SAM2_CONFIG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT     = os.path.join(THIRD_PARTY_ROOT, "sam2/checkpoints/sam2.1_hiera_large.pt")
DA_CKPT_DIR   = os.path.join(THIRD_PARTY_ROOT, "Depth-Anything-V2/checkpoints")
DA_ENCODER    = "vits"

ROBOT_IP     = "192.168.1.241"
VS_GAIN      = 0.08
VS_APPROACH  = 3.0
VS_DEAD_ZONE = 15
MAX_YZ_STEP  = 12.0
MAX_JUMP_PX  = 80
VS_SPEED     = 150
VS_MVACC     = 500
VS_RATE      = 0.3
CTRL_GAIN    = 0.5

CAL_DELTA    = 8.0
CAL_WAIT     = 1.5

TRACKER_WARMUP_FRAMES = 10
TRACKER_MAX_HISTORY   = 15
NEG_POINT_COUNT       = 5
NEG_POINT_MARGIN_PX   = 30
REDETECT_INTERVAL     = 30
IOU_DRIFT_THRESH      = 0.35

MIN_RECORDING_DIM     = 480
ZED_RESOLUTION        = "HD720"

try:
    import pyzed.sl as _sl
    PYZED_AVAILABLE = True
except ImportError:
    PYZED_AVAILABLE = False

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch  = None
    DEVICE = "cpu"


def _mask_centroid(mask: np.ndarray):
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] < 1.0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def _robust_centroid(mask: np.ndarray):
    """minAreaRect centre of the closed-hull, used instead of moments for boxes"""
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _mask_centroid(mask)          # last resort

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return _mask_centroid(mask)

    # --- Step 2: minAreaRect centre (ideal for rectangles) ------------
    rect = cv2.minAreaRect(largest)
    cx, cy = rect[0]
    cx_i, cy_i = int(round(cx)), int(round(cy))

    h, w = closed.shape[:2]
    if 0 <= cy_i < h and 0 <= cx_i < w and closed[cy_i, cx_i] > 0:
        return (cx_i, cy_i)

    # --- Step 3: convex-hull moments fallback -------------------------
    hull = cv2.convexHull(largest)
    hull_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    c = _mask_centroid(hull_mask)
    if c is not None:
        return c

    return _mask_centroid(mask)


def _nms(boxes: np.ndarray, scores: np.ndarray,
         iou_thresh: float = 0.5) -> list:
    """Simple greedy NMS.  boxes: Nx4 (x1,y1,x2,y2), scores: N."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """IoU between two binary uint8 masks (0/255)."""
    if m1.shape != m2.shape:
        m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    inter = np.count_nonzero((m1 > 0) & (m2 > 0))
    union = np.count_nonzero((m1 > 0) | (m2 > 0))
    return inter / (union + 1e-6)


class ZedUndistorter:
    """Lens distortion correction for ZED stereo camera side images."""

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 w: int, h: int):
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1)

    @classmethod
    def from_frame_size(cls, w: int, h: int):
        """Build with approximate ZED Mini intrinsics for the given resolution."""
        fx = fy = 700.0 * (w / 1280.0)
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        return cls(K, dist, w, h)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)


class MaskTracker:
    """holds prev mask/logits + recent history for SAM2 propagation"""

    def __init__(self):
        self.prev_logits:  np.ndarray | None = None
        self.prev_mask:    np.ndarray | None = None
        self.prev_centroid: tuple | None     = None
        self.frame_count:  int               = 0
        self.mask_history: list              = []

        # stable snapshot used as fallback on drift
        self.anchor_locked: bool             = False
        self.anchor_logits: np.ndarray | None = None
        self.anchor_mask:   np.ndarray | None = None

    def reset(self, keep_anchor: bool = False):
        self.prev_logits   = None
        self.prev_mask     = None
        self.prev_centroid = None
        self.frame_count   = 0
        self.mask_history.clear()
        if not keep_anchor:
            self.anchor_locked = False
            self.anchor_logits = None
            self.anchor_mask   = None

    def update(self, mask_np: np.ndarray | None,
               logits: np.ndarray | None,
               centroid: tuple | None):
        self.prev_mask     = mask_np
        self.prev_logits   = logits
        self.prev_centroid = centroid
        if mask_np is not None:
            self.frame_count += 1
            self.mask_history.append(mask_np.copy())
            if len(self.mask_history) > TRACKER_MAX_HISTORY:
                self.mask_history.pop(0)
            if not self.anchor_locked and logits is not None:
                self.anchor_logits = logits.copy()
                self.anchor_mask   = mask_np.copy()
                self.anchor_locked = True
        else:
            self.frame_count = max(0, self.frame_count - 1)

    @property
    def warmed_up(self) -> bool:
        return (self.frame_count >= TRACKER_WARMUP_FRAMES
                and len(self.mask_history) >= TRACKER_WARMUP_FRAMES)

    def sample_negative_points(self, h: int, w: int,
                               n: int = NEG_POINT_COUNT) -> np.ndarray | None:
        """points never inside any recent mask -> safe negatives"""
        if not self.warmed_up:
            return None

        always_bg = np.ones((h, w), dtype=np.uint8) * 255
        for m in self.mask_history[-TRACKER_WARMUP_FRAMES:]:
            mr = m if m.shape[:2] == (h, w) else \
                cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            always_bg = cv2.bitwise_and(always_bg, cv2.bitwise_not(mr))

        # keep margin away from mask boundary
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (NEG_POINT_MARGIN_PX * 2 + 1,
                                          NEG_POINT_MARGIN_PX * 2 + 1))
        safe = cv2.erode(always_bg, kern)

        coords = np.argwhere(safe > 0)
        if len(coords) < n:
            return None

        idx = np.random.choice(len(coords), n, replace=False)
        pts = coords[idx][:, ::-1].astype(np.float32)
        return pts

    def sample_positive_points(self, h: int, w: int,
                               n: int = 1) -> np.ndarray | None:
        """centroid + random interior points from prev_mask"""
        if self.prev_mask is None:
            return None

        pts = []
        if self.prev_centroid is not None:
            pts.append(list(self.prev_centroid))

        if n > len(pts):
            mr = self.prev_mask if self.prev_mask.shape[:2] == (h, w) else \
                cv2.resize(self.prev_mask, (w, h),
                           interpolation=cv2.INTER_NEAREST)
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            interior = cv2.erode(mr, kern)
            coords = np.argwhere(interior > 0)
            if len(coords) > 0:
                extra_n = min(n - len(pts), len(coords))
                idx = np.random.choice(len(coords), extra_n, replace=False)
                for r, c in coords[idx]:
                    pts.append([c, r])

        if not pts:
            return None
        return np.array(pts[:n], dtype=np.float32)


_gdino_model  = None
_gdino_failed = False
_owlv2_processor = None
_owlv2_model     = None
_owlv2_failed    = False
_sam2_pred    = None
_depth_model  = None


def _patch_bert_compat():
    try:
        import torch as _t
        from transformers import BertModel
        from transformers.modeling_utils import PreTrainedModel

        if not hasattr(PreTrainedModel, "get_head_mask"):
            def _ghm(self, head_mask, n, chunked=False):
                if head_mask is not None:
                    hm = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    hm = hm.expand(n, -1, -1, -1, -1)
                    if chunked:
                        hm = hm.unsqueeze(-1)
                    return hm
                return [None] * n
            PreTrainedModel.get_head_mask = _ghm

        _orig_geam = getattr(PreTrainedModel, "get_extended_attention_mask")
        if not getattr(_orig_geam, "_gdino_patched", False):
            def _geam(self, attention_mask, input_shape, dev_or_dtype=None, dtype=None):
                if isinstance(dev_or_dtype, _t.device):
                    dev_or_dtype = None
                try:
                    return _orig_geam(self, attention_mask, input_shape, dev_or_dtype)
                except TypeError:
                    return _orig_geam(self, attention_mask, input_shape)
            _geam._gdino_patched = True
            PreTrainedModel.get_extended_attention_mask = _geam
            BertModel.get_extended_attention_mask = _geam

        print("BERT compat patch applied")
    except Exception as e:
        print("BERT compat patch failed:", e)
        import traceback; traceback.print_exc()


def _get_gdino():
    global _gdino_model, _gdino_failed
    if _gdino_failed:
        return None
    if _gdino_model is None:
        try:
            _patch_bert_compat()
            sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "GroundingDINO"))
            from groundingdino.util.inference import load_model
            _gdino_model = load_model(GDINO_CONFIG, GDINO_WEIGHTS, device=DEVICE)
        except Exception as e:
            print("GroundingDINO load failed (won't retry):", e)
            _gdino_failed = True
    return _gdino_model


def _get_owlv2():
    global _owlv2_processor, _owlv2_model, _owlv2_failed
    if _owlv2_failed:
        return None, None
    if _owlv2_model is None:
        try:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            _owlv2_processor = Owlv2Processor.from_pretrained(
                "google/owlv2-base-patch16-ensemble")
            _owlv2_model = Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
            print("OWLv2 loaded")
        except Exception as e:
            print("OWLv2 load failed (won't retry):", e)
            _owlv2_failed = True
            return None, None
    return _owlv2_processor, _owlv2_model


def _get_sam2():
    global _sam2_pred
    if _sam2_pred is None:
        sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "sam2"))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model      = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=DEVICE)
        _sam2_pred = SAM2ImagePredictor(model)
    return _sam2_pred


def _get_depth_model():
    global _depth_model
    if _depth_model is None:
        try:
            sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "Depth-Anything-V2"))
            from depth_anything_v2.dpt import DepthAnythingV2
            cfgs = {
                "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
                "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536,1536,1536, 1536]},
            }
            ckpt = os.path.join(DA_CKPT_DIR, f"depth_anything_v2_{DA_ENCODER}.pth")
            if not os.path.exists(ckpt):
                print(f"Depth-Anything-V2 checkpoint not found: {ckpt}")
                return None
            import torch as _t
            m = DepthAnythingV2(**cfgs[DA_ENCODER])
            m.load_state_dict(_t.load(ckpt, map_location="cpu"))
            _depth_model = m.to(DEVICE).eval()
        except Exception as e:
            print("Depth-Anything-V2 load failed:", e)
    return _depth_model


def _detect_gdino(image_bgr: np.ndarray, prompt: str):
    """Return list of (xyxy_np, score, phrase) or empty list."""
    gdino = _get_gdino()
    if gdino is None:
        return []
    try:
        sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "GroundingDINO"))
        from groundingdino.util.inference import load_image, predict as gd_predict
        from torchvision.ops import box_convert
        import torch as _t

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, image_bgr)
        tmp.close()
        try:
            _, img_t = load_image(tmp.name)
            boxes, logits, phrases = gd_predict(
                gdino, img_t, caption=prompt,
                box_threshold=0.3, text_threshold=0.25, device=DEVICE)
            if boxes is None or len(boxes) == 0:
                print("GDINO: no boxes")
                return []
            h, w = image_bgr.shape[:2]
            bxyxy = (box_convert(boxes, "cxcywh", "xyxy")
                     * _t.tensor([w, h, w, h], dtype=_t.float32)).numpy()
            dets = []
            for i in range(len(boxes)):
                dets.append((bxyxy[i], float(logits[i]), phrases[i]))
            print(f"GDINO: {len(dets)} detection(s)")
            return dets
        finally:
            os.unlink(tmp.name)
    except Exception as e:
        import traceback
        print("GroundingDINO failed:", e)
        traceback.print_exc()
        return []


def _detect_owlv2(image_bgr: np.ndarray, prompt: str):
    """Return list of (xyxy_np, score, phrase) or empty list."""
    processor, model = _get_owlv2()
    if model is None:
        return []
    try:
        import torch as _t
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        texts = [[prompt]]
        inputs = processor(text=texts, images=pil, return_tensors="pt").to(DEVICE)

        with _t.no_grad():
            outputs = model(**inputs)

        h, w = image_bgr.shape[:2]
        target_sizes = _t.tensor([[h, w]], device=DEVICE)

        if hasattr(processor, "post_process_grounded_object_detection"):
            results = processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes,
                threshold=0.1, text_labels=texts)[0]
        else:
            results = processor.post_process_object_detection(
                outputs, threshold=0.1, target_sizes=target_sizes)[0]

        boxes  = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        if len(boxes) == 0:
            print("OWLv2: no boxes")
            return []

        dets = []
        for i in range(len(boxes)):
            dets.append((boxes[i], float(scores[i]), prompt))
        print(f"OWLv2: {len(dets)} detection(s)")
        return dets

    except Exception as e:
        import traceback
        print("OWLv2 failed:", e)
        traceback.print_exc()
        return []


def _disambiguate_top_box(dets: list, image_shape: tuple,
                          prev_centroid: tuple | None = None):
    """pick top box on stack: NMS -> area filter -> score by pos/conf/prox"""
    if not dets:
        return None

    h, w = image_shape[:2]
    img_area = h * w

    boxes  = np.array([d[0] for d in dets], dtype=np.float32)
    scores = np.array([d[1] for d in dets], dtype=np.float32)
    keep   = _nms(boxes, scores, iou_thresh=0.45)
    dets   = [dets[i] for i in keep]

    if not dets:
        return None

    filtered = []
    for det in dets:
        bx = det[0]
        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        area_ratio = (bw * bh) / img_area
        if area_ratio < 0.003 or area_ratio > 0.75:
            continue
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 8.0:
            continue
        filtered.append(det)

    if not filtered:
        return min(dets, key=lambda d: d[0][1])

    best, best_score = None, -1e9
    for det in filtered:
        bx, conf, _ = det
        y_centre = (bx[1] + bx[3]) / 2.0

        pos_score = 1.0 - (y_centre / h)

        prox_score = 0.0
        if prev_centroid is not None:
            cx = (bx[0] + bx[2]) / 2.0
            cy = (bx[1] + bx[3]) / 2.0
            dist = np.hypot(cx - prev_centroid[0], cy - prev_centroid[1])
            prox_score = max(0.0, 1.0 - dist / (0.25 * max(h, w)))

        combined = pos_score * 0.50 + conf * 0.25 + prox_score * 0.25
        if combined > best_score:
            best_score = combined
            best = det

    return best


def run_pipeline(image_bgr: np.ndarray, prompt: str,
                 tracker: MaskTracker | None = None) -> dict:
    """detect -> SAM2 (+mask propagation) -> depth -> grasp point"""
    res = dict(mask_np=None, depth_np=None,
               best_centroid=None, gdino_box=None)

    h, w = image_bgr.shape[:2]

    has_track = tracker is not None and tracker.prev_logits is not None
    periodic_redetect = has_track and tracker.frame_count % REDETECT_INTERVAL == 0
    need_detection = not has_track or periodic_redetect

    box_np = None
    if need_detection:
        if DETECTOR == "owlv2":
            dets = _detect_owlv2(image_bgr, prompt)
        else:
            dets = _detect_gdino(image_bgr, prompt)

        prev_c = tracker.prev_centroid if tracker else None
        pick = _disambiguate_top_box(dets, image_bgr.shape,
                                     prev_centroid=prev_c)
        if pick is not None:
            box_np = pick[0]
            res["gdino_box"] = box_np
            print(f"{DETECTOR.upper()}: '{pick[2]}' score={pick[1]:.2f} "
                  f"@ {box_np.astype(int)}  (top of {len(dets)})")

    tracker_logits = None
    try:
        pred = _get_sam2()
        rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pred.set_image(rgb)

        point_coords = []
        point_labels = []

        if tracker is not None:
            pos_pts = tracker.sample_positive_points(h, w, n=1)
            if pos_pts is not None:
                point_coords.extend(pos_pts)
                point_labels.extend([1] * len(pos_pts))

            neg_pts = tracker.sample_negative_points(h, w, NEG_POINT_COUNT)
            if neg_pts is not None:
                point_coords.extend(neg_pts)
                point_labels.extend([0] * len(neg_pts))
                print(f"SAM2: +{len(pos_pts) if pos_pts is not None else 0} pos, "
                      f"+{len(neg_pts)} neg point prompts")

        has_prior = tracker is not None and tracker.prev_logits is not None
        sam_kwargs = dict(
            multimask_output=not has_prior,
            return_logits=True,
        )

        if box_np is not None:
            sam_kwargs["box"] = box_np

        if point_coords:
            sam_kwargs["point_coords"] = np.array(point_coords, dtype=np.float32)
            sam_kwargs["point_labels"] = np.array(point_labels, dtype=np.int32)

        if has_prior:
            sam_kwargs["mask_input"] = tracker.prev_logits

        masks, scores_sam, logits = pred.predict(**sam_kwargs)

        if masks is not None and len(masks) > 0:
            best_idx = int(np.argmax(scores_sam))
            mask_out = (masks[best_idx] > 0).astype(np.uint8) * 255

            if tracker is not None and tracker.prev_mask is not None:
                iou = _mask_iou(mask_out, tracker.prev_mask)
                if iou < IOU_DRIFT_THRESH and not need_detection:
                    print(f"SAM2: mask IoU={iou:.2f} < {IOU_DRIFT_THRESH}, "
                          f"probable drift, resetting tracker")
                    tracker.reset(keep_anchor=True)
                    return run_pipeline(image_bgr, prompt, tracker)

            res["mask_np"] = mask_out
            tracker_logits = logits[best_idx:best_idx + 1]

            print(f"SAM2: mask obtained (score={scores_sam[best_idx]:.3f})")
        else:
            print("SAM2: no mask returned")
    except Exception as e:
        import traceback
        print("SAM2 failed:", e)
        traceback.print_exc()

    dm = _get_depth_model()
    if dm is not None:
        try:
            res["depth_np"] = dm.infer_image(image_bgr).astype(np.float32)
        except Exception as e:
            print("Depth inference failed:", e)

    centroid = None
    if res["mask_np"] is not None:
        centroid = _robust_centroid(res["mask_np"])
        if centroid is not None:
            print(f"Grasp point: robust centroid {centroid}")

    if centroid is None and res["gdino_box"] is not None:
        x1, y1, x2, y2 = res["gdino_box"]
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        reason = "centroid failed" if res["mask_np"] is not None else "no SAM2 mask"
        print(f"Grasp point: bbox centre {centroid} ({reason})")

    res["best_centroid"] = centroid

    if tracker is not None:
        tracker.update(res["mask_np"], tracker_logits,
                       res.get("best_centroid"))

    return res


def _measure_flow(frame_a: np.ndarray, frame_b: np.ndarray):
    ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(ga, maxCorners=300, qualityLevel=0.01,
                                  minDistance=8, blockSize=7)
    if pts is None or len(pts) < 10:
        return None
    pts_next, status, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None)
    ok = status.ravel() == 1
    if ok.sum() < 10:
        return None
    flow = pts_next[ok] - pts[ok]
    dx = float(np.median(flow[:, 0, 0]))
    dy = float(np.median(flow[:, 0, 1]))
    return dx, dy


class RobotController:
    def __init__(self, ip: str):
        self.ip               = ip
        self._arm             = None
        self._lock            = threading.Lock()
        self.enabled          = False
        self._last_t          = 0.0
        self._jac_yz          = None
        self._jac_yz_inv      = None
        self.cal_status       = "uncalibrated"
        self._last_centroid   = None
        self._last_centroid_t = 0.0
        # (dx_mm, dy_mm, dz_mm, err_px) for csv logging
        self._last_ibvs_cmd = (0.0, 0.0, 0.0, 0.0)

    def _get_pos(self):
        ret = self._arm.get_position()
        if isinstance(ret, (list, tuple)) and len(ret) >= 2 and ret[0] == 0:
            return list(ret[1])
        print("get_position failed:", ret)
        return None

    def _move_abs(self, pos, wait=True):
        self._arm.set_position(
            x=pos[0], y=pos[1], z=pos[2],
            roll=pos[3], pitch=pos[4], yaw=pos[5],
            speed=VS_SPEED, mvacc=VS_MVACC, wait=wait)

    def connect(self) -> bool:
        try:
            from xarm.wrapper import XArmAPI
            arm = XArmAPI(self.ip, baud_checkset=False)
            time.sleep(0.5)
            if not arm.connected:
                print(f"Robot not connected: {self.ip}")
                return False
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.5)
            self._arm = arm
            self.enabled = False
            self.cal_status = "waiting for calibration..."
            print(f"Robot connected: {self.ip}")
            return True
        except Exception as e:
            print(f"Robot connect failed: {e}")
            return False

    def calibrate(self, get_frame_fn):
        self.cal_status = "waiting for frame..."
        print("\n=== Calibration: waiting for camera frame ===")

        deadline = time.time() + 30.0
        while time.time() < deadline:
            if get_frame_fn() is not None:
                break
            time.sleep(0.2)
        else:
            print("Calibration skipped: no camera frame available.")
            self.cal_status = "skipped (no frame)"
            self.enabled = True
            return

        pos0 = self._get_pos()
        if pos0 is None:
            print("Calibration skipped: cannot read robot position.")
            self.cal_status = "skipped (pos read fail)"
            self.enabled = True
            return

        print(f"Home: [{pos0[0]:.1f}, {pos0[1]:.1f}, {pos0[2]:.1f}] mm")

        J_yz = np.zeros((2, 2), dtype=np.float64)

        for col, (robot_idx, ax) in enumerate([(1, "Y"), (2, "Z")]):
            self.cal_status = f"calibrating {ax}..."

            time.sleep(0.3)
            frame_before = get_frame_fn()
            if frame_before is None:
                print(f"  [{ax}] no frame before move — skipping axis")
                continue

            print(f"  [{ax}] moving +{CAL_DELTA:.0f} mm...")
            fwd = list(pos0)
            fwd[robot_idx] += CAL_DELTA
            self._move_abs(fwd, wait=True)
            time.sleep(CAL_WAIT)

            frame_after = get_frame_fn()

            print(f"  [{ax}] returning home...")
            self._move_abs(pos0, wait=True)

            if frame_after is None:
                print(f"  [{ax}] no frame after move — skipping axis")
                continue

            flow = _measure_flow(frame_before, frame_after)
            if flow is None:
                print(f"  [{ax}] optical flow failed (too few trackable points) — skipping axis")
                continue

            dpx, dpy = flow
            J_yz[0, col] = dpx / CAL_DELTA
            J_yz[1, col] = dpy / CAL_DELTA
            print(f"  [{ax}] optical flow: ({dpx:+.1f}, {dpy:+.1f}) px / {CAL_DELTA:.0f} mm"
                  f"  → J_yz[:,{col}]=[{J_yz[0,col]:+.4f}, {J_yz[1,col]:+.4f}]")

        self._jac_yz = J_yz
        print(f"\n  J_yz (px/mm):\n{J_yz}")

        rank = np.linalg.matrix_rank(J_yz)
        if rank == 0:
            print("  WARNING: J_yz rank=0 — no usable calibration. Approach-only mode.")
            self.cal_status = "skipped (no flow data)"
            self.enabled = True
            return

        self._jac_yz_inv = np.linalg.pinv(J_yz)
        print(f"  J_yz_inv (mm/px):\n{self._jac_yz_inv}")
        status = "calibrated" if rank == 2 else f"partial cal (rank={rank})"
        print(f"=== Calibration complete [{status}] — servo enabled ===\n")
        self.cal_status = status
        self.enabled = True

    def stop(self):
        self.enabled = False
        if self._arm is not None:
            try:
                self._arm.emergency_stop()
                print("Robot stopped.")
            except Exception as e:
                print("Robot stop failed:", e)

    def servo_step(self, centroid: tuple, image_shape: tuple):
        now = time.time()
        if self._arm is None:
            if now - self._last_t > 5.0:
                print("Servo: arm not connected")
                self._last_t = now
            return
        if not self.enabled:
            if now - self._last_t > 5.0:
                print(f"Servo: waiting [{self.cal_status}]")
                self._last_t = now
            return
        if now - self._last_t < VS_RATE:
            return

        if now - self._last_centroid_t > 1.0:
            self._last_centroid = None
        if self._last_centroid is not None:
            jump = np.hypot(centroid[0] - self._last_centroid[0],
                            centroid[1] - self._last_centroid[1])
            if jump > MAX_JUMP_PX:
                print(f"Servo: centroid jump {jump:.0f} px — ignored (likely mis-detection)")
                return
        self._last_centroid   = centroid
        self._last_centroid_t = now

        self._last_t = now

        h, w       = image_shape[:2]
        ic_x, ic_y = w // 2, h // 2
        ex    = centroid[0] - ic_x
        ey    = centroid[1] - ic_y
        err_r = float(np.hypot(ex, ey))

        dx_mm = VS_APPROACH

        if err_r <= VS_DEAD_ZONE:
            dy_mm, dz_mm = 0.0, 0.0
        elif self._jac_yz_inv is not None:
            err  = np.array([ex, ey], dtype=np.float64)
            yz   = -CTRL_GAIN * (self._jac_yz_inv @ err)
            dy_mm = float(np.clip(yz[0], -MAX_YZ_STEP, MAX_YZ_STEP))
            dz_mm = float(np.clip(yz[1], -MAX_YZ_STEP, MAX_YZ_STEP))
        else:
            dy_mm = 0.0
            dz_mm = 0.0

        # Publish the command so the per-frame CSV logger can pick it
        # up. Done BEFORE the set_position call so the log reflects what
        # the controller *tried* to send even if set_position fails.
        self._last_ibvs_cmd = (dx_mm, dy_mm, dz_mm, err_r)

        with self._lock:
            try:
                pos = self._get_pos()
                if pos is None:
                    return
                self._arm.set_position(
                    x=pos[0] + dx_mm, y=pos[1] + dy_mm, z=pos[2] + dz_mm,
                    roll=pos[3], pitch=pos[4], yaw=pos[5],
                    speed=VS_SPEED, mvacc=VS_MVACC, wait=True)
                print(f"Servo: err=({ex:+.0f},{ey:+.0f})px r={err_r:.0f}  "
                      f"Δ=({dx_mm:+.1f},{dy_mm:+.1f},{dz_mm:+.1f})mm"
                      f"  [{self.cal_status}]")
            except Exception as e:
                print("Servo step failed:", e)


class CameraStreamer(threading.Thread):
    def __init__(self, cam_index: int, stop_event: threading.Event,
                 robot: RobotController):
        super().__init__(daemon=True)
        self.cam_index  = cam_index
        self.stop_event = stop_event
        self.robot      = robot

        self._latest_left  = None
        self._frame_lock   = threading.Lock()
        self.data_lock     = threading.Lock()
        self._result: dict = {}
        self._models_ready = threading.Event()
        self._tracker      = MaskTracker()          # ← NEW

    def _get_centroid(self):
        with self.data_lock:
            return self._result.get("best_centroid")

    def _get_frame(self):
        with self._frame_lock:
            if self._latest_left is not None:
                return self._latest_left.copy()
            return None

    def _run_calibration(self):
        print("Calibration thread: waiting for models to finish loading...")
        self._models_ready.wait()
        print("Calibration thread: models ready, starting Y/Z Jacobian calibration.")
        self.robot.calibrate(self._get_frame)

    def _seg_loop(self):
        while not self.stop_event.is_set():
            with self._frame_lock:
                frame = self._latest_left.copy() if self._latest_left is not None else None
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                res = run_pipeline(frame, PROMPT, self._tracker)
                with self.data_lock:
                    self._result = res
                if not self._models_ready.is_set():
                    print("Models ready — calibration may now proceed.")
                    self._models_ready.set()
                if self.robot is not None and res.get("best_centroid") is not None:
                    self.robot.servo_step(res["best_centroid"], frame.shape)
            except Exception as e:
                print("Pipeline error:", e)
                import traceback; traceback.print_exc()
                time.sleep(1)

    def run(self):
        if cv2 is None:
            print("OpenCV not available.")
            return
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            print(f"Failed to open camera {self.cam_index}")
            return

        threading.Thread(target=self._seg_loop, daemon=True).start()
        if self.robot is not None and self.robot._arm is not None:
            threading.Thread(target=self._run_calibration, daemon=True).start()

        win = "Grasp point  |  [v] servo  [r] reset tracker  [q] quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        video_writer = None
        video_path   = None

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                h, w = frame.shape[:2]
                left = frame[:, : w // 2].copy()

                with self._frame_lock:
                    self._latest_left = left.copy()
                with self.data_lock:
                    res = dict(self._result)

                rendered = self._render(left, res)

                if video_writer is None:
                    rh, rw = rendered.shape[:2]
                    video_path = time.strftime("vs_recording_%Y%m%d_%H%M%S.mp4")
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, (rw, rh))
                    print(f"Recording to {video_path}")

                video_writer.write(rendered)
                cv2.imshow(win, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
                    break
                elif key == ord("v") and self.robot is not None:
                    self.robot.enabled = not self.robot.enabled
                    print("Servo:", "ON" if self.robot.enabled else "OFF")
                elif key == ord("r"):
                    self._tracker.reset()
                    print("Tracker reset (will re-detect next frame)")
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                print(f"Recording saved: {video_path}")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _render(self, left: np.ndarray, res: dict) -> np.ndarray:
        display = left.copy()
        h, w    = display.shape[:2]

        mask_np       = res.get("mask_np")
        depth_np      = res.get("depth_np")
        best_centroid = res.get("best_centroid")
        gdino_box     = res.get("gdino_box")

        # ── Resize mask to display resolution if needed ──────────────
        mask_np_r = mask_np
        if mask_np_r is not None and mask_np_r.shape[:2] != (h, w):
            mask_np_r = cv2.resize(mask_np_r, (w, h),
                                   interpolation=cv2.INTER_NEAREST)

        # ── SAM2 mask overlay (green tint) ───────────────────────────
        if mask_np_r is not None:
            overlay          = np.zeros_like(display)
            overlay[:, :, 1] = mask_np_r
            display = cv2.addWeighted(display, 0.72, overlay, 0.28, 0)

            cnts, _ = cv2.findContours(mask_np_r, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, cnts, -1, (0, 255, 0), 2)

        # ── Bounding box ─────────────────────────────────────────────
        if gdino_box is not None:
            x1, y1, x2, y2 = gdino_box.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)

        # ── Image centre (servo reference) ───────────────────────────
        ic = (w // 2, h // 2)
        cv2.drawMarker(display, ic, (255, 80, 0), cv2.MARKER_CROSS, 22, 2)

        # ── Grasp point (mask centroid or bbox fallback) ─────────────
        if best_centroid is not None:
            gp = best_centroid
            cv2.circle(display, gp, 18, (0, 0, 255), 2)
            cv2.drawMarker(display, gp, (0, 0, 255),
                           cv2.MARKER_CROSS, 26, 2)
            cv2.putText(display, "GRASP", (gp[0] + 22, gp[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

            cv2.arrowedLine(display, ic, gp, (0, 220, 255), 2, tipLength=0.12)
            ex = gp[0] - ic[0]
            ey = gp[1] - ic[1]
            cv2.putText(display, f"err ({ex:+d},{ey:+d}) px",
                        (8, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 220, 255), 1)

        # ── Status bar ───────────────────────────────────────────────
        # Tracker info
        trk = self._tracker
        trk_txt = (f"trk: {trk.frame_count} frm"
                   f"{'  [warmed up]' if trk.warmed_up else ''}")
        cv2.putText(display, trk_txt, (8, h - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 180, 0), 1)

        if self.robot is not None:
            cal = self.robot.cal_status
            if not self.robot.enabled:
                txt = f"SERVO OFF  [{cal}]"
                col = (80, 80, 80)
            else:
                txt = f"SERVO ON  [{cal}]"
                col = (0, 255, 80)
            cv2.putText(display, txt, (8, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

        # ── Detector label ───────────────────────────────────────────
        cv2.putText(display, DETECTOR.upper(), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 0), 1)

        # ── Depth inset (top-right) ──────────────────────────────────
        if depth_np is not None:
            try:
                d = depth_np.astype(np.float32)
                d_mn, d_mx = float(np.nanmin(d)), float(np.nanmax(d))
                d_u8  = ((d - d_mn) / (d_mx - d_mn + 1e-6) * 255).astype(np.uint8)
                cmap  = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
                iw    = max(100, w // 4)
                ih    = int(iw * h / w)
                inset = cv2.resize(cmap, (iw, ih))
                display[10:10 + ih, w - iw - 10: w - 10] = inset
                cv2.putText(display, "depth", (w - iw - 8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            except Exception:
                pass

        return display


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SemVS Pipeline")
    parser.add_argument("cam_index", nargs="?", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--detector", choices=["gdino", "owlv2"],
                        default=None,
                        help="Override detector (default: env DETECTOR or gdino)")
    args = parser.parse_args()

    if args.detector:
        DETECTOR = args.detector
    print(f"Using detector: {DETECTOR.upper()}")

    robot = RobotController(ROBOT_IP)
    robot.connect()

    stop_ev    = threading.Event()
    cam_thread = CameraStreamer(args.cam_index, stop_ev, robot)
    cam_thread.start()

    try:
        cam_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        robot.stop()
        stop_ev.set()
        cam_thread.join(timeout=2)
        print("Done.")