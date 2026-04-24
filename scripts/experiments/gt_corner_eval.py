#!/usr/bin/env python3
# per-frame GT-vs-centroid evaluation for saved vs_dinov2 videos.
# method A = extract SAM2 green overlay; B = Canny+Hough fallback.
import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR  = REPO_ROOT / "runs"

VIDEO_RE = re.compile(
    r"vs_dinov2_([a-z0-9_]+?)_(ekf|pbvs|fp)_(\d{8})_(\d{6})\.mp4$"
)

# method A: dinov2_servo.py blends SAM2 mask as G_out = 0.72*G + 0.28*255
GREEN_EXCESS_THRESH = 12
GREEN_MORPH_K       = 9
MASK_MIN_AREA       = 2500
MASK_MAX_AREA_FRAC  = 0.25

# method B (edge/Hough)
EDGE_ROI_RADIUS     = 280
EDGE_CANNY_LO       = 40
EDGE_CANNY_HI       = 120
EDGE_HOUGH_THRESH   = 25
EDGE_MIN_LINE_FRAC  = 0.12

# box geometry sanity
BOX_MIN_DIM      = 40
BOX_MAX_DIM_FRAC = 0.80
BOX_MAX_ASPECT   = 5.0


def _fname_ts_to_unix(date_str: str, time_str: str) -> float:
    dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    return dt.timestamp()


def find_matching_csv(object_name: str, controller: str,
                      date_str: str, video_unix: float) -> Path | None:
    """match video to CSV; fallback tries _2 suffix on/off"""
    def _search(name: str) -> list:
        found = []
        for d in RUNS_DIR.iterdir():
            if not d.is_dir():
                continue
            m = re.match(
                rf"^{re.escape(name)}_(\d{{8}})_(\d{{6}})$", d.name
            )
            if not m or m.group(1) != date_str:
                continue
            csv_path = d / f"{name}_{controller}.csv"
            if not csv_path.exists():
                continue
            dir_unix = _fname_ts_to_unix(m.group(1), m.group(2))
            found.append((abs(dir_unix - video_unix), csv_path))
        return found

    candidates = _search(object_name)

    # Fallback: try the sister name (strip or append '_2')
    if not candidates:
        if object_name.endswith("_2"):
            alt = object_name[:-2]
        else:
            alt = object_name + "_2"
        candidates = _search(alt)

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def build_timestamp_lookup(csv_path: Path) -> list:
    """Return sorted list of (unix_ts, centroid_u|None, centroid_v|None)."""
    result = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            ts_s = row.get("timestamp", "")
            if not ts_s:
                continue
            try:
                ts = float(ts_s)
            except ValueError:
                continue
            u_s = row.get("centroid_u", "")
            v_s = row.get("centroid_v", "")
            u = int(u_s) if u_s else None
            v = int(v_s) if v_s else None
            result.append((ts, u, v))
    result.sort(key=lambda x: x[0])
    return result


def nearest_centroid(lookup: list, query_ts: float):
    if not lookup:
        return None, None
    lo, hi = 0, len(lookup) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if lookup[mid][0] < query_ts:
            lo = mid + 1
        else:
            hi = mid
    best = lo
    if lo > 0 and (abs(lookup[lo-1][0] - query_ts)
                   < abs(lookup[lo][0] - query_ts)):
        best = lo - 1
    return lookup[best][1], lookup[best][2]


def _cluster_split(vals: list) -> tuple:
    if len(vals) < 2:
        return vals, []
    gaps = [(vals[i+1] - vals[i], i) for i in range(len(vals)-1)]
    _, idx = max(gaps)
    return vals[:idx+1], vals[idx+1:]


def _rect_ok(x_lft, y_top, x_rgt, y_bot, fw, fh) -> bool:
    w, h = x_rgt - x_lft, y_bot - y_top
    if w < BOX_MIN_DIM or h < BOX_MIN_DIM:
        return False
    if w > fw * BOX_MAX_DIM_FRAC or h > fh * BOX_MAX_DIM_FRAC:
        return False
    return max(w, h) / min(w, h) <= BOX_MAX_ASPECT


def _corners(x_lft, y_top, x_rgt, y_bot) -> np.ndarray:
    return np.array([
        [x_lft, y_top],
        [x_rgt, y_top],
        [x_rgt, y_bot],
        [x_lft, y_bot],
    ], dtype=np.float32)


def _detect_from_green_mask(frame_bgr: np.ndarray,
                             hint_point: tuple | None) -> np.ndarray | None:
    """find the SAM2 green overlay, return [TL,TR,BR,BL]"""
    fh, fw = frame_bgr.shape[:2]
    b = frame_bgr[:, :, 0].astype(np.int16)
    g = frame_bgr[:, :, 1].astype(np.int16)
    r = frame_bgr[:, :, 2].astype(np.int16)

    raw = (((g - r) > GREEN_EXCESS_THRESH) &
           ((g - b) > GREEN_EXCESS_THRESH)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (GREEN_MORPH_K, GREEN_MORPH_K))
    clean = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    max_area = MASK_MAX_AREA_FRAC * fw * fh
    valid = [c for c in contours if MASK_MIN_AREA <= cv2.contourArea(c) <= max_area]
    if not valid:
        return None

    def _ctr(c):
        M = cv2.moments(c)
        return (M["m10"]/M["m00"], M["m01"]/M["m00"]) if M["m00"] else (fw/2, fh/2)

    if hint_point is not None:
        contour = min(valid, key=lambda c: math.hypot(
            _ctr(c)[0] - hint_point[0], _ctr(c)[1] - hint_point[1]))
    else:
        contour = max(valid, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)
    if not _rect_ok(x, y, x+w, y+h, fw, fh):
        return None
    return _corners(x, y, x+w, y+h)


def _detect_from_edges(frame_bgr: np.ndarray,
                        hint_point: tuple) -> np.ndarray | None:
    """canny+hough inside ROI around hint_point"""
    fh, fw = frame_bgr.shape[:2]
    hx, hy = int(hint_point[0]), int(hint_point[1])

    x1 = max(0,  hx - EDGE_ROI_RADIUS)
    x2 = min(fw, hx + EDGE_ROI_RADIUS)
    y1 = max(0,  hy - EDGE_ROI_RADIUS)
    y2 = min(fh, hy + EDGE_ROI_RADIUS)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_w = x2 - x1

    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, EDGE_CANNY_LO, EDGE_CANNY_HI)

    min_len = max(20, int(roi_w * EDGE_MIN_LINE_FRAC))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=EDGE_HOUGH_THRESH,
        minLineLength=min_len,
        maxLineGap=8,
    )
    if lines is None:
        return None

    h_mids, v_mids = [], []
    for ln in lines:
        lx1, ly1, lx2, ly2 = ln[0]
        dx, dy = abs(lx2 - lx1), abs(ly2 - ly1)
        if dx == 0 and dy == 0:
            continue
        if dy < 0.25 * dx:
            h_mids.append((ly1 + ly2) / 2.0)
        elif dx < 0.25 * dy:
            v_mids.append((lx1 + lx2) / 2.0)

    if len(h_mids) < 2 or len(v_mids) < 2:
        return None

    h_a, h_b = _cluster_split(sorted(h_mids))
    v_a, v_b = _cluster_split(sorted(v_mids))
    if not h_a or not h_b or not v_a or not v_b:
        return None

    y_top_roi = float(np.median(h_a))
    y_bot_roi = float(np.median(h_b))
    x_lft_roi = float(np.median(v_a))
    x_rgt_roi = float(np.median(v_b))

    y_top = min(y_top_roi, y_bot_roi) + y1
    y_bot = max(y_top_roi, y_bot_roi) + y1
    x_lft = min(x_lft_roi, x_rgt_roi) + x1
    x_rgt = max(x_lft_roi, x_rgt_roi) + x1

    if not _rect_ok(x_lft, y_top, x_rgt, y_bot, fw, fh):
        return None
    return _corners(x_lft, y_top, x_rgt, y_bot)


def detect_box_corners(frame_bgr: np.ndarray,
                        hint_point: tuple | None = None,
                        edge_only: bool = False,
                        ) -> tuple[np.ndarray | None, str]:
    """returns (corners, method) where method is 'mask' | 'edge' | 'none'"""
    if not edge_only:
        corners = _detect_from_green_mask(frame_bgr, hint_point)
        if corners is not None:
            return corners, "mask"

    if hint_point is not None:
        corners = _detect_from_edges(frame_bgr, hint_point)
        if corners is not None:
            return corners, "edge"

    return None, "none"


# ── Drawing ----------------------------------------

def draw_overlays(frame: np.ndarray,
                  corners: np.ndarray | None,
                  gt_cx: float | None, gt_cy: float | None,
                  centroid_u: int | None, centroid_v: int | None,
                  err_px: float | None,
                  gt_diag: float | None,
                  method: str,
                  use_cam_center: bool = False) -> np.ndarray:
    out = frame.copy()

    # GT corners — yellow filled circles
    if corners is not None:
        for (cx, cy) in corners:
            cv2.circle(out, (int(cx), int(cy)), 6, (0, 255, 255), -1)

    # GT center — bright green crosshair
    if gt_cx is not None and gt_cy is not None:
        gx, gy = int(gt_cx), int(gt_cy)
        arm = 15
        cv2.line(out, (gx - arm, gy), (gx + arm, gy), (0, 255, 0), 2)
        cv2.line(out, (gx, gy - arm), (gx, gy + arm), (0, 255, 0), 2)

    if centroid_u is not None and centroid_v is not None:
        if use_cam_center:
            arm = 12
            cv2.line(out, (centroid_u - arm, centroid_v),
                     (centroid_u + arm, centroid_v), (255, 100, 0), 2)
            cv2.line(out, (centroid_u, centroid_v - arm),
                     (centroid_u, centroid_v + arm), (255, 100, 0), 2)
            cv2.circle(out, (centroid_u, centroid_v), 8, (255, 100, 0), 2)
        else:
            cv2.circle(out, (centroid_u, centroid_v), 5, (0, 0, 220), -1)

    if (gt_cx is not None and centroid_u is not None
            and centroid_v is not None):
        cv2.line(out,
                 (int(gt_cx), int(gt_cy)),
                 (centroid_u, centroid_v),
                 (255, 255, 255), 1)

    ref_label = "cam_ctr" if use_cam_center else "tracked"
    if err_px is not None and gt_diag is not None:
        norm = err_px / gt_diag
        label = f"gt_err: {err_px:.1f}px ({norm:.3f} norm) [{method}|{ref_label}]"
        cv2.putText(out, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (255, 255, 255), 2, cv2.LINE_AA)
    elif err_px is not None:
        cv2.putText(out, f"gt_err: {err_px:.1f}px [{method}|{ref_label}]",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "gt: not detected", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (100, 100, 255), 2, cv2.LINE_AA)
    return out


def process_video(video_path: Path, out_dir: Path, verbose: bool = True,
                  edge_only: bool = False) -> dict:
    m = VIDEO_RE.match(video_path.name)
    if not m:
        return {}

    object_name = m.group(1)
    controller  = m.group(2)
    date_str    = m.group(3)
    time_str    = m.group(4)
    video_unix  = _fname_ts_to_unix(date_str, time_str)
    stem        = video_path.stem

    if verbose:
        print(f"\n{'='*62}")
        print(f"Video : {video_path.name}")
        print(f"Object: {object_name}   Controller: {controller}")

    csv_path    = find_matching_csv(object_name, controller, date_str, video_unix)
    csv_matched = csv_path is not None
    if verbose:
        print(f"CSV   : {csv_path or 'NOT FOUND'}")

    lookup = build_timestamp_lookup(csv_path) if csv_path else []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return {}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # error is measured against camera centre; csv centroid used only as hint
    cam_cx = width  // 2
    cam_cy = height // 2
    if verbose:
        print(f"Reference: camera centre ({cam_cx}, {cam_cy})")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / f"{stem}_gt.mp4"
    out_csv_path   = out_dir / f"{stem}_gt.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    per_frame_rows = []
    err_list   = []   # (err_px, err_norm, gt_diag)
    diag_list  = []
    method_counts = {"mask": 0, "edge": 0, "none": 0}

    frame_idx = 0
    last_gt_center: tuple | None = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_time_s = frame_idx / fps
        query_ts     = video_unix + video_time_s

        cu_csv, cv_csv = (nearest_centroid(lookup, query_ts)
                          if lookup else (None, None))
        hint = (float(cu_csv), float(cv_csv)) if cu_csv is not None else None

        cu, cv_ = cam_cx, cam_cy

        if hint is None:
            if last_gt_center is not None:
                hint = last_gt_center
            else:
                hint = (float(cam_cx), float(cam_cy))

        corners, method = detect_box_corners(frame, hint_point=hint,
                                              edge_only=edge_only)
        method_counts[method] += 1

        gt_cx = gt_cy = gt_diag = None
        if corners is not None:
            gt_cx   = float(np.mean(corners[:, 0]))
            gt_cy   = float(np.mean(corners[:, 1]))
            gt_diag = float(np.hypot(corners[2, 0] - corners[0, 0],
                                     corners[2, 1] - corners[0, 1]))
            diag_list.append(gt_diag)
            last_gt_center = (gt_cx, gt_cy)

        err_px = err_norm = None
        if gt_cx is not None:
            err_px   = math.hypot(cu - gt_cx, cv_ - gt_cy)
            err_norm = err_px / gt_diag if gt_diag else None
            err_list.append((err_px, err_norm, gt_diag))

        annotated = draw_overlays(
            frame, corners, gt_cx, gt_cy, cu, cv_,
            err_px, gt_diag, method, use_cam_center=True
        )
        writer.write(annotated)

        per_frame_rows.append({
            "frame_idx":      frame_idx,
            "video_time_s":   f"{video_time_s:.4f}",
            "cam_cx":         cu,
            "cam_cy":         cv_,
            "csv_centroid_u": cu_csv if cu_csv is not None else "",
            "csv_centroid_v": cv_csv if cv_csv is not None else "",
            "gt_center_u":    f"{gt_cx:.2f}"   if gt_cx   is not None else "",
            "gt_center_v":    f"{gt_cy:.2f}"   if gt_cy   is not None else "",
            "gt_box_diag_px": f"{gt_diag:.2f}" if gt_diag is not None else "",
            "gt_method":      method,
            "err_gt_px":      f"{err_px:.3f}"   if err_px   is not None else "",
            "err_gt_norm":    f"{err_norm:.5f}" if err_norm is not None else "",
        })
        frame_idx += 1

    cap.release()
    writer.release()

    fieldnames = [
        "frame_idx", "video_time_s", "cam_cx", "cam_cy",
        "csv_centroid_u", "csv_centroid_v",
        "gt_center_u", "gt_center_v", "gt_box_diag_px",
        "gt_method", "err_gt_px", "err_gt_norm",
    ]
    with out_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_frame_rows)

    mask_frames = method_counts["mask"]
    edge_frames = method_counts["edge"]
    gt_frames   = mask_frames + edge_frames
    gt_frac     = gt_frames / max(frame_idx, 1)

    errs       = [e[0] for e in err_list]
    norms      = [e[1] for e in err_list if e[1] is not None]
    final_err  = errs[-1]  if errs else None
    mean_err   = float(np.mean(errs))   if errs else None
    median_err = float(np.median(errs)) if errs else None
    mean_norm  = float(np.mean(norms))  if norms else None
    med_diag   = float(np.median(diag_list)) if diag_list else None

    if verbose:
        print(f"Frames: {frame_idx}  gt_detected: {gt_frames} "
              f"({gt_frac*100:.0f}%)  "
              f"[mask={mask_frames} edge={edge_frames}]  "
              f"err_pairs: {len(errs)}")
        if errs:
            print(f"Error : mean={mean_err:.1f}px  "
                  f"median={median_err:.1f}px  "
                  f"final={final_err:.1f}px")
        print(f"Out   : {out_video_path.name}  {out_csv_path.name}")

    return {
        "video":             video_path.name,
        "object":            object_name,
        "controller":        controller,
        "n_frames":          frame_idx,
        "gt_frac":           f"{gt_frac:.3f}",
        "mask_frames":       mask_frames,
        "edge_frames":       edge_frames,
        "median_gt_diag_px": f"{med_diag:.1f}" if med_diag else "",
        "mean_err_gt_px":    f"{mean_err:.3f}"   if mean_err   is not None else "",
        "median_err_gt_px":  f"{median_err:.3f}" if median_err is not None else "",
        "final_err_gt_px":   f"{final_err:.3f}"  if final_err  is not None else "",
        "mean_err_gt_norm":  f"{mean_norm:.5f}"  if mean_norm  is not None else "",
        "csv_matched":       "1" if csv_matched else "0",
    }


# ── Entry point ----------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video", type=Path, default=None,
                    help="Process a single video instead of all matching ones")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: experiments/gt_eval/ or "
                         "experiments/gt_eval_edge/ with --edge-only)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--edge-only", action="store_true",
                    help="Skip SAM2 green-mask detection; use Canny edge only")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = (Path(__file__).parent / "gt_eval_edge"
                        if args.edge_only
                        else Path(__file__).parent / "gt_eval")

    if args.video:
        videos = [args.video.resolve()]
        if not videos[0].exists():
            sys.exit(f"Video not found: {videos[0]}")
    else:
        videos = sorted(
            p for p in REPO_ROOT.glob("vs_dinov2_*.mp4")
            if VIDEO_RE.match(p.name)
        )
        if not videos:
            sys.exit(f"No matching videos found in {REPO_ROOT}")
        print(f"Found {len(videos)} videos to process.")

    summary_rows = []
    for vp in videos:
        row = process_video(vp, args.out_dir, verbose=not args.quiet,
                            edge_only=args.edge_only)
        if row:
            summary_rows.append(row)

    if summary_rows:
        summary_path = args.out_dir / "gt_summary.csv"
        fieldnames = [
            "video", "object", "controller", "n_frames",
            "gt_frac", "mask_frames", "edge_frames", "median_gt_diag_px",
            "mean_err_gt_px", "median_err_gt_px", "final_err_gt_px",
            "mean_err_gt_norm", "csv_matched",
        ]
        # When processing a single video (--video), merge into the existing
        # summary so we don't overwrite results from other videos.
        if args.video and summary_path.exists():
            existing: list[dict] = []
            with summary_path.open(newline="") as f:
                existing = list(csv.DictReader(f))
            new_video_names = {r["video"] for r in summary_rows}
            merged = [r for r in existing if r["video"] not in new_video_names]
            merged.extend(summary_rows)
            summary_rows = merged

        with summary_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary written to {summary_path}")

    n_with_gt = sum(1 for r in summary_rows if float(r.get("gt_frac", 0)) > 0)
    print(f"Done. {n_with_gt}/{len(summary_rows)} videos with GT detected.")


if __name__ == "__main__":
    main()
