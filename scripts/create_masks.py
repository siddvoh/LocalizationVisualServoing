#!/usr/bin/env python3
# SAM2-based cutouts for boxes in objects/ -> masked_objects/
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
THIRD_PARTY = HERE / "third-party"
SAM2_DIR = THIRD_PARTY / "sam2"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = str(SAM2_DIR / "checkpoints" / "sam2.1_hiera_large.pt")

sys.path.insert(0, str(SAM2_DIR))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_predictor(device: str) -> SAM2ImagePredictor:
    model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)
    return SAM2ImagePredictor(model)


def center_prompts(h: int, w: int, y_spread: float):
    """positive points near centre + corner negatives"""
    cx, cy = w // 2, h // 2
    dx = w // 8
    half = y_spread * h / 2.0
    ys = [
        int(cy - half),
        int(cy - half * 0.5),
        cy,
        int(cy + half * 0.5),
        int(cy + half),
    ]
    pos = [[cx, max(0, min(h - 1, y))] for y in ys]
    pos += [[cx - dx, cy], [cx + dx, cy]]
    pos = np.array(pos, dtype=np.float32)

    margin = 8
    neg = np.array(
        [
            [margin, margin],
            [w - margin, margin],
            [margin, h - margin],
            [w - margin, h - margin],
        ],
        dtype=np.float32,
    )
    points = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate(
        [np.ones(len(pos), dtype=np.int32), np.zeros(len(neg), dtype=np.int32)]
    )
    return points, labels


def center_box_prompt(h: int, w: int, x_frac: float, y_frac: float) -> np.ndarray:
    x_margin = int(w * x_frac)
    y_margin = int(h * y_frac)
    return np.array(
        [x_margin, y_margin, w - x_margin, h - y_margin], dtype=np.float32
    )


def largest_component_containing(mask: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    """Keep only the connected component that contains `point`, else the largest."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n <= 1:
        return mask
    cx, cy = point
    cx = int(np.clip(cx, 0, mask.shape[1] - 1))
    cy = int(np.clip(cy, 0, mask.shape[0] - 1))
    target = int(labels[cy, cx])
    if target == 0:
        # centre fell on bg, take largest CC
        areas = stats[1:, cv2.CC_STAT_AREA]
        target = int(np.argmax(areas)) + 1
    return (labels == target).astype(np.uint8) * 255


def refine_mask(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern, iterations=1)
    # fill holes so alpha is opaque inside
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def strip_backdrop_leaks(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    centre: tuple[int, int],
    v_thresh: int = 65,
    s_thresh: int = 60,
    max_drop_frac: float = 0.18,
) -> np.ndarray:
    """strip dark+grey backdrop leaks that bridge inside/outside and reach the border"""
    m_bin = (mask > 0).astype(np.uint8)
    mask_area = int(m_bin.sum())
    if mask_area == 0:
        return mask

    h, w = m_bin.shape
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    backdrop = ((v < v_thresh) & (s < s_thresh)).astype(np.uint8)

    # close so drape + intrusion form one CC
    bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    backdrop = cv2.morphologyEx(backdrop, cv2.MORPH_CLOSE, bridge)

    n_cc, labels = cv2.connectedComponents(backdrop)
    if n_cc <= 1:
        return mask

    border_mask = np.zeros_like(m_bin)
    border_mask[0, :] = 1
    border_mask[-1, :] = 1
    border_mask[:, 0] = 1
    border_mask[:, -1] = 1

    to_drop = np.zeros_like(m_bin)
    for cc_id in range(1, n_cc):
        cc_mask = labels == cc_id
        if cc_mask.sum() < 500:
            continue
        inside_part = cc_mask & (m_bin > 0)
        outside_part = cc_mask & (m_bin == 0)
        inside = int(inside_part.sum())
        outside = int(outside_part.sum())
        if inside < 150 or outside < 150:
            continue
        if not np.any(outside_part & (border_mask > 0)):
            continue
        if inside > max_drop_frac * mask_area:
            # too big to be a leak; likely a dark box panel
            continue
        to_drop |= inside_part.astype(np.uint8)

    if to_drop.sum() == 0:
        return mask

    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = (m_bin & (1 - to_drop)).astype(np.uint8) * 255
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kern, iterations=1)
    cleaned = largest_component_containing(cleaned, centre)
    return cleaned


PROMPT_CONFIGS = [
    # (name, y_spread as fraction of h, box x_frac, box y_frac)
    ("wide",   0.50, 0.15, 0.08),
    ("medium", 0.30, 0.20, 0.15),
    ("tight",  0.15, 0.28, 0.25),
]


def _run_single_prompt(
    predictor: SAM2ImagePredictor,
    h: int,
    w: int,
    y_spread: float,
    x_frac: float,
    y_frac: float,
) -> tuple[np.ndarray | None, float]:
    points, labels = center_prompts(h, w, y_spread)
    box = center_box_prompt(h, w, x_frac, y_frac)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=box,
        multimask_output=True,
    )
    centre = (w // 2, h // 2)
    img_area = h * w
    best_mask, best_score = None, -1.0

    for m, s in zip(masks, scores):
        m_u8 = (m > 0).astype(np.uint8) * 255
        m_cc = largest_component_containing(m_u8, centre)
        frac = int(np.count_nonzero(m_cc)) / img_area
        # skip tiny/whole-frame masks
        if frac < 0.02 or frac > 0.92:
            continue
        if float(s) > best_score:
            best_score = float(s)
            best_mask = m_cc
    return best_mask, best_score


def predict_box_mask(
    predictor: SAM2ImagePredictor, image_bgr: np.ndarray
) -> tuple[np.ndarray, float, str]:
    """try each prompt config, keep highest-scoring valid mask"""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    best_mask, best_score, best_name = None, -1.0, "none"
    for name, y_spread, xf, yf in PROMPT_CONFIGS:
        mask, score = _run_single_prompt(predictor, h, w, y_spread, xf, yf)
        if mask is None:
            continue
        if score > best_score:
            best_mask, best_score, best_name = mask, score, name

    if best_mask is None:
        # fallback: take wide-config top mask even if degenerate
        points, labels = center_prompts(h, w, PROMPT_CONFIGS[0][1])
        box = center_box_prompt(
            h, w, PROMPT_CONFIGS[0][2], PROMPT_CONFIGS[0][3]
        )
        masks, scores, _ = predictor.predict(
            point_coords=points, point_labels=labels, box=box,
            multimask_output=True,
        )
        idx = int(np.argmax(scores))
        best_mask = (masks[idx] > 0).astype(np.uint8) * 255
        best_score = float(scores[idx])
        best_name = "fallback"
        best_mask = largest_component_containing(
            best_mask, (w // 2, h // 2)
        )

    centre = (w // 2, h // 2)
    cleaned = strip_backdrop_leaks(image_bgr, best_mask, centre)
    return refine_mask(cleaned), best_score, best_name


def to_rgba(image_bgr: np.ndarray, mask: np.ndarray, crop: bool) -> np.ndarray:
    """Combine BGR image + binary mask into a BGRA cutout; optionally crop."""
    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask

    if not crop:
        return bgra

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return bgra
    pad = 4
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(bgra.shape[1], int(xs.max()) + 1 + pad)
    y2 = min(bgra.shape[0], int(ys.max()) + 1 + pad)
    return bgra[y1:y2, x1:x2].copy()


def debug_overlay(image_bgr: np.ndarray, mask: np.ndarray, score: float) -> np.ndarray:
    overlay = image_bgr.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = mask
    overlay = cv2.addWeighted(overlay, 0.7, green, 0.3, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    cv2.putText(
        overlay,
        f"score={score:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", default=str(HERE / "objects"))
    parser.add_argument("--out-dir", default=str(HERE / "masked_objects"))
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Keep original image size (transparent outside mask) instead of "
        "cropping to the mask bbox.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also write overlay previews to masked_objects/debug/.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated filename stems to process (e.g. 'tofu_box,lamp_box').",
    )
    args = parser.parse_args()

    objects_dir = Path(args.objects_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in objects_dir.iterdir() if p.suffix.lower() in exts)

    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        images = [p for p in images if p.stem in wanted]

    if not images:
        print(f"No images found under {objects_dir}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2 on {device} ...")
    predictor = build_predictor(device)
    print(f"Loaded. Processing {len(images)} image(s) from {objects_dir}")

    ok = 0
    for img_path in images:
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[skip] could not read {img_path.name}")
            continue

        try:
            mask, score, cfg_name = predict_box_mask(predictor, image_bgr)
        except Exception as exc:  # pragma: no cover - runtime robustness
            print(f"[fail] {img_path.name}: {exc}")
            continue

        rgba = to_rgba(image_bgr, mask, crop=not args.no_crop)
        out_path = out_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), rgba)

        if args.debug:
            preview = debug_overlay(image_bgr, mask, score)
            cv2.imwrite(str(debug_dir / f"{img_path.stem}_debug.png"), preview)

        area = int(np.count_nonzero(mask))
        frac = area / (mask.shape[0] * mask.shape[1])
        print(
            f"[ok]  {img_path.name:<30s}  cfg={cfg_name:<6s}  "
            f"score={score:.3f}  mask={frac*100:.1f}%  ->  {out_path.name}"
        )
        ok += 1

    print(f"\nDone: {ok}/{len(images)} written to {out_dir}")
    return 0 if ok == len(images) else 2


if __name__ == "__main__":
    sys.exit(main())
