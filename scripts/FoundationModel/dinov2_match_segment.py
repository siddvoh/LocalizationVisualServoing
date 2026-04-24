#!/usr/bin/env python3
# DINOv2 patch-match + ResNet color similarity + SAM2 mask refinement
import sys
import os
import argparse

import cv2
import numpy as np

THIRD_PARTY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third-party")
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = os.path.join(THIRD_PARTY_ROOT, "sam2/checkpoints/sam2.1_hiera_large.pt")

try:
    import torch
    import torch.nn.functional as F
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    print("ERROR: PyTorch is required")
    sys.exit(1)

_dinov2_model = None

def _get_dinov2():
    global _dinov2_model
    if _dinov2_model is None:
        print("Loading DINOv2 (ViT-B/14)...")
        _dinov2_model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14",
            pretrained=True,
        )
        _dinov2_model = _dinov2_model.to(DEVICE).eval()
        print(f"DINOv2 loaded on {DEVICE}")
    return _dinov2_model


def _preprocess_for_dinov2(image_bgr: np.ndarray, patch_size: int = 14):
    """resize to patch multiple, imagenet-normalise, return (1,3,H,W)"""
    h, w = image_bgr.shape[:2]
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor, new_h, new_w


def extract_patch_features(image_bgr: np.ndarray, patch_size: int = 14):
    """returns (features, proc_h, proc_w) where features is (nh, nw, D)"""
    model = _get_dinov2()
    tensor, proc_h, proc_w = _preprocess_for_dinov2(image_bgr, patch_size)

    with torch.no_grad():
        out = model.forward_features(tensor)
        patch_tokens = out["x_norm_patchtokens"]

    n_h = proc_h // patch_size
    n_w = proc_w // patch_size
    features = patch_tokens[0].reshape(n_h, n_w, -1).cpu().numpy()
    return features, proc_h, proc_w


_resnet_model = None

def _get_resnet():
    global _resnet_model
    if _resnet_model is None:
        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # up to layer4, stride 32, 512-d
        _resnet_model = torch.nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        ).to(DEVICE).eval()
        print(f"ResNet18 feature extractor loaded on {DEVICE}")
    return _resnet_model


def _resnet_preprocess(image_bgr: np.ndarray):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def compute_resnet_similarity(
    ref_bgr: np.ndarray,
    ref_alpha: np.ndarray,
    scene_bgr: np.ndarray,
    scene_proc_h: int,
    scene_proc_w: int,
    patch_size: int = 14,
):
    """
    Compute per-patch feature similarity using ResNet18 conv features.

    Extracts dense ResNet18 feature maps from reference foreground and scene,
    then computes cosine similarity between the mean reference foreground
    feature and each spatial location in the scene.  The result is resampled
    to DINOv2 patch resolution so it can be combined with the DINOv2 map.

    Returns
    -------
    resnet_map : (n_patches_h, n_patches_w) similarity per DINOv2 patch
    """
    model = _get_resnet()
    sh_patches = scene_proc_h // patch_size
    sw_patches = scene_proc_w // patch_size

    with torch.no_grad():
        ref_tensor = _resnet_preprocess(ref_bgr)
        ref_feats = model(ref_tensor)

        fh, fw = ref_feats.shape[2], ref_feats.shape[3]
        ref_alpha_resized = cv2.resize(
            ref_alpha, (fw, fh), interpolation=cv2.INTER_NEAREST)
        fg_mask = torch.from_numpy(
            (ref_alpha_resized > 128).astype(np.float32)
        ).to(DEVICE)

        if fg_mask.sum() < 1:
            return np.ones((sh_patches, sw_patches), dtype=np.float32)

        # mean-pool fg features
        ref_feats_flat = ref_feats[0]
        fg_mask_3d = fg_mask.unsqueeze(0)
        ref_fg_feat = (ref_feats_flat * fg_mask_3d).sum(dim=(1, 2)) / fg_mask.sum()
        ref_fg_feat = F.normalize(ref_fg_feat, dim=0)

        scene_resized = cv2.resize(
            scene_bgr, (scene_proc_w, scene_proc_h),
            interpolation=cv2.INTER_LINEAR)
        scene_tensor = _resnet_preprocess(scene_resized)
        scene_feats = model(scene_tensor)

        scene_feats_norm = F.normalize(scene_feats[0], dim=0)
        sim = torch.einsum('d,dhw->hw', ref_fg_feat, scene_feats_norm)
        sim = sim.cpu().numpy()

        resnet_map = cv2.resize(
            sim, (sw_patches, sh_patches),
            interpolation=cv2.INTER_LINEAR)

    return resnet_map


def compute_similarity_map(
    ref_features: np.ndarray,
    ref_mask_patches: np.ndarray,
    scene_features: np.ndarray,
):
    """mean cosine sim between ref fg patches and every scene patch"""
    fg_mask = ref_mask_patches > 0
    fg_feats = ref_features[fg_mask]
    if len(fg_feats) == 0:
        print("WARNING: no foreground patches in reference mask")
        return np.zeros(scene_features.shape[:2], dtype=np.float32)

    fg_feats = fg_feats / (np.linalg.norm(fg_feats, axis=1, keepdims=True) + 1e-8)
    sh, sw, D = scene_features.shape
    scene_flat = scene_features.reshape(-1, D)
    scene_flat = scene_flat / (np.linalg.norm(scene_flat, axis=1, keepdims=True) + 1e-8)

    sim_scores = np.zeros(scene_flat.shape[0], dtype=np.float32)
    chunk_size = 512
    for i in range(0, len(scene_flat), chunk_size):
        chunk = scene_flat[i:i + chunk_size]
        cos_sim = chunk @ fg_feats.T
        sim_scores[i:i + chunk_size] = cos_sim.mean(axis=1)

    sim_map = sim_scores.reshape(sh, sw)
    return sim_map


def combine_similarity_maps(dinov2_sim, resnet_sim, alpha=0.5):
    """alpha*dinov2 + (1-alpha)*resnet, each normalised to [0,1]"""
    d_min, d_max = dinov2_sim.min(), dinov2_sim.max()
    if d_max - d_min > 1e-8:
        d_norm = (dinov2_sim - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros_like(dinov2_sim)

    r_min, r_max = resnet_sim.min(), resnet_sim.max()
    if r_max - r_min > 1e-8:
        r_norm = (resnet_sim - r_min) / (r_max - r_min)
    else:
        r_norm = np.ones_like(resnet_sim)

    combined = alpha * d_norm + (1.0 - alpha) * r_norm
    return combined


def similarity_to_bbox(
    sim_map: np.ndarray,
    scene_h: int,
    scene_w: int,
    patch_size: int = 14,
    threshold_percentile: float = 80,
):
    """returns ((x1,y1,x2,y2), binary_mask, sim_upscaled, mean_sim|-1.0)"""
    sim_upscaled = cv2.resize(
        sim_map, (scene_w, scene_h), interpolation=cv2.INTER_LINEAR
    )

    thresh_val = np.percentile(sim_upscaled, threshold_percentile)
    binary = (sim_upscaled >= thresh_val).astype(np.uint8) * 255

    # keep CLOSE mild so two stacked identical boxes don't fuse into one CC
    kern_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kern_open,  iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern_close, iterations=1)

    # rank CCs by mean sim; break ties by picking topmost
    pick = select_best_cc_by_similarity(
        binary, sim_upscaled, min_area_frac=0.002, sim_tie_eps=0.02)
    if pick is None:
        print("WARNING: no connected component found, using peak location")
        peak = np.unravel_index(np.argmax(sim_upscaled), sim_upscaled.shape)
        cy, cx = peak
        pad = 50
        return (
            max(0, cx - pad), max(0, cy - pad),
            min(scene_w, cx + pad), min(scene_h, cy + pad),
        ), binary, sim_upscaled, -1.0

    if pick["n_candidates"] > 1:
        print(
            f"  DINOv2 picker: {pick['n_candidates']} CCs → "
            f"mean_sim={pick['mean_sim']:.3f}"
            f"{' (topmost-tiebreak)' if pick['tiebreak_used'] else ''}"
            f"; dropped sims={[round(s, 3) for s in pick['other_mean_sims']]}"
        )
    else:
        print(f"  Selected component with mean similarity {pick['mean_sim']:.3f}")

    comp_mask = pick["mask"]
    x1, y1, x2, y2 = pick["bbox"]
    pad = 15
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(scene_w, x2 + pad)
    y2 = min(scene_h, y2 + pad)

    return (x1, y1, x2, y2), comp_mask, sim_upscaled, pick["mean_sim"]


_sam2_pred = None

def _get_sam2():
    global _sam2_pred
    if _sam2_pred is None:
        sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "sam2"))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=DEVICE)
        _sam2_pred = SAM2ImagePredictor(model)
        print(f"SAM2 loaded on {DEVICE}")
    return _sam2_pred


def select_best_cc_by_similarity(
    mask: np.ndarray,
    sim_upscaled: np.ndarray,
    min_area_frac: float = 0.002,
    sim_tie_eps: float = 0.02,
):
    """pick best CC by mean sim; tiebreak = topmost (smallest y_min)"""
    if mask is None:
        return None
    h, w = mask.shape[:2]
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    if n_cc <= 1:
        return None

    min_area = int(min_area_frac * h * w)
    candidates = []
    for cc_id in range(1, n_cc):
        area = int(stats[cc_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cc_mask = labels == cc_id
        mean_sim = float(np.mean(sim_upscaled[cc_mask]))
        y_min = int(stats[cc_id, cv2.CC_STAT_TOP])
        x_min = int(stats[cc_id, cv2.CC_STAT_LEFT])
        bw = int(stats[cc_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[cc_id, cv2.CC_STAT_HEIGHT])
        candidates.append(dict(
            cc_id=cc_id, area=area, mean_sim=mean_sim,
            y_min=y_min,
            bbox=(x_min, y_min, x_min + bw, y_min + bh),
        ))

    if not candidates:
        return None

    best_sim = max(c["mean_sim"] for c in candidates)
    top_tier = [c for c in candidates
                if best_sim - c["mean_sim"] <= sim_tie_eps]
    winner = min(top_tier, key=lambda c: c["y_min"])

    winner_mask = (labels == winner["cc_id"]).astype(np.uint8) * 255
    other_sims = sorted(
        (c["mean_sim"] for c in candidates if c["cc_id"] != winner["cc_id"]),
        reverse=True,
    )
    return dict(
        mask=winner_mask,
        bbox=winner["bbox"],
        mean_sim=winner["mean_sim"],
        area=winner["area"],
        y_min=winner["y_min"],
        n_candidates=len(candidates),
        tiebreak_used=(len(top_tier) > 1 and len(candidates) > 1),
        other_mean_sims=list(other_sims),
    )


def _select_topmost_mask(mask: np.ndarray) -> np.ndarray:
    """keep only the topmost CC"""
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n_cc <= 2:
        return mask

    best_cc = 1
    best_top_y = stats[1, cv2.CC_STAT_TOP]
    for cc_id in range(2, n_cc):
        area = stats[cc_id, cv2.CC_STAT_AREA]
        if area < 100:
            continue
        top_y = stats[cc_id, cv2.CC_STAT_TOP]
        if top_y < best_top_y:
            best_top_y = top_y
            best_cc = cc_id

    return (labels == best_cc).astype(np.uint8) * 255


def refine_with_sam2(image_bgr: np.ndarray, bbox, select_topmost: bool = True,
                     return_logits: bool = False):
    """SAM2 bbox-prompted mask; if select_topmost, peel off top object on stacks"""
    x1, y1, x2, y2 = bbox
    box_np = np.array([x1, y1, x2, y2], dtype=np.float32)

    pred = _get_sam2()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred.set_image(rgb)

    masks, scores, logits = pred.predict(
        box=box_np,
        multimask_output=True,
    )

    best_idx = int(np.argmax(scores))
    mask = (masks[best_idx] > 0).astype(np.uint8) * 255
    score = float(scores[best_idx])
    best_logits = logits[best_idx:best_idx + 1]
    print(f"SAM2: best mask score = {score:.3f}")

    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

    def _ret(m, s, lg):
        return (m, s, lg) if return_logits else (m, s)

    if not select_topmost:
        return _ret(mask, score, best_logits)

    # topmost selection: try CC split first, then point-prompt top band
    fg_rows = np.any(mask > 0, axis=1)
    fg_cols = np.any(mask > 0, axis=0)
    if not np.any(fg_rows):
        return _ret(mask, score, best_logits)
    mask_y1 = int(np.argmax(fg_rows))
    mask_y2 = int(len(fg_rows) - np.argmax(fg_rows[::-1]))
    mask_x1 = int(np.argmax(fg_cols))
    mask_x2 = int(len(fg_cols) - np.argmax(fg_cols[::-1]))
    mask_h = mask_y2 - mask_y1
    mask_w = mask_x2 - mask_x1
    original_area = np.count_nonzero(mask)

    if mask_w == 0 or mask_h == 0:
        return _ret(mask, score, best_logits)

    top_mask = _select_topmost_mask(mask)
    n_cc = cv2.connectedComponentsWithStats(mask)[0] - 1
    if n_cc > 1:
        print(f"  Split into {n_cc} components, selected topmost")
        fg = np.argwhere(top_mask > 0)
        cy, cx = fg.mean(axis=0).astype(int)
        point = np.array([[cx, cy]], dtype=np.float32)
        label = np.array([1], dtype=np.int32)
        masks2, scores2, logits2 = pred.predict(
            point_coords=point, point_labels=label,
            multimask_output=True)
        best2 = int(np.argmax(scores2))
        refined = (masks2[best2] > 0).astype(np.uint8) * 255
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kern, iterations=2)
        print(f"  Refined topmost mask: score={scores2[best2]:.3f}")
        return _ret(refined, float(scores2[best2]), logits2[best2:best2 + 1])

    # fallback: re-run SAM2 with cropped top-40% bbox
    crop_y2 = min(mask_y1 + int(mask_h * 0.4), image_bgr.shape[0])
    top_box = np.array([mask_x1, mask_y1, mask_x2, crop_y2], dtype=np.float32)
    print(f"  Top-crop bbox: ({mask_x1}, {mask_y1}, {mask_x2}, {crop_y2})")

    masks2, scores2, logits2 = pred.predict(
        box=top_box, multimask_output=True)

    if masks2 is not None and len(masks2) > 0:
        best2 = int(np.argmax(scores2))
        refined = (masks2[best2] > 0).astype(np.uint8) * 255
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kern, iterations=2)
        cnts, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(refined, cnts, -1, 255, cv2.FILLED)
        refined_area = np.count_nonzero(refined)
        print(f"  Topmost mask: score={scores2[best2]:.3f}, "
              f"area ratio={refined_area / max(original_area, 1):.2f}")
        return _ret(refined, float(scores2[best2]), logits2[best2:best2 + 1])

    print("  Could not isolate topmost — using full mask")
    return _ret(mask, score, best_logits)


def compute_metrics(sim_map, mask, bbox, scene_shape):
    h, w = scene_shape[:2]
    mask_area = np.count_nonzero(mask)
    total_area = h * w

    metrics = {
        "mask_area_px": mask_area,
        "mask_area_pct": 100.0 * mask_area / total_area,
        "bbox": list(bbox),
        "bbox_area_px": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
    }

    if sim_map is not None:
        sim_in_mask = sim_map[mask > 0] if mask_area > 0 else np.array([0])
        sim_outside = sim_map[mask == 0]
        metrics["sim_in_mask_mean"] = float(np.mean(sim_in_mask))
        metrics["sim_in_mask_std"] = float(np.std(sim_in_mask))
        metrics["sim_outside_mean"] = float(np.mean(sim_outside))
        metrics["sim_contrast"] = metrics["sim_in_mask_mean"] - metrics["sim_outside_mean"]

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return metrics


def visualize_results(
    scene_bgr, ref_bgr, ref_alpha,
    sim_upscaled, dinov2_bbox, dinov2_binary,
    sam_mask, sam_score, metrics,
    output_dir, display=True,
):
    import matplotlib
    if not display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene_rgb = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    h, w = scene_bgr.shape[:2]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1: Reference image
    if ref_alpha is not None:
        ref_display = ref_rgb.copy().astype(np.float32)
        alpha_f = ref_alpha[:, :, None].astype(np.float32) / 255.0
        checker = _checkerboard(ref_rgb.shape[0], ref_rgb.shape[1], sq=16)
        ref_display = (checker * (1 - alpha_f) + ref_display * alpha_f).astype(np.uint8)
    else:
        ref_display = ref_rgb
    axes[0, 0].imshow(ref_display)
    axes[0, 0].set_title("1. Reference Object", fontsize=13)
    axes[0, 0].axis("off")

    # Panel 2: Scene image
    axes[0, 1].imshow(scene_rgb)
    axes[0, 1].set_title("2. Scene Image", fontsize=13)
    axes[0, 1].axis("off")

    # Panel 3: DINOv2 similarity heatmap
    axes[0, 2].imshow(scene_rgb, alpha=0.4)
    im = axes[0, 2].imshow(sim_upscaled, cmap="jet", alpha=0.6)
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    axes[0, 2].set_title("3. DINOv2 Similarity Heatmap", fontsize=13)
    axes[0, 2].axis("off")

    # Panel 4: DINOv2 bbox on scene
    bbox_vis = scene_rgb.copy()
    x1, y1, x2, y2 = dinov2_bbox
    cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    axes[1, 0].imshow(bbox_vis)
    axes[1, 0].set_title("4. DINOv2 Bounding Box", fontsize=13)
    axes[1, 0].axis("off")

    # Panel 5: SAM2 mask overlay
    mask_vis = scene_rgb.copy()
    overlay = np.zeros_like(mask_vis)
    overlay[:, :, 1] = sam_mask  # green channel
    mask_vis = cv2.addWeighted(mask_vis, 0.55, overlay, 0.45, 0)
    # Draw contour
    contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_vis, contours, -1, (255, 255, 0), 2)
    # Draw bbox
    cv2.rectangle(mask_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    axes[1, 1].imshow(mask_vis)
    axes[1, 1].set_title(f"5. SAM2 Mask (score={sam_score:.3f})", fontsize=13)
    axes[1, 1].axis("off")

    # Panel 6: Segmented object (masked cutout)
    cutout = scene_rgb.copy()
    cutout[sam_mask == 0] = 255  # white background for non-object
    axes[1, 2].imshow(cutout)
    metrics_text = (
        f"Mask area: {metrics['mask_area_pct']:.1f}%\n"
        f"Sim in mask: {metrics.get('sim_in_mask_mean', 0):.3f}\n"
        f"Sim outside: {metrics.get('sim_outside_mean', 0):.3f}\n"
        f"Contrast: {metrics.get('sim_contrast', 0):.3f}"
    )
    axes[1, 2].text(
        0.02, 0.98, metrics_text,
        transform=axes[1, 2].transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    axes[1, 2].set_title("6. Segmented Object + Metrics", fontsize=13)
    axes[1, 2].axis("off")

    plt.suptitle(
        "DINOv2 Feature Matching + SAM2 Segmentation",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()

    vis_path = os.path.join(output_dir, "dinov2_match_result.png")
    fig.savefig(vis_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved: {vis_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)


def _checkerboard(h, w, sq=16):
    board = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(0, h, sq):
        for x in range(0, w, sq):
            c = 220.0 if ((x // sq) + (y // sq)) % 2 == 0 else 180.0
            board[y:y + sq, x:x + sq] = c
    return board


def main():
    parser = argparse.ArgumentParser(
        description="Segment a reference object from a scene using "
                    "DINOv2 feature matching + SAM2",
    )
    parser.add_argument("--scene", "-s", required=True,
                        help="Path to the scene image")
    parser.add_argument("--reference", "-r", required=True,
                        help="Path to the reference image (RGBA with transparent bg)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: same as scene)")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't show the matplotlib window")
    parser.add_argument("--threshold-pct", type=float, default=80,
                        help="Percentile threshold for similarity map (default: 80)")
    parser.add_argument("--resnet-weight", type=float, default=0.5,
                        help="Weight for ResNet18 similarity (0=DINOv2 only, 1=ResNet only, default: 0.5)")
    args = parser.parse_args()

    scene_bgr = cv2.imread(args.scene)
    if scene_bgr is None:
        print(f"ERROR: could not load scene: {args.scene}")
        sys.exit(1)

    ref_full = cv2.imread(args.reference, cv2.IMREAD_UNCHANGED)
    if ref_full is None:
        print(f"ERROR: could not load reference: {args.reference}")
        sys.exit(1)

    # Extract alpha channel from reference
    if ref_full.shape[2] == 4:
        ref_bgr = ref_full[:, :, :3]
        ref_alpha = ref_full[:, :, 3]
        print(f"Reference: {ref_bgr.shape[1]}x{ref_bgr.shape[0]}, "
              f"alpha channel present, "
              f"fg pixels: {np.count_nonzero(ref_alpha > 128)}")
    else:
        ref_bgr = ref_full
        ref_alpha = np.ones(ref_full.shape[:2], dtype=np.uint8) * 255
        print("WARNING: reference has no alpha channel, treating entire image as object")

    sh, sw = scene_bgr.shape[:2]
    print(f"Scene: {sw}x{sh}")

    output_dir = args.output or os.path.dirname(os.path.abspath(args.scene))
    os.makedirs(output_dir, exist_ok=True)

    patch_size = 14
    print("\n--- Step 1: Extracting DINOv2 features ---")

    ref_features, ref_proc_h, ref_proc_w = extract_patch_features(ref_bgr, patch_size)
    print(f"  Reference features: {ref_features.shape}")

    scene_features, scene_proc_h, scene_proc_w = extract_patch_features(scene_bgr, patch_size)
    print(f"  Scene features: {scene_features.shape}")

    ref_alpha_resized = cv2.resize(
        ref_alpha, (ref_proc_w, ref_proc_h), interpolation=cv2.INTER_NEAREST
    )
    rh_patches = ref_proc_h // patch_size
    rw_patches = ref_proc_w // patch_size
    ref_mask_patches = np.zeros((rh_patches, rw_patches), dtype=np.uint8)
    for py in range(rh_patches):
        for px in range(rw_patches):
            patch_region = ref_alpha_resized[
                py * patch_size:(py + 1) * patch_size,
                px * patch_size:(px + 1) * patch_size,
            ]
            if np.mean(patch_region > 128) > 0.5:
                ref_mask_patches[py, px] = 1

    fg_patches = np.count_nonzero(ref_mask_patches)
    total_patches = ref_mask_patches.size
    print(f"  Reference foreground patches: {fg_patches}/{total_patches}")

    print("\n--- Step 2: Computing DINOv2 similarity map ---")
    dinov2_sim = compute_similarity_map(ref_features, ref_mask_patches, scene_features)
    print(f"  DINOv2 similarity range: [{dinov2_sim.min():.3f}, {dinov2_sim.max():.3f}]")

    print("\n--- Step 2b: Computing ResNet18 feature similarity map ---")
    resnet_sim = compute_resnet_similarity(
        ref_bgr, ref_alpha, scene_bgr,
        scene_proc_h, scene_proc_w, patch_size,
    )
    print(f"  ResNet18 similarity range: [{resnet_sim.min():.3f}, {resnet_sim.max():.3f}]")

    print("\n--- Step 2c: Combining DINOv2 + ResNet18 similarity ---")
    sim_map = combine_similarity_maps(dinov2_sim, resnet_sim, alpha=1.0 - args.resnet_weight)
    print(f"  Combined similarity range: [{sim_map.min():.3f}, {sim_map.max():.3f}]")

    print("\n--- Step 3: Extracting bounding box ---")
    bbox, dinov2_binary, sim_upscaled, _sim_mean = similarity_to_bbox(
        sim_map, sh, sw, patch_size, args.threshold_pct,
    )
    print(f"  DINOv2+Color bbox: {bbox}")

    print("\n--- Step 4: SAM2 mask refinement ---")
    sam_mask, sam_score = refine_with_sam2(scene_bgr, bbox, select_topmost=True)
    mask_area = np.count_nonzero(sam_mask)
    print(f"  Mask area: {mask_area} px ({100.0 * mask_area / (sh * sw):.1f}%)")

    metrics = compute_metrics(sim_upscaled, sam_mask, bbox, scene_bgr.shape)

    import json
    metrics["sam_score"] = sam_score
    metrics["threshold_pct"] = args.threshold_pct
    metrics["resnet_weight"] = args.resnet_weight
    metrics_path = os.path.join(output_dir, "metrics.json")
    def _json_safe(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, list):
            return [_json_safe(x) for x in v]
        return v
    with open(metrics_path, "w") as f:
        json.dump({k: _json_safe(v) for k, v in metrics.items()}, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    mask_path = os.path.join(output_dir, "dinov2_sam_mask.png")
    cv2.imwrite(mask_path, sam_mask)
    print(f"\nMask saved: {mask_path}")

    heatmap_u8 = (np.clip(sim_upscaled, 0, 1) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(output_dir, "dinov2_similarity_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap_color)
    print(f"Heatmap saved: {heatmap_path}")

    overlay_vis = scene_bgr.copy()
    green_overlay = np.zeros_like(overlay_vis)
    green_overlay[:, :, 1] = sam_mask
    overlay_vis = cv2.addWeighted(overlay_vis, 0.6, green_overlay, 0.4, 0)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(overlay_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_vis, contours, -1, (0, 255, 255), 2)
    overlay_path = os.path.join(output_dir, "dinov2_sam_overlay.png")
    cv2.imwrite(overlay_path, overlay_vis)
    print(f"Overlay saved: {overlay_path}")

    print("\n--- Step 6: Visualization ---")
    visualize_results(
        scene_bgr, ref_bgr, ref_alpha,
        sim_upscaled, bbox, dinov2_binary,
        sam_mask, sam_score, metrics,
        output_dir, display=not args.no_display,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
