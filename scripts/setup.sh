#!/usr/bin/env bash
# run once after clone

set -euo pipefail

echo "[1/3] Initializing git submodules..."
git submodule update --init --recursive
echo "      Submodules ready."

echo "[2/3] Downloading model weights..."

# --- Depth-Anything-V2 -------------------------------------------------------
DA2_DIR="third-party/Depth-Anything-V2/checkpoints"
mkdir -p "$DA2_DIR"
if [ ! -f "$DA2_DIR/depth_anything_v2_vits.pth" ]; then
    echo "  Downloading Depth-Anything-V2 ViT-S checkpoint..."
    wget -q --show-progress -O "$DA2_DIR/depth_anything_v2_vits.pth" \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
else
    echo "  Depth-Anything-V2 weights already present, skipping."
fi

# --- GroundingDINO -----------------------------------------------------------
GDINO_DIR="third-party/GroundingDINO/weights"
mkdir -p "$GDINO_DIR"
if [ ! -f "$GDINO_DIR/groundingdino_swint_ogc.pth" ]; then
    echo "  Downloading GroundingDINO SwinT OGC checkpoint..."
    wget -q --show-progress -O "$GDINO_DIR/groundingdino_swint_ogc.pth" \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
else
    echo "  GroundingDINO weights already present, skipping."
fi

# --- SAM 2 -------------------------------------------------------------------
SAM2_DIR="third-party/sam2/checkpoints"
mkdir -p "$SAM2_DIR"
if [ ! -f "$SAM2_DIR/sam2.1_hiera_large.pt" ]; then
    echo "  Downloading SAM 2.1 Hiera-L checkpoint..."
    wget -q --show-progress -O "$SAM2_DIR/sam2.1_hiera_large.pt" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
else
    echo "  SAM 2 weights already present, skipping."
fi

# --- SAM 3 -------------------------------------------------------------------
# SAM 3 weights are downloaded separately; see:
# https://github.com/facebookresearch/sam3
echo "  SAM 3: follow https://github.com/facebookresearch/sam3 for checkpoint download."

echo "[3/3] Done. All submodules and weights are ready."
