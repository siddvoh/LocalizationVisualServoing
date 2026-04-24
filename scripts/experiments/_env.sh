# shared env for experiment scripts, source this

source /home/akanksha/miniconda3/etc/profile.d/conda.sh
conda activate foundationpose

# strip global PYTHONPATH from ~/.bashrc
unset PYTHONPATH

export PATH=/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4

export FOUNDATIONPOSE_ROOT=$HOME/FoundationPose

PROJECT_ROOT=/home/akanksha/repo/localization_for_visual_servoing
cd "$PROJECT_ROOT"

echo "[env] $CONDA_DEFAULT_ENV @ $(pwd)"

if [ -z "${DISPLAY:-}" ]; then
    echo ""
    echo "[env] WARNING: DISPLAY is unset. OpenCV windows will not open."
    echo "              Run this from a terminal on the workstation"
    echo "              desktop, or ssh with -X forwarding."
    echo ""
fi
