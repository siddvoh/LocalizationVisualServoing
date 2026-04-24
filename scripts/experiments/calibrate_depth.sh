#!/usr/bin/env bash
# one-off depth calibration. place the object DEPTH_CAL_M metres in front of the
# ZED, wait for the mask to lock, then press 'c'. scale gets saved to
# experiments/depth_scale.json. q to quit.
set -euo pipefail

source "$(dirname "$0")/_env.sh"

REFERENCE="${REFERENCE:-experiments/input_image_transparent.png}"
DEPTH_CAL_M="${DEPTH_CAL_M:-0.30}"

echo "calibrating depth against $REFERENCE at ${DEPTH_CAL_M} m"
echo "press 'c' in the opencv window once the mask has locked."

exec python FoundationModel/dinov2_servo.py \
    --reference "$REFERENCE" \
    --mode ekf \
    --depth-cal-m "$DEPTH_CAL_M" \
    --no-robot
