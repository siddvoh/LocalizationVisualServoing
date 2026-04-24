#!/usr/bin/env bash
# runs ekf / foundationpose / ibvs back-to-back on one reference image.
# MOVES THE ROBOT.
# usage: ./run_all_experiment.sh masked_objects/cheez_it_box.png
# outputs runs/<ref>_<ts>/<ref>_{ekf,fp,pbvs}.csv + run.log
# env overrides: Z_FLOOR_MM, STOP_DEPTH_M, SKIP_MODES, FP_BOX="W H D", etc.
# operator jogs arm to desired start pose before invoking; script returns
# the arm to that pose between trials.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    cat >&2 <<EOF
Usage: $0 <path/to/reference.png>
See header of this script for env-var overrides.
EOF
    exit 2
fi
REFERENCE="$1"

source "$(dirname "$0")/_env.sh"

if [[ ! -f "$REFERENCE" ]]; then
    echo "[run-all] ERROR: reference image not found: $REFERENCE" >&2
    exit 2
fi

REFBASE="$(basename "${REFERENCE%.*}")"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/${REFBASE}_${TS}"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run.log"

Z_FLOOR_MM="${Z_FLOOR_MM:--150}"
AUTO_EXIT_CONVERGE_SEC="${AUTO_EXIT_CONVERGE_SEC:-3.0}"
AUTO_EXIT_LOST_SEC="${AUTO_EXIT_LOST_SEC:-5.0}"
AUTO_EXIT_MAX_SEC="${AUTO_EXIT_MAX_SEC:-90.0}"
# bash hard kill in case python auto-exit hangs
HARD_TIMEOUT_SEC="${HARD_TIMEOUT_SEC:-$(python - <<PY
print(float("${AUTO_EXIT_MAX_SEC}") + 30.0)
PY
)}"
CAM_TO_ROBOT="${CAM_TO_ROBOT:-zed_forward}"
ARM_IP="${ARM_IP:-192.168.1.241}"
SKIP_MODES="${SKIP_MODES:-}"
SKIP_HOMING="${SKIP_HOMING:-0}"
# slow homing so arm doesn't whip between trials
HOME_SPEED="${HOME_SPEED:-40}"
HOME_MVACC="${HOME_MVACC:-200}"
INIT_POSE_FILE="$OUT_DIR/initial_pose.json"
STOP_DEPTH_M="${STOP_DEPTH_M:-0.37}"

# FP_BOX extents (W H D in metres) by reference basename
lookup_fp_box() {
    local ref="$1"
    case "$ref" in
        *cheez_it*)           echo "0.19 0.06 0.22" ;;
        *amazon_tissue*|*tissue*) echo "0.21 0.07 0.12" ;;
        *cardboard*)          echo "0.25 0.15 0.20" ;;
        *protein*)            echo "0.16 0.04 0.05" ;;
        *brownie*)            echo "0.18 0.05 0.20" ;;
        *cake*)               echo "0.14 0.05 0.20" ;;
        *tofu*)               echo "0.10 0.05 0.14" ;;
        *baking_mix*)         echo "0.17 0.06 0.22" ;;
        *mac_and_cheese*|*mac*cheese*) echo "0.16 0.04 0.20" ;;
        *mashed_potato*)      echo "0.10 0.04 0.17" ;;
        *stuffing*)           echo "0.16 0.05 0.20" ;;
        *lamp*)               echo "0.20 0.15 0.25" ;;
        *)                    echo "" ;;
    esac
}
# fallback when refbase isn't in the lookup table
FP_BOX_FALLBACK="${FP_BOX_FALLBACK:-0.18 0.05 0.20}"
FP_BOX="${FP_BOX:-$(lookup_fp_box "$REFBASE")}"

echo "[preflight] reference: $REFERENCE"
echo "[preflight] out dir  : $OUT_DIR"

if [[ "${SKIP_ARM_PRECHECK:-0}" != "1" ]]; then
    echo "[preflight] pinging xArm at $ARM_IP ..."
    if ! ping -c 1 -W 2 "$ARM_IP" >/dev/null 2>&1; then
        echo "[preflight] ERROR: cannot reach xArm at $ARM_IP. Abort." >&2
        exit 1
    fi
fi

if ! python -c "from xarm.wrapper import XArmAPI" >/dev/null 2>&1; then
    echo "[preflight] ERROR: xarm Python SDK not importable." >&2
    exit 1
fi

has_mode() {  # returns 0 iff $1 is NOT in SKIP_MODES
    local m="$1"
    for skipped in $SKIP_MODES; do
        [[ "$skipped" == "$m" ]] && return 1
    done
    return 0
}

capture_initial_pose() {
    # snapshot current arm pose; every send_home reads it back
    if [[ "$SKIP_HOMING" == "1" ]]; then
        echo "[init] SKIP_HOMING=1, not capturing start pose." | tee -a "$LOG_FILE"
        return 0
    fi
    echo "[init] capturing current robot pose as 'initial' ..." | tee -a "$LOG_FILE"
    python - "$ARM_IP" "$INIT_POSE_FILE" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, sys
from xarm.wrapper import XArmAPI

ip, out_path = sys.argv[1], sys.argv[2]
arm = XArmAPI(ip, baud_checkset=False)
arm.clean_error(); arm.clean_warn()
arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)

code, pose = arm.get_position()
if code != 0 or pose is None:
    print(f"[init][py] get_position failed: code={code}")
    sys.exit(2)
x, y, z, roll, pitch, yaw = pose[:6]
with open(out_path, "w") as f:
    json.dump(dict(x=float(x), y=float(y), z=float(z),
                   roll=float(roll), pitch=float(pitch), yaw=float(yaw)),
              f, indent=2)
print(f"[init][py] captured x={x:.1f} y={y:.1f} z={z:.1f} "
      f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} mm/deg")
PY
    if [[ ! -s "$INIT_POSE_FILE" ]]; then
        echo "[init] ERROR: failed to capture initial pose." >&2
        return 1
    fi
    echo "[init] saved -> $INIT_POSE_FILE" | tee -a "$LOG_FILE"
}

send_home() {
    if [[ "$SKIP_HOMING" == "1" ]]; then
        echo "[home] SKIP_HOMING=1, not returning arm to home." | tee -a "$LOG_FILE"
        return 0
    fi
    if [[ ! -s "$INIT_POSE_FILE" ]]; then
        echo "[home] WARNING: no $INIT_POSE_FILE; skipping return." | tee -a "$LOG_FILE"
        return 1
    fi
    echo "[home] returning arm to initial pose (slow: speed=${HOME_SPEED} mvacc=${HOME_MVACC}) ..." \
        | tee -a "$LOG_FILE"
    python - "$ARM_IP" "$INIT_POSE_FILE" "$HOME_SPEED" "$HOME_MVACC" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, sys
from xarm.wrapper import XArmAPI

ip, pose_path, speed_s, mvacc_s = sys.argv[1:5]
speed = float(speed_s); mvacc = float(mvacc_s)
pose = json.load(open(pose_path))

arm = XArmAPI(ip, baud_checkset=False)
arm.clean_error(); arm.clean_warn()
arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)

print(f"[home][py] target x={pose['x']:.1f} y={pose['y']:.1f} z={pose['z']:.1f} "
      f"roll={pose['roll']:.1f} pitch={pose['pitch']:.1f} yaw={pose['yaw']:.1f}  "
      f"speed={speed} mvacc={mvacc}")
ret = arm.set_position(
    x=pose["x"], y=pose["y"], z=pose["z"],
    roll=pose["roll"], pitch=pose["pitch"], yaw=pose["yaw"],
    speed=speed, mvacc=mvacc, wait=True,
)
if ret == 0:
    print("[home][py] reached initial pose.")
    sys.exit(0)
print(f"[home][py] set_position returned {ret}")
sys.exit(3)
PY
    local rc=${PIPESTATUS[0]}
    if [[ $rc -eq 0 ]]; then
        echo "[home] ok." | tee -a "$LOG_FILE"
        return 0
    fi
    echo "[home] WARNING: return-to-initial did not succeed (rc=$rc)." >&2
    return 1
}

run_experiment() {
    # $1 = suffix written into the output CSV filename (ekf|fp|pbvs)
    # $2 = --mode value passed to dinov2_servo.py (ekf|foundationpose|ibvs)
    local suffix="$1"; local mode="$2"
    local out_csv="$OUT_DIR/${REFBASE}_${suffix}.csv"

    echo "" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    echo "[trial] ${REFBASE}_${suffix}   (mode=${mode})" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"

    local -a py_args=(
        FoundationModel/dinov2_servo.py
        --reference "$REFERENCE"
        --out-prefix "${REFBASE}_${suffix}"
        --mode "$mode"
        --cam-to-robot "$CAM_TO_ROBOT"
        --z-floor-mm "$Z_FLOOR_MM"
        --auto-exit-converge-sec "$AUTO_EXIT_CONVERGE_SEC"
        --auto-exit-lost-sec "$AUTO_EXIT_LOST_SEC"
        --auto-exit-max-sec "$AUTO_EXIT_MAX_SEC"
        --stop-depth-m "$STOP_DEPTH_M"
        --run-tag "${REFBASE}_${suffix}"
    )

    if [[ -f "experiments/depth_scale.json" ]]; then
        local saved_scale
        saved_scale="$(python - <<PY
import json
try:
    print(json.load(open("experiments/depth_scale.json"))["scale"])
except Exception:
    pass
PY
)"
        if [[ -n "$saved_scale" ]]; then
            py_args+=( --depth-scale "$saved_scale" )
        fi
    fi

    if [[ "$mode" == "foundationpose" ]]; then
        local fp_extents="$FP_BOX"
        if [[ -z "$fp_extents" ]]; then
            fp_extents="$FP_BOX_FALLBACK"
            echo "[trial] WARN: no FP_BOX known for '$REFBASE'. Using " \
                 "fallback extents ($fp_extents m). Override with " \
                 "FP_BOX=\"W H D\" for measured dimensions." \
                 | tee -a "$LOG_FILE"
        fi
        # shellcheck disable=SC2206
        local fp_arr=( $fp_extents )
        py_args+=( --fp-box "${fp_arr[@]}" )
    fi


    echo "[trial] \$ python ${py_args[*]}" | tee -a "$LOG_FILE"
    echo "[trial] hard-timeout: ${HARD_TIMEOUT_SEC} s" | tee -a "$LOG_FILE"

    # so we can identify the new metrics_*.csv produced by this trial
    local before_list
    before_list="$(ls -1 metrics_*.csv 2>/dev/null || true)"

    set +e
    timeout --signal=TERM --kill-after=10 "${HARD_TIMEOUT_SEC}" \
        python "${py_args[@]}" 2>&1 | tee -a "$LOG_FILE"
    local rc=${PIPESTATUS[0]}
    set -e

    case "$rc" in
        0)   echo "[trial] exit 0 (clean)" | tee -a "$LOG_FILE" ;;
        124) echo "[trial] exit 124 (hard-timeout; Python did not shut down in time)" | tee -a "$LOG_FILE" ;;
        137) echo "[trial] exit 137 (hard-timeout KILL)" | tee -a "$LOG_FILE" ;;
        *)   echo "[trial] exit $rc (see $LOG_FILE)" | tee -a "$LOG_FILE" ;;
    esac

    local new_csv=""
    for f in metrics_*.csv; do
        [[ -f "$f" ]] || continue
        if ! grep -qxF "$f" <<<"$before_list"; then
            if [[ -z "$new_csv" || "$f" -nt "$new_csv" ]]; then
                new_csv="$f"
            fi
        fi
    done
    if [[ -z "$new_csv" ]]; then
        echo "[trial] WARNING: no new metrics_*.csv produced." | tee -a "$LOG_FILE"
        return 0
    fi
    mv -- "$new_csv" "$out_csv"
    echo "[trial] CSV -> $out_csv" | tee -a "$LOG_FILE"
}

capture_initial_pose || {
    echo "[fatal] cannot capture initial pose; aborting batch." >&2
    exit 1
}

MODES_TO_RUN=( "ekf:ekf" "fp:foundationpose" "pbvs:ibvs" )

for spec in "${MODES_TO_RUN[@]}"; do
    suffix="${spec%%:*}"
    mode="${spec##*:}"
    if ! has_mode "$suffix"; then
        echo "[skip] suffix=$suffix (in SKIP_MODES)"
        continue
    fi
    run_experiment "$suffix" "$mode"
    send_home || echo "[home] continuing despite homing failure" >&2
done

echo ""
echo "===================================================================="
echo "[done] All trials complete. Output directory: $OUT_DIR"
echo "       CSVs:"
ls -1 "$OUT_DIR"/*.csv 2>/dev/null || echo "       (none produced)"
echo "===================================================================="
