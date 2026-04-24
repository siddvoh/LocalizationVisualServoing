#!/usr/bin/env python3
# summarizes metrics CSVs: final err, convergence iterations, trajectory, timing
import argparse
import csv
import os
from pathlib import Path
from statistics import median


DEFAULT_TOLS_CM = "0.5,1.0,2.0"

# ZED Mini left @ 1280x720
IMG_CX = 640.0
IMG_CY = 360.0

# commanded-delta magnitude below this = warm-up, not servo
SERVO_ACTIVE_EPS_MM = 0.01


def _f(row, key, default=None):
    val = row.get(key, "")
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _parse_tols(spec: str) -> list:
    try:
        vals = sorted({float(t.strip()) for t in spec.split(",") if t.strip()})
    except ValueError as exc:
        raise SystemExit(f"Invalid --tols-cm value {spec!r}: {exc}")
    if not vals:
        raise SystemExit("--tols-cm produced an empty list")
    if any(v <= 0 for v in vals):
        raise SystemExit("--tols-cm values must all be positive (cm)")
    return vals


def analyze_one(path: Path, tols_m: list) -> dict:
    """parse one CSV; tols_m in METRES"""
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {"file": path.name, "error": "empty"}

    frames = len(rows)
    pipeline = rows[0].get("pipeline") or ""
    run_tag = rows[0].get("run_tag") or ""

    # first frame with a real servo command; skip warm-up zeros
    first_active_idx = None
    for idx, r in enumerate(rows):
        dx = _f(r, "servo_dx_mm") or 0.0
        dy = _f(r, "servo_dy_mm") or 0.0
        dz = _f(r, "servo_dz_mm") or 0.0
        if max(abs(dx), abs(dy), abs(dz)) > SERVO_ACTIVE_EPS_MM:
            first_active_idx = idx
            break

    # ibvs doesn't populate err_m (it servos in image space)
    is_ibvs = pipeline == "ibvs"

    final_err_m = None
    if not is_ibvs and first_active_idx is not None:
        for r in reversed(rows[first_active_idx:]):
            e = _f(r, "err_m")
            if e is not None and e > 0:
                final_err_m = e
                break

    iters_to_tol = {t: None for t in tols_m}
    if not is_ibvs and first_active_idx is not None:
        for offset, r in enumerate(rows[first_active_idx:], 1):
            e = _f(r, "err_m")
            if e is None or e <= 0:
                continue
            for t in tols_m:
                if iters_to_tol[t] is None and e <= t:
                    iters_to_tol[t] = offset

    # ibvs-only: last-centroid distance from image centre
    final_px_err = None
    if is_ibvs and first_active_idx is not None:
        for r in reversed(rows[first_active_idx:]):
            u, v = _f(r, "centroid_u"), _f(r, "centroid_v")
            if u is not None and v is not None:
                final_px_err = ((u - IMG_CX) ** 2 + (v - IMG_CY) ** 2) ** 0.5
                break

    fk = []
    for r in rows:
        x = _f(r, "robot_x_mm")
        y = _f(r, "robot_y_mm")
        z = _f(r, "robot_z_mm")
        if x is not None and y is not None and z is not None:
            fk.append((x, y, z))
    traj_len_mm = 0.0
    for i in range(1, len(fk)):
        dx = fk[i][0] - fk[i - 1][0]
        dy = fk[i][1] - fk[i - 1][1]
        dz = fk[i][2] - fk[i - 1][2]
        traj_len_mm += (dx * dx + dy * dy + dz * dz) ** 0.5
    straight_mm = None
    path_eff = None
    if len(fk) >= 2:
        sx = fk[-1][0] - fk[0][0]
        sy = fk[-1][1] - fk[0][1]
        sz = fk[-1][2] - fk[0][2]
        straight_mm = (sx * sx + sy * sy + sz * sz) ** 0.5
        if traj_len_mm > 1e-6:
            path_eff = straight_mm / traj_len_mm

    iter_times = [_f(r, "iter_time_ms") for r in rows]
    iter_times = [t for t in iter_times if t is not None]
    median_iter_ms = median(iter_times) if iter_times else None

    ts = [_f(r, "timestamp") for r in rows]
    ts = [t for t in ts if t is not None]
    duration_s = (ts[-1] - ts[0]) if len(ts) >= 2 else None

    # FP jitter over last 2s (raw and EKF-smoothed) to show filter damping
    fp_raw_step_std_mm = None
    fp_ekf_step_std_mm = None
    if pipeline == "foundationpose" and ts and len(ts) >= 2:
        import statistics as st
        t_end = ts[-1]
        tail = [r for r in rows
                if _f(r, "timestamp") is not None
                and _f(r, "timestamp") >= t_end - 2.0]

        def _step_std_mm(points):
            if len(points) < 3:
                return None
            diffs = []
            for i in range(1, len(points)):
                dx = points[i][0] - points[i - 1][0]
                dy = points[i][1] - points[i - 1][1]
                dz = points[i][2] - points[i - 1][2]
                diffs.append((dx * dx + dy * dy + dz * dz) ** 0.5)
            return st.pstdev(diffs) * 1000.0

        def _xyz(rs, kx, ky, kz):
            out = []
            for r in rs:
                x, y, z = _f(r, kx), _f(r, ky), _f(r, kz)
                if x is not None and y is not None and z is not None:
                    out.append((x, y, z))
            return out

        fp_raw_step_std_mm = _step_std_mm(_xyz(tail, "fp_raw_x", "fp_raw_y", "fp_raw_z"))
        fp_ekf_step_std_mm = _step_std_mm(_xyz(tail, "ekf_x", "ekf_y", "ekf_z"))

    out = {
        "file": path.name,
        "pipeline": pipeline,
        "run_tag": run_tag,
        "frames": frames,
        "duration_s": duration_s,
        "final_err_cm": (final_err_m * 100.0) if final_err_m is not None else None,
        "final_px_err": final_px_err,
        "traj_len_mm": traj_len_mm if fk else None,
        "straight_mm": straight_mm,
        "path_eff": path_eff,
        "median_iter_ms": median_iter_ms,
        "fp_raw_step_std_mm": fp_raw_step_std_mm,
        "fp_ekf_step_std_mm": fp_ekf_step_std_mm,
    }
    for t_m in tols_m:
        t_cm = t_m * 100.0
        key = f"iter_at_{_tol_key(t_cm)}cm"
        out[key] = iters_to_tol[t_m]
    return out


def _tol_key(t_cm: float) -> str:
    if abs(t_cm - round(t_cm)) < 1e-6:
        return f"{int(round(t_cm))}"
    return f"{t_cm:g}".replace(".", "p")


def fmt(val, spec):
    return f"{val:{spec}}" if val is not None else "-"


def print_table(rows, tols_cm):
    fixed_headers = [
        ("run_tag", 28), ("pipeline", 14), ("frames", 6), ("dur_s", 7),
        ("err_cm", 7), ("px_err", 7),
    ]
    tol_headers = [(f"it@{_tol_key(t)}cm", 9) for t in tols_cm]
    trailing_headers = [
        ("traj_mm", 8), ("straight_mm", 11), ("path_eff", 8),
        ("iter_ms", 8), ("fp_raw_mm", 9), ("fp_ekf_mm", 9),
    ]
    headers = fixed_headers + tol_headers + trailing_headers

    line = " ".join(f"{h:<{w}}" for h, w in headers)
    print(line)
    print("-" * len(line))

    for r in rows:
        cells = [
            f"{r.get('run_tag', '')[:28]:<28}",
            f"{r.get('pipeline', ''):<14}",
            f"{r.get('frames', 0):<6}",
            f"{fmt(r.get('duration_s'), '.1f'):<7}",
            f"{fmt(r.get('final_err_cm'), '.2f'):<7}",
            f"{fmt(r.get('final_px_err'), '.1f'):<7}",
        ]
        for t_cm in tols_cm:
            key = f"iter_at_{_tol_key(t_cm)}cm"
            cells.append(f"{fmt(r.get(key), 'd'):<9}")
        cells.extend([
            f"{fmt(r.get('traj_len_mm'), '.1f'):<8}",
            f"{fmt(r.get('straight_mm'), '.1f'):<11}",
            f"{fmt(r.get('path_eff'), '.3f'):<8}",
            f"{fmt(r.get('median_iter_ms'), '.1f'):<8}",
            f"{fmt(r.get('fp_raw_step_std_mm'), '.2f'):<9}",
            f"{fmt(r.get('fp_ekf_step_std_mm'), '.2f'):<9}",
        ])
        print(" ".join(cells))


def print_aggregate(rows, tols_cm):
    """pass-rate per pipeline at each tolerance"""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("pipeline") or "(unknown)"].append(r)

    print()
    print("Aggregate pass-rate by pipeline:")
    headers = ["pipeline", "n"]
    for t_cm in tols_cm:
        headers.append(f"<={t_cm}cm")
    headers += ["med_err_cm", "med_path_eff"]
    print("  " + "  ".join(f"{h:<12}" for h in headers))
    print("  " + "-" * (14 * len(headers)))
    for pipe, grp in sorted(groups.items()):
        n = len(grp)
        row = [f"{pipe:<12}", f"{n:<12}"]
        # IBVS runs don't populate cm-space error, so the pass-rate columns
        # stay blank rather than reporting a misleading 0/0.
        cm_pool = [r for r in grp if r.get("final_err_cm") is not None]
        for t_cm in tols_cm:
            key = f"iter_at_{_tol_key(t_cm)}cm"
            if not cm_pool:
                row.append(f"{'n/a':<12}")
                continue
            passed = sum(1 for r in cm_pool if r.get(key) is not None)
            row.append(f"{passed/len(cm_pool)*100:5.1f}%      ")
        errs = [r.get("final_err_cm") for r in cm_pool]
        effs = [r.get("path_eff") for r in grp
                if r.get("path_eff") is not None]
        row.append(f"{median(errs):<12.2f}" if errs else f"{'n/a':<12}")
        row.append(f"{median(effs):<12.3f}" if effs else f"{'-':<12}")
        print("  " + "  ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path,
                    help="Directory of metrics CSVs or a single CSV file")
    ap.add_argument("--tols-cm", type=str, default=DEFAULT_TOLS_CM,
                    help="Comma-separated convergence tolerances in cm "
                         f"(default: {DEFAULT_TOLS_CM})")
    ap.add_argument("--tol-cm", type=float, default=None,
                    help="Legacy single-tolerance flag; equivalent to "
                         "--tols-cm VALUE. Kept for backward compatibility.")
    ap.add_argument("--csv", type=Path,
                    help="Also write the summary rows to this CSV")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="Skip the pipeline-level aggregate at the bottom")
    args = ap.parse_args()

    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(args.path.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found under {args.path}")

    # legacy --tol-cm overrides --tols-cm
    tols_spec = (str(args.tol_cm) if args.tol_cm is not None
                 else args.tols_cm)
    tols_cm = _parse_tols(tols_spec)
    tols_m = [t / 100.0 for t in tols_cm]

    rows = [analyze_one(p, tols_m) for p in files]

    print_table(rows, tols_cm)

    if not args.no_aggregate and len(rows) > 1:
        print_aggregate(rows, tols_cm)

    if args.csv:
        base_fields = [
            "file", "pipeline", "run_tag", "frames", "duration_s",
            "final_err_cm", "final_px_err",
        ]
        tol_fields = [f"iter_at_{_tol_key(t)}cm" for t in tols_cm]
        trail_fields = [
            "traj_len_mm", "straight_mm", "path_eff",
            "median_iter_ms", "fp_raw_step_std_mm", "fp_ekf_step_std_mm",
        ]
        fieldnames = base_fields + tol_fields + trail_fields
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"\nSummary written to {args.csv}")


if __name__ == "__main__":
    main()
