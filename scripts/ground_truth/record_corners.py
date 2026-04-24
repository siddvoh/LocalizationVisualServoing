#!/usr/bin/env python3
# jog xArm TCP to each top-face corner, run Horn SVD -> ground-truth pose.
import argparse
import datetime
import json
import pathlib
import sys

import numpy as np
import yaml

from xarm.wrapper import XArmAPI


HERE = pathlib.Path(__file__).resolve().parent
OBJECTS_YAML = HERE / "objects.yaml"
OUTPUT_DIR = HERE / "poses"

DEFAULT_IP = "192.168.1.241"
CORNER_LABELS = ["C1", "C2", "C3", "C4"]
RESIDUAL_WARN_MM = 5.0


def load_objects():
    if not OBJECTS_YAML.exists():
        return {}
    with open(OBJECTS_YAML) as f:
        return yaml.safe_load(f) or {}


def save_objects(db):
    with open(OBJECTS_YAML, "w") as f:
        yaml.safe_dump(db, f, sort_keys=True)


def prompt_dimensions(name):
    print(f"\nObject '{name}' not in {OBJECTS_YAML.name}. Enter its dimensions.")
    print("Top face = whichever face you intend to probe (will be facing up).")
    while True:
        try:
            L = float(input("  Length (mm, long edge of top face): ").strip())
            W = float(input("  Width  (mm, short edge of top face): ").strip())
            H = float(input("  Height (mm, top-face to bottom-face): ").strip())
            if min(L, W, H) <= 0:
                raise ValueError("dimensions must be positive")
            return [L, W, H]
        except ValueError as e:
            print(f"  Invalid input ({e}); try again.")


def body_frame_corners(dims_mm):
    # origin = centroid, X=length, Y=width, Z=height; CW from (+L/2,+W/2,+H/2)
    L, W, H = dims_mm
    return np.array([
        [+L/2, +W/2, +H/2],   # C1
        [+L/2, -W/2, +H/2],   # C2
        [-L/2, -W/2, +H/2],   # C3
        [-L/2, +W/2, +H/2],   # C4
    ], dtype=float)


def horn_rigid_fit(body_pts, base_pts):
    """R, t s.t. base = R @ body + t"""
    B = np.asarray(body_pts, dtype=float)
    A = np.asarray(base_pts, dtype=float)
    cB, cA = B.mean(axis=0), A.mean(axis=0)
    H = (B - cB).T @ (A - cA)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cA - R @ cB
    residuals = np.linalg.norm((R @ B.T).T + t - A, axis=1)
    return R, t, residuals


def connect_arm(ip):
    print(f"\nConnecting to xArm at {ip} ...", flush=True)
    arm = XArmAPI(ip, baud_checkset=False)
    code, ver = arm.get_version()
    if code != 0:
        raise RuntimeError(f"xArm get_version failed, code={code}")
    print(f"  connected (firmware: {ver.strip() if isinstance(ver, str) else ver})")
    return arm


def read_tcp(arm):
    code, pose = arm.get_position(is_radian=True)
    if code != 0:
        raise RuntimeError(f"get_position failed, code={code}")
    return np.array(pose[:3], dtype=float), np.array(pose[3:], dtype=float)


def register_corners(arm):
    print()
    print(" CORNER REGISTRATION ".center(60, "="))
    print("For each corner:")
    print("  - In xArm Studio, jog the TCP to hover ~1 mm above the corner.")
    print("  - Keep the arm in the SAME ORIENTATION across all 4 corners")
    print("    (tool pointing straight down is ideal).")
    print("  - Press ENTER here to record. Type 'r' + ENTER to redo last.")
    print()
    base, rpys = [], []
    i = 0
    while i < 4:
        label = CORNER_LABELS[i]
        resp = input(f"  [{label}] Jog to corner {i+1}/4, then ENTER "
                     f"(or 'r' to redo): ").strip().lower()
        if resp == "r":
            if i == 0:
                print("    nothing to redo yet.")
                continue
            i -= 1
            base.pop(); rpys.pop()
            print(f"    redoing {CORNER_LABELS[i]}")
            continue
        pos, rpy = read_tcp(arm)
        base.append(pos); rpys.append(rpy)
        print(f"    recorded: [{pos[0]:8.2f}, {pos[1]:8.2f}, {pos[2]:8.2f}] mm")
        i += 1
    return np.array(base), np.array(rpys)


def report_fit(dims_mm, base_corners, residuals):
    L, W, _ = dims_mm
    print()
    print(" FIT REPORT ".center(60, "="))
    # sanity-check edges as multiset
    edges = [np.linalg.norm(base_corners[(k+1) % 4] - base_corners[k])
             for k in range(4)]
    nominal_multiset = sorted([L, L, W, W])
    measured_sorted = sorted(edges)
    print("Measured edges (mm, sorted): "
          + ", ".join(f"{e:.2f}" for e in measured_sorted))
    print(f"Nominal  edges (mm, sorted): "
          + ", ".join(f"{n:.2f}" for n in nominal_multiset))
    edge_err = max(abs(m - n) for m, n in zip(measured_sorted, nominal_multiset))
    print(f"Max edge-length error: {edge_err:.2f} mm")
    print()
    print("Per-corner fit residual (mm): "
          + ", ".join(f"{r:.2f}" for r in residuals))
    rms = float(np.sqrt(np.mean(residuals**2)))
    print(f"RMS residual: {rms:.2f} mm")
    return rms, edge_err


def save_output(obj_name, dims_mm, R, t, body_corners, base_corners,
                residuals, rpys):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # t IS the centroid since body frame origin = centroid by construction
    centroid = t
    top_face_centroid = base_corners.mean(axis=0)
    payload = {
        "object": obj_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "dimensions_mm": list(dims_mm),
        "pose_base_frame": {
            "translation_mm": t.tolist(),
            "rotation_matrix": R.tolist(),
        },
        "centroid_base_mm": centroid.tolist(),
        "top_face_centroid_base_mm": top_face_centroid.tolist(),
        "body_corners_mm": body_corners.tolist(),
        "base_corners_mm": base_corners.tolist(),
        "tcp_orientations_rpy_rad": rpys.tolist(),
        "fit_residual_per_corner_mm": residuals.tolist(),
        "fit_residual_rms_mm": float(np.sqrt(np.mean(residuals**2))),
        "method": "xarm TCP hover over top-face corners, Horn SVD fit",
        "notes": ("Body frame: origin = centroid, X=length, Y=width, Z=height. "
                  "TCP hovered ~1 mm above each corner (no contact). "
                  "Translation includes a constant TCP-to-sight-point offset "
                  "(absorbed into t as long as arm orientation was held fixed)."),
    }
    out_path = OUTPUT_DIR / f"{obj_name}_{stamp}.json"
    latest = OUTPUT_DIR / f"{obj_name}_latest.json"
    for p in (out_path, latest):
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
    print(f"\nGround truth written to:")
    print(f"  {out_path}")
    print(f"  {latest}  (convenience pointer for analysis scripts)")
    print()
    print(f"Object centroid in base frame: "
          f"[{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}] mm")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object", default=None,
                    help="object name (matches keys in objects.yaml)")
    ap.add_argument("--ip", default=DEFAULT_IP, help=f"xArm IP (default {DEFAULT_IP})")
    args = ap.parse_args()

    db = load_objects()
    name = args.object or input("Object name: ").strip()
    if not name:
        sys.exit("No object name provided; aborting.")

    if name in db and "dimensions_mm" in db[name]:
        dims = db[name]["dimensions_mm"]
        print(f"Loaded dimensions for '{name}': "
              f"{dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm (LxWxH)")
    else:
        dims = prompt_dimensions(name)
        db[name] = {"dimensions_mm": dims}
        save_objects(db)
        print(f"  saved to {OBJECTS_YAML.name}")

    body_corners = body_frame_corners(dims)
    arm = connect_arm(args.ip)

    while True:
        base_corners, rpys = register_corners(arm)
        R, t, residuals = horn_rigid_fit(body_corners, base_corners)
        rms, edge_err = report_fit(dims, base_corners, residuals)

        if rms > RESIDUAL_WARN_MM or edge_err > RESIDUAL_WARN_MM:
            print(f"\n  Residual ({rms:.2f} mm) or edge error ({edge_err:.2f} mm) "
                  f"exceeds {RESIDUAL_WARN_MM} mm.")
            print("Likely: one corner mis-sighted, arm orientation changed mid-"
                  "registration, or wrong dimensions in objects.yaml.")
            if input("Redo all 4 corners? [y/N]: ").strip().lower() == "y":
                continue
        break

    save_output(name, dims, R, t, body_corners, base_corners, residuals, rpys)


if __name__ == "__main__":
    main()
