"""Phase 4: solve X_C^G from collected samples (the "black box" step).

For every sample i we have X_G^B_i (Pico) and X_T^C_i (solvePnP on the chessboard).
Hand-eye calibration finds the *fixed* X_C^G such that, for any pair (i, j):

    (X_G^B_j)^-1 @ X_G^B_i  @  X_C^G  =  X_C^G  @  X_T^C_j @ (X_T^C_i)^-1
        \\_____________   ___________/         \\________________   ____________/
                         A                                          B
    i.e. the classic AX = XB form solved by cv2.calibrateHandEye, where
        A = relative gripper motion, X = X_C^G (unknown), B = relative camera observation.

We:
  1. re-run solvePnP to get each X_T^C (so the solver and the validator share one source
     of truth and you can change intrinsics without re-collecting data);
  2. convert each raw Pico pose to X_G^B using the dataset's axis_remap;
  3. run all five OpenCV methods and report their spread so you can pick the most stable;
  4. warn about poor *rotational* excitation (Tsai/Park/Horaud are ill-conditioned without
     at least ~30 deg between consecutive gripper orientations);
  5. save handeye_result.json (the chosen method) plus a per-method table on stdout.

Run example
    python scripts/calibration/handeye_solve.py \\
        --record-dir calibration_data/exp01 --method PARK
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from handeye_common import (
    detect_chessboard,
    list_samples,
    load_sample,
    make_object_points,
    pico_pose_to_matrix,
    read_dataset_config,
    rotmat_distance_deg,
    rotmat_to_quat_xyzw,
    save_handeye_result,
    solve_target_to_camera,
)

# All methods exposed by cv2.calibrateHandEye.
METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve X_C^G (eye-in-hand) from collected samples.")
    p.add_argument("--record-dir", type=Path, required=True)
    p.add_argument("--method", default="PARK", choices=list(METHODS.keys()), help="which result to persist.")
    p.add_argument(
        "--min-rot-deg",
        type=float,
        default=30.0,
        help="below this mean pairwise gripper rotation, the solver is poorly conditioned (warn only).",
    )
    return p.parse_args()


def build_pair_lists(record_dir: Path):
    """Re-derive (X_G^B_i, X_T^C_i) for every sample. Skips samples where the board is not found."""
    cfg = read_dataset_config(record_dir)
    objp = make_object_points(cfg["square_size_m"], cfg["pattern_size"])
    K, dist = cfg["camera_matrix"], cfg["dist_coeffs"]
    axis_remap = cfg["axis_remap"]

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    used = []
    for jp in list_samples(record_dir):
        s = load_sample(jp)
        img = cv2.imread(str(record_dir / s["image"]), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] cannot read image {s['image']}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = detect_chessboard(gray, cfg["pattern_size"])
        if not ok:
            print(f"  [skip] board not detected in {s['image']} (was it blurred during capture?)")
            continue
        T_t2c = solve_target_to_camera(corners, objp, K, dist)
        if T_t2c is None:
            continue
        T_g2b = pico_pose_to_matrix(s["pico_pose_xyzq"], axis_remap=axis_remap)
        R_g2b.append(T_g2b[:3, :3])
        t_g2b.append(T_g2b[:3, 3])
        R_t2c.append(T_t2c[:3, :3])
        t_t2c.append(T_t2c[:3, 3])
        used.append(jp.stem)

    return cfg, (R_g2b, t_g2b, R_t2c, t_t2c), used


def solve_all(R_g2b, t_g2b, R_t2c, t_t2c) -> dict[str, np.ndarray]:
    """Run every method; returns {method_name: 4x4 T_cam2gripper}."""
    results = {}
    for name, flag in METHODS.items():
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(
                R_gripper2base=R_g2b,
                t_gripper2base=t_g2b,
                R_target2cam=R_t2c,
                t_target2cam=t_t2c,
                method=flag,
            )
        except cv2.error as exc:
            print(f"  [skip] {name} failed: {exc.err}")
            continue
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_c2g
        T[:3, 3] = t_c2g.ravel()
        results[name] = T
    return results


def pairwise_rotation_stats(R_g2b) -> tuple[float, float, float]:
    """Mean / min / max of pairwise gripper-rotation magnitudes (degrees)."""
    angles = []
    n = len(R_g2b)
    for i in range(n):
        for j in range(i + 1, n):
            angles.append(rotmat_distance_deg(R_g2b[i], R_g2b[j]))
    if not angles:
        return 0.0, 0.0, 0.0
    return float(np.mean(angles)), float(np.min(angles)), float(np.max(angles))


def translation_range(t_g2b) -> float:
    """Largest axis-aligned translation span across gripper poses, in metres."""
    if not t_g2b:
        return 0.0
    ts = np.asarray(t_g2b, dtype=np.float64).reshape(len(t_g2b), 3)
    return float(np.linalg.norm(ts.max(axis=0) - ts.min(axis=0)))


def method_spread(per_method: dict[str, np.ndarray]) -> tuple[float, float]:
    """Translation (m) and rotation (deg) spread across methods -- small => robust data."""
    if len(per_method) < 2:
        return 0.0, 0.0
    ts = np.array([T[:3, 3] for T in per_method.values()])
    t_spread = float(np.linalg.norm(ts.max(axis=0) - ts.min(axis=0)))
    base = list(per_method.values())[0][:3, :3]
    rot_spread = max(rotmat_distance_deg(base, T[:3, :3]) for T in per_method.values())
    return t_spread, rot_spread


def main() -> None:
    args = parse_args()
    cfg, (R_g2b, t_g2b, R_t2c, t_t2c), used = build_pair_lists(args.record_dir)
    n = len(R_g2b)
    if n < 3:
        raise SystemExit(f"Need >= 3 valid samples to solve, got {n}.")

    mean_rot, min_rot, max_rot = pairwise_rotation_stats(R_g2b)
    pos_span = translation_range(t_g2b)

    print(f"\n=== {args.record_dir} : {n} valid samples ({len(used)} total) ===")
    print(f"Gripper pairwise rotation  mean={mean_rot:6.1f} deg  min={min_rot:6.1f}  max={max_rot:6.1f}")
    print(f"Gripper translation span   {pos_span*1000:6.1f} mm")
    if max_rot < 1.0 and pos_span < 0.001:
        raise SystemExit(
            "\nNo usable gripper motion was recorded: all Pico poses are effectively identical.\n"
            "Re-capture the dataset while moving/rotating the controller-mounted camera, and first\n"
            "verify that XrClient.get_pose_by_name(...) changes when you move the controller."
        )
    if mean_rot < args.min_rot_deg:
        print(f"  [warn] mean rotation < {args.min_rot_deg:.0f} deg -- add larger roll/pitch/yaw")
        print("         variations; rotational DOF is what constrains X_C^G's orientation.")

    per_method = solve_all(R_g2b, t_g2b, R_t2c, t_t2c)
    if not per_method:
        raise SystemExit("All OpenCV hand-eye methods failed. Re-capture with larger, varied rotations.")
    if args.method not in per_method:
        ok = ", ".join(per_method)
        raise SystemExit(f"Requested method {args.method} failed. Successful method(s): {ok}")
    t_spread, rot_spread = method_spread(per_method)

    print(f"Cross-method spread       t={t_spread*1000:6.2f} mm   R={rot_spread:5.2f} deg")
    if t_spread > 0.020 or rot_spread > 5.0:
        print("  [warn] methods disagree a lot -- data is noisy / poorly excited; trust validation over any single method.")

    print("\n--- X_C^G per method (cam optical frame -> gripper) ---")
    print(f"{'method':<11}{'t [m]':<34}{'|t| [mm]':<10}{'quat xyzw'}")
    chosen = per_method[args.method]
    for name, T in per_method.items():
        t = T[:3, 3]
        q = rotmat_to_quat_xyzw(T[:3, :3])
        mark = "  <-- saved" if name == args.method else ""
        print(f"{name:<11}{str(np.round(t, 5)):<34}{np.linalg.norm(t)*1000:<10.2f}{np.round(q, 5)}{mark}")

    diagnostics = {
        "n_samples": n,
        "used_samples": used,
        "gripper_rot_deg": {"mean": mean_rot, "min": min_rot, "max": max_rot},
        "method_spread": {"t_mm": t_spread * 1000, "rot_deg": rot_spread},
        "all_methods": {k: v.tolist() for k, v in per_method.items()},
    }
    out = save_handeye_result(args.record_dir / "handeye_result.json", args.method, chosen, diagnostics)
    print(f"\nSaved {args.method} -> {out}")
    print("Next: run handeye_validate.py to cross-check.")


if __name__ == "__main__":
    main()
