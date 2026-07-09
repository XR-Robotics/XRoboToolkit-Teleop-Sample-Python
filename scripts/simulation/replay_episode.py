"""Replay a recorded UMI-style handheld episode on the dual-YAM model (kinematic, Placo).

This is the **A-level** replay: drive each arm so its modelled Pico frame
(`*_pico_link`) tracks the recorded controller trajectory, actuate the gripper
fingers from the encoder width, and log per-frame tracking residual. No object /
no contact physics (that is B-level, needs MuJoCo + MJCF + an object).

Design decisions (see why in comments):
  * Pose format in `pico_controllers.npz` is [x,y,z, qx,qy,qz,qw] (xyzw), but
    meshcat.transformations wants [w,x,y,z] -> we reorder. Same convention the
    capture code uses (base_teleop_controller._process_xr_pose).
  * The teleop that produced this rig used *relative* (trigger-gated) control, so
    the recorded absolute Pico-world poses are NOT in the robot frame. We anchor:
    pick an initial joint config q0, and align so at t0 the FK of `*_pico_link`
    equals the first *valid* recorded pose. Thereafter the arm follows the
    controller's relative SE(3) motion exactly. Anchoring at t0 means zero initial
    error and needs no extra calibration. (Chessboard-based scene alignment is a
    documented upgrade hook -- see `--anchor`.)
  * Invalid samples (left_valid/right_valid == 0, e.g. tracking dropout) HOLD the
    last valid target -- never teleport, which would blow up IK.
  * Playback honours recorded timestamps (real-time * 1/speed), so motion speed
    matches the demo.

Usage:
    # 1) sanity-check the data + model first
    python scripts/simulation/replay_episode.py \
        --urdf /path/to/dual_yam/dual_yam.urdf \
        --episode /path/to/session_XXXX/episode_000 --inspect

    # 2) full replay (opens a meshcat tab in your browser)
    python scripts/simulation/replay_episode.py \
        --urdf /path/to/dual_yam/dual_yam.urdf \
        --episode /path/to/session_XXXX/episode_000

    # headless, just print tracking error (no viewer)
    python scripts/simulation/replay_episode.py ... --no-viz

Env (Mac): `uv pip install mujoco placo meshcat`  (placo pulls placo_utils).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


# --- Which arms we can replay and how each maps to the model -----------------
# link      : IK target link = the modelled Pico controller frame on that arm.
# grip_joints: prismatic finger joints to actuate from the encoder width.
# encoder   : encoder npz filename under episode/lowdim/.
ARMS = {
    "left": {
        "pose_key": "left_pose",
        "valid_key": "left_valid",
        "link": "left_pico_link",
        "grip_joints": ["left_joint7", "left_joint8"],
        "encoder": "encoder_left.npz",
    },
    "right": {
        "pose_key": "right_pose",
        "valid_key": "right_valid",
        "link": "right_pico_link",
        "grip_joints": ["right_joint7", "right_joint8"],
        "encoder": "encoder_right.npz",
    },
}

# Gripper finger travel from the URDF (joint7/joint8 prismatic limit).
GRIPPER_FINGER_MAX = 0.0475  # metres, per finger

# A modest non-singular ready pose (rad), applied to jointN of each arm.
# joint2/joint3 have lower limit 0, so keep them positive. Anchoring at t0 makes
# the exact choice non-critical for tracking; adjust if IK struggles / unreachable.
DEFAULT_Q0 = {1: 0.0, 2: 1.2, 3: 1.2, 4: 0.0, 5: 0.0, 6: 0.0}


# --- pose helpers ------------------------------------------------------------
def _quat_xyzw_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit-normalised [x,y,z,w] quaternion -> 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def pose7_to_T(pose7: np.ndarray) -> np.ndarray:
    """[x,y,z, qx,qy,qz,qw] -> 4x4 homogeneous transform."""
    x, y, z, qx, qy, qz, qw = [float(v) for v in pose7[:7]]
    T = np.eye(4)
    T[:3, :3] = _quat_xyzw_to_R(qx, qy, qz, qw)
    T[:3, 3] = [x, y, z]
    return T


def T_pos(T: np.ndarray) -> np.ndarray:
    return T[:3, 3]


def rot_angle_between(Ta: np.ndarray, Tb: np.ndarray) -> float:
    """Geodesic angle (rad) between the rotation parts of two transforms."""
    R = Ta[:3, :3].T @ Tb[:3, :3]
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


# --- data loading ------------------------------------------------------------
def load_pico(episode: Path) -> dict:
    p = episode / "lowdim" / "pico_controllers.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    with np.load(p) as d:
        out = {k: d[k] for k in d.files}
    return out


def load_encoder(episode: Path, fname: str) -> dict | None:
    p = episode / "lowdim" / fname
    if not p.exists():
        return None
    with np.load(p) as d:
        return {k: d[k] for k in d.files}


def encoder_width_series(enc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, width_per_finger[m]).

    Prefer calibrated 'metric_m' (metres, total stroke) when present & finite;
    otherwise linearly normalise 'raw' over the episode's observed range. Either
    way we map to *per-finger* travel in [0, GRIPPER_FINGER_MAX].
    """
    ts = enc["timestamp"].astype(np.float64)
    if "metric_m" in enc and np.isfinite(enc["metric_m"]).any():
        total = np.asarray(enc["metric_m"], dtype=np.float64)
        total = np.nan_to_num(total, nan=0.0)
        # metric_m is total opening; per finger = half, clipped to travel.
        per_finger = np.clip(total * 0.5, 0.0, GRIPPER_FINGER_MAX)
    else:
        raw = np.asarray(enc["raw"], dtype=np.float64)
        valid = raw >= 0
        if valid.sum() < 2:
            per_finger = np.zeros_like(raw)
        else:
            lo, hi = np.percentile(raw[valid], [2, 98])
            span = max(hi - lo, 1e-6)
            norm = np.clip((raw - lo) / span, 0.0, 1.0)
            per_finger = norm * GRIPPER_FINGER_MAX
    return ts, per_finger


def nearest(ts: np.ndarray, t: float) -> int:
    return int(np.searchsorted(ts, t).clip(0, len(ts) - 1))


# --- target construction (anchor + hold-on-dropout) --------------------------
def build_targets(pico: dict, arm_cfg: dict, T_anchor_fk: np.ndarray,
                  base_pos: np.ndarray | None = None, mode: str = "center",
                  reach: float = 0.40):
    """Return (timestamps, [T_world_target...]) for one arm.

    T_world_target(t) = A @ T_picoworld_pico(t). The rotation of A anchors the
    orientation so the FIRST valid frame matches T_anchor_fk (FK of the arm's
    pico_link at q0). The translation of A depends on `mode`:

      * "start"  : also pin the first frame's *position* to FK(q0). Simple, but
                   if q0 sits near the reach boundary the demo can spill out of
                   the workspace (large position residual).
      * "center" : shift so the demo's *centroid* sits `reach` metres from the
                   arm base along its current bearing -- centres the motion in the
                   dexterous workspace. This is the default and usually the
                   difference between ~50 mm and ~1 mm tracking error.

    (A scene-frame anchor from the chessboard start-pose would replace this
    heuristic with a metric ground truth -- future `--anchor scene`.)
    Invalid frames hold the previous valid target (never teleport).
    """
    ts = pico["timestamp"].astype(np.float64)
    poses = np.asarray(pico[arm_cfg["pose_key"]], dtype=np.float64)  # (N,7)
    valid = np.asarray(pico[arm_cfg["valid_key"]]).astype(bool)

    first = np.argmax(valid) if valid.any() else -1
    if first < 0 or not valid[first]:
        return ts, None, 0.0  # this arm has no usable data

    A = T_anchor_fk @ np.linalg.inv(pose7_to_T(poses[first]))

    if mode == "center" and base_pos is not None:
        # centroid of valid poses, mapped to the world under the start-anchor
        cen_pico = poses[valid][:, :3].mean(axis=0)
        cen_world = A[:3, :3] @ cen_pico + A[:3, 3]
        bearing = cen_world - base_pos
        nb = float(np.linalg.norm(bearing))
        if nb > 1e-6:
            target_centroid = base_pos + (bearing / nb) * reach
            A = A.copy()
            A[:3, 3] += target_centroid - cen_world  # pure translation shift

    targets: list[np.ndarray] = []
    last = None
    held = 0
    for i in range(len(ts)):
        if valid[i]:
            last = A @ pose7_to_T(poses[i])
        elif last is None:
            last = T_anchor_fk  # before first valid: sit at anchor
        else:
            held += 1
        targets.append(last)
    valid_frac = float(valid.mean())
    return ts, targets, valid_frac


# --- placo setup -------------------------------------------------------------
def setup_placo(urdf: Path, arms: list[str], q0_scale: float):
    import placo  # imported here so --inspect can run without placo installed

    # Pass the ABSOLUTE urdf path so placo resolves relative mesh paths
    # (assets/*.stl) from the urdf's own directory. ignore_collisions silences
    # the neutral-pose self-collision spam (we do pure kinematic IK).
    robot = placo.RobotWrapper(str(urdf), placo.Flags.ignore_collisions)

    # placo adds a 7-DOF free-flyer (q dim 23 = 7 base + 16 joints). Put it at
    # identity and lock it so the base doesn't float during IK.
    robot.state.q[:7] = [0, 0, 0, 0, 0, 0, 1]

    solver = placo.KinematicsSolver(robot)
    solver.dt = 0.01
    try:
        solver.mask_fbase(True)
    except Exception:
        pass

    # ready pose per arm
    for arm in arms:
        for jn, val in DEFAULT_Q0.items():
            name = f"{arm}_joint{jn}"
            try:
                robot.set_joint(name, val * q0_scale)
            except Exception as e:
                print(f"[warn] could not set {name}: {e}")
    robot.update_kinematics()

    tasks = {}
    for arm in arms:
        link = ARMS[arm]["link"]
        T0 = robot.get_T_world_frame(link)
        task = solver.add_frame_task(link, T0)
        task.configure(f"{arm}_ee", "soft", 1.0)
        m = solver.add_manipulability_task(link, "both", 1.0)
        m.configure(f"{arm}_manip", "soft", 1e-2)
        tasks[arm] = task
    robot.update_kinematics()
    return robot, solver, tasks


# --- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True, help="path to .../episode_XXX")
    ap.add_argument("--urdf", type=Path, required=True, help="path to dual_yam.urdf")
    ap.add_argument("--arms", default="auto", choices=["auto", "left", "right", "both"])
    ap.add_argument("--anchor", default="center", choices=["center", "start"],
                    help="center: put demo centroid --reach m from base (recommended); "
                         "start: pin first frame to FK(q0)")
    ap.add_argument("--reach", type=float, default=0.40,
                    help="target centroid distance from arm base (m), for --anchor center")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1.0=real time)")
    ap.add_argument("--q0-scale", type=float, default=1.0, help="scale the default ready pose")
    ap.add_argument("--valid-arm-frac", type=float, default=0.2,
                    help="in auto mode, keep an arm only if this fraction of samples are valid")
    ap.add_argument("--no-viz", action="store_true", help="headless: no meshcat, just residual stats")
    ap.add_argument("--show-cameras", action="store_true",
                    help="show the camera-link meshes (the big blue frustums); hidden by default")
    ap.add_argument("--inspect", action="store_true", help="print data/model schema and exit")
    ap.add_argument("--log", type=Path, default=None, help="write per-frame residual csv here")
    args = ap.parse_args()

    episode = args.episode.expanduser().resolve()
    urdf = args.urdf.expanduser().resolve()
    pico = load_pico(episode)

    # ---- inspect mode: verify assumptions against the real files -------------
    if args.inspect:
        print(f"[pico] {episode/'lowdim'/'pico_controllers.npz'}")
        for k, v in pico.items():
            v = np.asarray(v)
            extra = ""
            if k in ("left_valid", "right_valid"):
                extra = f"  valid_frac={v.astype(bool).mean():.3f}"
            print(f"  {k:24s} {str(v.shape):12s} {v.dtype}{extra}")
        n = len(pico["timestamp"])
        dur = float(pico["timestamp"][-1] - pico["timestamp"][0]) if n > 1 else 0.0
        print(f"  ~{n} samples, {dur:.2f}s, ~{(n-1)/dur if dur>0 else 0:.1f} Hz")
        for arm, cfg in ARMS.items():
            enc = load_encoder(episode, cfg["encoder"])
            if enc is not None:
                ets, w = encoder_width_series(enc)
                print(f"[enc {arm}] {len(ets)} samples, per-finger width "
                      f"min={w.min()*1000:.1f}mm max={w.max()*1000:.1f}mm")
            else:
                print(f"[enc {arm}] (missing)")
        print("\n[urdf] loading model to list joints/links ...")
        try:
            import placo
            robot = placo.RobotWrapper(str(urdf), placo.Flags.ignore_collisions)
            names = list(robot.model.names)
            print("  joints:", ", ".join(n for n in names if n != "universe"))
            for arm in ARMS:
                link = ARMS[arm]["link"]
                try:
                    print(f"  FK {link}: pos={robot.get_T_world_frame(link)[:3,3]}")
                except Exception as e:
                    print(f"  [warn] link {link} not found: {e}")
        except ImportError:
            print("  placo not installed -> skipping model load "
                  "(uv pip install placo). Data schema above is still valid.")
        return

    # ---- decide which arms to replay ----------------------------------------
    if args.arms == "both":
        arms = ["left", "right"]
    elif args.arms in ("left", "right"):
        arms = [args.arms]
    else:  # auto
        arms = [a for a, c in ARMS.items()
                if np.asarray(pico[c["valid_key"]]).astype(bool).mean() >= args.valid_arm_frac]
        if not arms:
            print("[error] no arm has enough valid samples; check the data or lower --valid-arm-frac")
            sys.exit(1)
    print(f"[replay] arms = {arms}")

    robot, solver, tasks = setup_placo(urdf, arms, args.q0_scale)

    # anchor each arm to its own FK-at-q0
    targets = {}
    tstamps = None
    for arm in arms:
        T_fk = robot.get_T_world_frame(ARMS[arm]["link"])
        base_pos = robot.get_T_world_frame(f"{arm}_base_link")[:3, 3]
        ts, tg, vf = build_targets(pico, ARMS[arm], T_fk, base_pos, args.anchor, args.reach)
        if tg is None:
            print(f"[warn] arm {arm} has no usable pose data, skipping")
            continue
        print(f"[replay] arm {arm}: valid_frac={vf:.3f}")
        targets[arm] = tg
        tstamps = ts
    arms = [a for a in arms if a in targets]
    if not arms:
        sys.exit("[error] nothing to replay")

    # encoder width series per arm (time-aligned by nearest timestamp at run time)
    enc_series = {}
    for arm in arms:
        enc = load_encoder(episode, ARMS[arm]["encoder"])
        enc_series[arm] = encoder_width_series(enc) if enc is not None else None

    # ---- viz ---------------------------------------------------------------
    vis = None
    if not args.no_viz:
        try:
            from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz
            vis = robot_viz(robot)
            if not args.show_cameras:
                # hide the camera-link meshes (big blue frustums) that clutter the view
                import pinocchio as pin
                for go in robot.visual_model.geometryObjects:
                    if "camera" in go.name.lower():
                        node = vis.getViewerNodeName(go, pin.GeometryType.VISUAL)
                        vis.viewer[node].set_property("visible", False)
            import webbrowser
            webbrowser.open(vis.viewer.url())
            time.sleep(1.0)
        except Exception as e:
            print(f"[warn] viz unavailable ({e}); continuing headless")
            vis = None

    # ---- replay loop --------------------------------------------------------
    N = len(tstamps)
    residual_rows = []  # (t, arm, pos_err_mm, rot_err_deg)
    t_wall0 = time.perf_counter()
    t_data0 = float(tstamps[0])

    for i in range(N):
        # set IK targets + gripper
        for arm in arms:
            tasks[arm].T_world_frame = targets[arm][i]
            es = enc_series.get(arm)
            if es is not None:
                ets, w = es
                j = nearest(ets, float(tstamps[i]))
                for gj in ARMS[arm]["grip_joints"]:
                    try:
                        robot.set_joint(gj, float(w[j]))
                    except Exception:
                        pass
        try:
            solver.solve(True)
            robot.update_kinematics()
        except RuntimeError as e:
            print(f"[ik] frame {i} solve failed: {e}")

        # residual: commanded target vs achieved FK
        for arm in arms:
            T_fk = robot.get_T_world_frame(ARMS[arm]["link"])
            pos_err = np.linalg.norm(T_pos(targets[arm][i]) - T_pos(T_fk)) * 1000.0
            rot_err = np.degrees(rot_angle_between(targets[arm][i], T_fk))
            residual_rows.append((float(tstamps[i]), arm, pos_err, rot_err))

        if vis is not None:
            vis.display(robot.state.q)

        # real-time pacing
        if i + 1 < N:
            dt_data = (float(tstamps[i + 1]) - float(tstamps[i])) / max(args.speed, 1e-6)
            dt_data = min(dt_data, 0.2)  # cap long gaps (dropouts) so we don't stall
            target_wall = t_wall0 + ((float(tstamps[i + 1]) - t_data0) / max(args.speed, 1e-6))
            sleep = min(dt_data, target_wall - time.perf_counter())
            if sleep > 0:
                time.sleep(sleep)

    # ---- residual summary ---------------------------------------------------
    print("\n=== tracking residual (commanded pico_link target vs achieved FK) ===")
    rr = np.array([(r[2], r[3]) for r in residual_rows])
    for arm in arms:
        mask = np.array([r[1] == arm for r in residual_rows])
        if mask.any():
            a = rr[mask]
            print(f"  {arm:5s}  pos: mean={a[:,0].mean():6.2f}mm  p95={np.percentile(a[:,0],95):6.2f}mm  "
                  f"max={a[:,0].max():6.2f}mm   |  rot: mean={a[:,1].mean():5.2f}deg  "
                  f"max={a[:,1].max():5.2f}deg")
    print("Rule of thumb: free-space tracking should be sub-mm / sub-deg. Large/spiky "
          "residual => unreachable poses, joint limits (wrist ±90°), or a frame bug.")

    if args.log is not None:
        import csv
        with open(args.log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "arm", "pos_err_mm", "rot_err_deg"])
            w.writerows(residual_rows)
        print(f"[log] wrote {args.log}")


if __name__ == "__main__":
    main()
