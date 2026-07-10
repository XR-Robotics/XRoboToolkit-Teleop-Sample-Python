"""Overlay the per-frame camera ground-truth trajectory vs the Pico trajectory.

Requires the wrist camera to see the AprilTag on (ideally) EVERY frame. For each
camera frame we solve the camera pose in the tag (scene) frame by PnP -- this is
the drift-free ground truth (RED). We time-match the Pico sample and express its
camera-frame estimate in the same tag frame (BLUE). Plots both 3D tracks plus
the per-frame position/rotation error over time.

Alignment (--align):
  anchor  : rigidly align the two worlds using the FIRST frame, then watch them
            diverge. Error(t) = accumulated Pico drift from t0. (default)
  umeyama : best-fit rigid alignment over all frames (Kabsch). Error(t) = the
            intrinsic residual with the t0 bias removed.

Only joint_pico (camera_link->pico) from the URDF is used, so this is invariant
to the gripper/camera mount correction.

Usage (new AprilTag data):
    python scripts/calibration/aruco_pico_trajectory.py \
        --episode /path/session_XXXX/episode_000 --urdf /path/dual_yam.urdf
    # older ArUco data: add --tag-family aruco4x4

Env: opencv-contrib-python, numpy, matplotlib (sim/vision venv).
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aruco_gripper_check as agc  # noqa: E402


def tag_candidates(img, K, dist, dict_id, tag_id, marker_size,
                   max_reproj=4.0, max_dist=2.0):
    """Return (marker_id, [T_optcam_tag, ...]) — up to the 2 IPPE_SQUARE
    solutions of the planar-square ambiguity, gated on reprojection + distance.
    The caller disambiguates by temporal continuity. None if no valid tag."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(img)
    if ids is None or len(ids) == 0:
        return None
    ids = ids.flatten().tolist()
    if tag_id is None:
        k = int(np.argmax([cv2.contourArea(corners[j][0]) for j in range(len(ids))]))
    elif tag_id in ids:
        k = ids.index(tag_id)
    else:
        return None
    c = corners[k][0]
    h = marker_size / 2
    objp = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
    try:
        n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
            objp, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return None
    reproj = np.asarray(reproj, float).flatten()
    cands = []
    for j in range(n):
        tvec = np.asarray(tvecs[j], float).flatten()
        if reproj[j] > max_reproj or np.linalg.norm(tvec) > max_dist:
            continue
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(np.asarray(rvecs[j], float))[0]
        T[:3, 3] = tvec
        cands.append((T, float(reproj[j])))
    if not cands:
        return None
    return ids[k], cands


def nearest_idx(sorted_ts: np.ndarray, t: float) -> int:
    j = int(np.searchsorted(sorted_ts, t))
    if j <= 0:
        return 0
    if j >= len(sorted_ts):
        return len(sorted_ts) - 1
    return j - 1 if abs(sorted_ts[j - 1] - t) <= abs(sorted_ts[j] - t) else j


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Best-fit rigid R,t mapping P onto Q (minimise ||Q - (R@P + t)||)."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cq - R @ cp


def collect_arm(episode: Path, root, arm: str, dict_id, tag_id, marker_size, stride: int):
    """Return dict with times, vis camera positions (tag frame), pico camera poses."""
    c = agc.ARM_DEFAULTS[arm]
    T_camlink_pico = agc.joint_T(root, c["joint_pico"])
    T_pico_camlink = np.linalg.inv(T_camlink_pico)  # pico_link -> camera_link
    K, dist = agc.load_intrinsics(episode, c["cam_role"])

    pico = dict(np.load(episode / "lowdim" / "pico_controllers.npz"))
    p_ts = pico["timestamp"].astype(np.float64)
    p_pose = np.asarray(pico[c["pose"]], float)
    p_valid = np.asarray(pico[c["valid"]]).astype(bool)

    cam_dir = episode / "cameras" / c["cam"]
    cam_ts = np.load(cam_dir / "color_timestamps.npy").astype(np.float64)
    cap = cv2.VideoCapture(str(cam_dir / "color.mp4"))

    times, vis_p, vis_T, pico_T = [], [], [], []
    last_cam_pos = None  # camera-in-tag position of the last accepted frame
    vmax = 0.15          # max plausible camera move between processed frames (m)
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % stride or i >= len(cam_ts):
            continue
        got = tag_candidates(frame, K, dist, dict_id, tag_id, marker_size)
        if got is None:
            continue
        _, cands = got
        cam_Ts = [np.linalg.inv(T) for T, _ in cands]  # camera (optical==camera_link) in tag
        if last_cam_pos is None:  # seed with the lowest-reprojection solution
            pick = int(np.argmin([rp for _, rp in cands]))
        else:  # resolve the flip ambiguity by continuity with the previous frame
            pick = int(np.argmin([np.linalg.norm(ct[:3, 3] - last_cam_pos) for ct in cam_Ts]))
        T_tag_cam = cam_Ts[pick]
        if last_cam_pos is not None and np.linalg.norm(T_tag_cam[:3, 3] - last_cam_pos) > vmax:
            continue  # even the best candidate jumps implausibly -> drop this frame
        last_cam_pos = T_tag_cam[:3, 3]
        t = float(cam_ts[i])
        j = nearest_idx(p_ts, t)
        if not p_valid[j]:
            continue
        T_pw_cam = agc.pose7_to_T(p_pose[j]) @ T_pico_camlink  # pico's camera pose in pico-world
        times.append(t)
        vis_p.append(T_tag_cam[:3, 3])
        vis_T.append(T_tag_cam)
        pico_T.append(T_pw_cam)
    cap.release()
    return dict(t=np.array(times), vis_p=np.array(vis_p), vis_T=vis_T, pico_T=pico_T)


def align_pico(d: dict, mode: str):
    """Return pico camera positions expressed in the tag frame, aligned to vision."""
    vis_p = d["vis_p"]
    pico_p = np.array([T[:3, 3] for T in d["pico_T"]])
    if len(vis_p) < 2:
        return pico_p
    if mode == "umeyama":
        R, t = kabsch(pico_p, vis_p)
        return (R @ pico_p.T).T + t
    # anchor at first frame (full SE(3)): A = vis_T[0] @ inv(pico_T[0])
    A = d["vis_T"][0] @ np.linalg.inv(d["pico_T"][0])
    return np.array([(A @ T)[:3, 3] for T in d["pico_T"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["right", "left"], choices=["right", "left"])
    ap.add_argument("--tag-family", default="apriltag16h5", choices=list(agc.TAG_PRESETS))
    ap.add_argument("--marker-size", type=float, default=None)
    ap.add_argument("--dict", default=None)
    ap.add_argument("--right-tag", default="auto")
    ap.add_argument("--left-tag", default="auto")
    ap.add_argument("--align", default="anchor", choices=["anchor", "umeyama"])
    ap.add_argument("--gap-s", type=float, default=0.2, help="break the plotted line across time gaps > this (s)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth camera frame")
    ap.add_argument("--out", type=Path, default=None, help="save figure(s) to this path stem")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode = args.episode.expanduser().resolve()
    root = ET.parse(str(args.urdf.expanduser().resolve())).getroot()
    preset = agc.TAG_PRESETS[args.tag_family]
    dict_id = getattr(cv2.aruco, args.dict or preset["dict"])
    marker_size = args.marker_size or preset["marker_size"]
    tag_of = {"right": args.right_tag, "left": args.left_tag}

    for arm in args.arms:
        tag_id = None if str(tag_of[arm]).lower() == "auto" else int(tag_of[arm])
        d = collect_arm(episode, root, arm, dict_id, tag_id, marker_size, args.stride)
        n = len(d["t"])
        cam_frames = len(np.load(episode / "cameras" / agc.ARM_DEFAULTS[arm]["cam"] / "color_timestamps.npy"))
        print(f"[{arm}] tag detected on {n}/{cam_frames} frames "
              f"({100*n/max(cam_frames,1):.0f}%)")
        if n < 2:
            print(f"[{arm}] too few detections to plot")
            continue

        vis_p = d["vis_p"]
        pico_p = align_pico(d, args.align)
        err_mm = np.linalg.norm(vis_p - pico_p, axis=1) * 1000.0
        tt = d["t"] - d["t"][0]
        print(f"[{arm}] pos error ({args.align}): mean={err_mm.mean():.1f} "
              f"p95={np.percentile(err_mm,95):.1f} max={err_mm.max():.1f} mm")

        # break lines across time gaps (tag not seen) so they aren't drawn as
        # a straight segment implying data / low error where there is none
        gaps = np.where(np.diff(d["t"]) > args.gap_s)[0] + 1
        vp = np.insert(vis_p, gaps, np.nan, axis=0)
        pp = np.insert(pico_p, gaps, np.nan, axis=0)
        te = np.insert(tt, gaps, np.nan)
        ee = np.insert(err_mm, gaps, np.nan)

        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(f"{episode.name}  {arm} arm  ({args.align} align, {n}/{cam_frames} frames)")
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.plot(*vp.T, color="red", lw=1.5, label="camera GT (AprilTag)")
        ax.plot(*pp.T, color="blue", lw=1.5, label="Pico")
        ax.scatter(*vis_p[0], color="k", s=30, label="start")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.legend(loc="upper left", fontsize=8)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(te, ee, color="purple", lw=1.2)
        ax2.axhline(err_mm.mean(), color="gray", ls="--", lw=0.8, label=f"mean {err_mm.mean():.1f}mm")
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("Pico vs camera error (mm)")
        ax2.set_title("per-frame position error"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        fig.tight_layout()

        if args.out is not None:
            p = args.out.expanduser()
            p = p.with_name(f"{p.stem}_{arm}{p.suffix or '.png'}")
            fig.savefig(p, dpi=130)
            print(f"[{arm}] saved {p}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
