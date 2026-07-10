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


def make_detector(dict_id):
    params = cv2.aruco.DetectorParameters()
    # sub-pixel corners: PnP rotation noise scales directly with corner noise
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)


def pick_tag_corners(det, img, tag_id):
    """Detect and return (marker_id, 4x2 corners) for the pinned id, or the
    largest marker when tag_id is None. None if nothing detected."""
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
    return ids[k], corners[k][0]


class TagPoseTracker:
    """Temporally-coherent single-tag PnP.

    A planar square has two mirror IPPE poses whose reprojection errors are
    often nearly equal under oblique views, so picking per-frame by reprojection
    lets the solution flip/wander between mirror branches. The rotation error is
    then amplified by the camera-to-tag distance (lever arm): at 0.7 m, 40 deg of
    wander = ~500 mm of camera-position error. Measured on
    session_20260710_151337: vision inter-frame rotation p95 was 7.3 deg/frame vs
    1.6 deg/frame actual (Pico), i.e. mostly PnP noise, not motion.

    Strategy:
      seed  : IPPE_SQUARE, lowest reprojection
      track : SOLVEPNP_ITERATIVE refined from the previous frame's pose --
              stays on the same mirror branch by construction
      gate  : mean reprojection < max_reproj px, rotation step < max_rot_step
      reseed: after `reseed_after` consecutive rejections, start over with IPPE
    """

    def __init__(self, K, dist, marker_size, max_reproj=3.0,
                 max_rot_step_deg=25.0, reseed_after=15):
        h = marker_size / 2
        self.objp = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
        self.K, self.dist = K, dist
        self.max_reproj = float(max_reproj)
        self.max_rot = np.radians(max_rot_step_deg)
        self.reseed_after = int(reseed_after)
        self.rvec = None
        self.tvec = None
        self.misses = 0

    def _reproj_px(self, rvec, tvec, c):
        proj, _ = cv2.projectPoints(self.objp, rvec, tvec, self.K, self.dist)
        return float(np.sqrt(((proj.reshape(-1, 2) - c) ** 2).sum(1)).mean())

    def _seed(self, c):
        try:
            n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                self.objp, c, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return None
        if n < 1:
            return None
        b = int(np.argmin(np.asarray(reproj, float).flatten()))
        return (np.asarray(rvecs[b], float).reshape(3, 1),
                np.asarray(tvecs[b], float).reshape(3, 1))

    def _miss(self):
        self.misses += 1
        if self.misses >= self.reseed_after:
            self.rvec = self.tvec = None
            self.misses = 0
        return None

    def update(self, c):
        """c: 4x2 marker corners. Returns T_optcam_tag or None (rejected)."""
        prev = self.rvec
        if prev is None:
            got = self._seed(c)
            if got is None:
                return None
            rvec, tvec = got
        else:
            ok, rvec, tvec = cv2.solvePnP(
                self.objp, c, self.K, self.dist,
                rvec=self.rvec.copy(), tvec=self.tvec.copy(),
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return self._miss()
            rvec = np.asarray(rvec, float).reshape(3, 1)
            tvec = np.asarray(tvec, float).reshape(3, 1)
        if self._reproj_px(rvec, tvec, c) > self.max_reproj:
            return self._miss()
        if prev is not None:
            R0 = cv2.Rodrigues(prev)[0]
            R1 = cv2.Rodrigues(rvec)[0]
            ang = np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1))
            if ang > self.max_rot:
                return self._miss()
        self.rvec, self.tvec = rvec, tvec
        self.misses = 0
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = np.asarray(tvec, float).flatten()
        return T


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

    det = make_detector(dict_id)
    tracker = TagPoseTracker(K, dist, marker_size)
    times, vis_p, vis_T, pico_T = [], [], [], []
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % stride or i >= len(cam_ts):
            continue
        got = pick_tag_corners(det, frame, tag_id)
        if got is None:
            continue
        T_oc_tag = tracker.update(got[1])
        if T_oc_tag is None:
            continue
        T_tag_cam = np.linalg.inv(T_oc_tag)  # camera (optical==camera_link) in tag frame
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
    ap.add_argument("--animate", action="store_true",
                    help="play the two trajectories as a growing real-time animation")
    ap.add_argument("--anim-speed", type=float, default=1.0, help="animation speed multiplier")
    ap.add_argument("--save-video", type=Path, default=None,
                    help="save the animation as mp4 (needs ffmpeg); implies --animate")
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

        animate = args.animate or args.save_video is not None

        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(f"{episode.name}  {arm} arm  ({args.align} align, {n}/{cam_frames} frames)")
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        # static full tracks (faint) so the animation has context + fixed limits
        ax.plot(*vp.T, color="red", lw=1.5 if not animate else 0.4,
                alpha=1.0 if not animate else 0.25, label="camera GT (AprilTag)")
        ax.plot(*pp.T, color="blue", lw=1.5 if not animate else 0.4,
                alpha=1.0 if not animate else 0.25, label="Pico")
        ax.scatter(*vis_p[0], color="k", s=30, label="start")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.legend(loc="upper left", fontsize=8)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axhline(err_mm.mean(), color="gray", ls="--", lw=0.8, label=f"mean {err_mm.mean():.1f}mm")
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("Pico vs camera error (mm)")
        ax2.set_title("per-frame position error"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        if animate:
            ax2.set_xlim(0, np.nanmax(te))
            ax2.set_ylim(0, np.nanmax(ee) * 1.05)
        else:
            ax2.plot(te, ee, color="purple", lw=1.2)
        fig.tight_layout()

        if animate:
            from matplotlib.animation import FuncAnimation

            lv, = ax.plot([], [], [], color="red", lw=1.8)
            lp, = ax.plot([], [], [], color="blue", lw=1.8)
            hv, = ax.plot([], [], [], "o", color="red", ms=6)
            hp, = ax.plot([], [], [], "o", color="blue", ms=6)
            le, = ax2.plot([], [], color="purple", lw=1.2)
            txt = ax2.text(0.02, 0.95, "", transform=ax2.transAxes, fontsize=9, va="top")

            def update(i):
                lv.set_data_3d(vp[:i + 1, 0], vp[:i + 1, 1], vp[:i + 1, 2])
                lp.set_data_3d(pp[:i + 1, 0], pp[:i + 1, 1], pp[:i + 1, 2])
                if np.isfinite(vp[i]).all():
                    hv.set_data_3d([vp[i, 0]], [vp[i, 1]], [vp[i, 2]])
                if np.isfinite(pp[i]).all():
                    hp.set_data_3d([pp[i, 0]], [pp[i, 1]], [pp[i, 2]])
                le.set_data(te[:i + 1], ee[:i + 1])
                if np.isfinite(ee[i]):
                    txt.set_text(f"t={te[i]:5.2f}s   err={ee[i]:6.1f} mm")
                return lv, lp, hv, hp, le, txt

            dtm = float(np.nanmedian(np.diff(te)))
            fps = min(60.0, max(5.0, args.anim_speed / max(dtm, 1e-3)))
            anim = FuncAnimation(fig, update, frames=len(te),
                                 interval=1000.0 * dtm / args.anim_speed, blit=False)
            if args.save_video is not None:
                p = args.save_video.expanduser()
                p = p.with_name(f"{p.stem}_{arm}{p.suffix or '.mp4'}")
                anim.save(str(p), fps=fps, dpi=110)
                print(f"[{arm}] saved video {p}")
            fig._anim = anim  # keep a reference so plt.show() animates

        if args.out is not None:
            p = args.out.expanduser()
            p = p.with_name(f"{p.stem}_{arm}{p.suffix or '.png'}")
            fig.savefig(p, dpi=130)
            print(f"[{arm}] saved {p}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
