"""Cross-check Pico gripper tracking against ArUco vision, per episode.

Each arm's wrist camera sees a single static ArUco tag on the table at the start
and end of the episode. We solve the gripper pose in the tag (scene) frame two
independent ways:

    vision : solvePnP(tag) -> T_optcam_tag  ->  (URDF camera_link->gripper)
    pico   : recorded controller pose        ->  (URDF pico_link->gripper)

We anchor the two world frames (tag-world vs Pico-world) using the FIRST frame,
then report the mismatch at the LAST frame. That mismatch is the accumulated Pico
tracking drift over the episode (plus any calibration / frame-convention error) --
i.e. a metric ground-truth check on the tracker, using no robot.

On session_20260709_141300/episode_002 this reports ~4 mm (right) / ~10 mm (left)
of drift over 25 s, consistent with the known ~8 mm Pico tracking floor, and it
validates that the URDF camera_link is (to within that floor) the optical frame.

Rig mapping: right arm -> cam0 (right_cam), left arm -> cam1 (left_cam). Each
wrist camera sees one tag, so tag ids are auto-detected (--right-tag/--left-tag
to pin). Tag family via --tag-family:
    apriltag16h5 (default): DICT_APRILTAG_16h5, 150 mm  -- new collections
    aruco4x4              : DICT_4X4_50,        152 mm  -- the earlier sessions

Usage:
    # new AprilTag sessions
    python scripts/calibration/aruco_gripper_check.py \
        --episode /path/session_XXXX/episode_002 --urdf /path/dual_yam.urdf
    # older ArUco sessions
    python scripts/calibration/aruco_gripper_check.py ... --tag-family aruco4x4

Env: opencv-contrib-python (cv2.aruco), numpy. Run in the sim/vision venv, not
the numpy-2.3 conda base.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


# --- per-arm rig mapping (dual-YAM defaults) ---------------------------------
ARM_DEFAULTS = {
    "right": dict(cam="cam0", cam_role="right_cam",
                  pose="right_pose", valid="right_valid",
                  joint_cam="right_jointcamera", joint_pico="right_joint_pico"),
    "left": dict(cam="cam1", cam_role="left_cam",
                 pose="left_pose", valid="left_valid",
                 joint_cam="left_jointcamera", joint_pico="left_joint_pico"),
}

# Tag families. Each wrist camera sees one tag, so the id is auto-detected by
# default (--right-tag/--left-tag to pin). apriltag16h5 is the new default;
# aruco4x4 kept for the earlier sessions (DICT_4X4_50, 152mm, right=15/left=19).
#
# SIZE PITFALL: marker_size is the BLACK square edge, which is what OpenCV's
# corners delimit. A tag16h5 print is 8x8 modules (6x6 black + 1-module white
# border), so a "15x15cm" printed tag has a 15*6/8 = 11.25cm black square.
# Getting this wrong scales every PnP distance by the same ratio (~33% here,
# -> ~100mm trajectory error). Verified against Pico on session_20260710_151337:
# scale fit gave true size 112.4/111.1mm for the two tags. Measure the black
# square with a ruler if in doubt.
TAG_PRESETS = {
    "apriltag16h5": {"dict": "DICT_APRILTAG_16h5", "marker_size": 0.1125},
    "aruco4x4": {"dict": "DICT_4X4_50", "marker_size": 0.152},
}


# --- small SE(3) helpers -----------------------------------------------------
def rpy_to_R(r, p, y):
    """URDF fixed-axis roll-pitch-yaw -> rotation matrix (Rz*Ry*Rx)."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pose7_to_T(pose7):
    """[x,y,z, qx,qy,qz,qw] -> 4x4."""
    x, y, z, qx, qy, qz, qw = [float(v) for v in pose7[:7]]
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])
    T[:3, 3] = [x, y, z]
    return T


def rot_angle(Ta, Tb):
    R = Ta[:3, :3].T @ Tb[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def pos_err_mm(Ta, Tb):
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1000)


# --- inputs ------------------------------------------------------------------
def joint_T(root, name):
    for j in root.iter("joint"):
        if j.get("name") == name:
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz").split()]
            rpy = [float(v) for v in o.get("rpy").split()]
            T = np.eye(4)
            T[:3, :3] = rpy_to_R(*rpy)
            T[:3, 3] = xyz
            return T
    raise KeyError(f"joint {name} not in URDF")


def load_intrinsics(episode: Path, role: str):
    meta = json.loads((episode / "cameras" / "metadata.json").read_text())
    intr = meta["camera_config_data"]["roles"][role]["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], float)
    dist = np.array(intr["coeffs"], float)
    return K, dist


def read_frame(mp4: Path, which: str):
    """which in {'first','last'}: return the frame as BGR ndarray."""
    cap = cv2.VideoCapture(str(mp4))
    if which == "first":
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n - 1, 0))
    ok, frame = cap.read()
    if not ok:  # some codecs won't seek; decode through
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = None
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frame = f
    cap.release()
    return frame


def solve_tag(img, K, dist, dict_id, tag_id, marker_size):
    """Return (T_optcam_tag, detected_id) or None.

    tag_id None => auto: use the largest-area detected marker (each wrist camera
    sees a single tag; largest-area rejects spurious 16h5 detections).
    """
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
    ok, rvec, tvec = cv2.solvePnP(objp, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.flatten()
    return T, ids[k]  # T_optcam_tag, detected id


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["right", "left"], choices=["right", "left"])
    ap.add_argument("--tag-family", default="apriltag16h5", choices=list(TAG_PRESETS),
                    help="tag preset (apriltag16h5=new default; aruco4x4=old sessions)")
    ap.add_argument("--marker-size", type=float, default=None, help="override marker square size (m)")
    ap.add_argument("--dict", default=None, help="override cv2.aruco dictionary name")
    ap.add_argument("--right-tag", default="auto", help="right tag id, or 'auto' (largest marker)")
    ap.add_argument("--left-tag", default="auto", help="left tag id, or 'auto'")
    args = ap.parse_args()

    episode = args.episode.expanduser().resolve()
    root = ET.parse(str(args.urdf.expanduser().resolve())).getroot()
    preset = TAG_PRESETS[args.tag_family]
    dict_name = args.dict or preset["dict"]
    marker_size = args.marker_size or preset["marker_size"]
    dict_id = getattr(cv2.aruco, dict_name)
    tag_ids = {"right": args.right_tag, "left": args.left_tag}
    pico = dict(np.load(episode / "lowdim" / "pico_controllers.npz"))

    print(f"episode: {episode.name}   dict: {dict_name}   marker: {marker_size*1000:.0f}mm\n")
    for arm in args.arms:
        c = ARM_DEFAULTS[arm]
        tag_id = None if str(tag_ids[arm]).lower() == "auto" else int(tag_ids[arm])
        # URDF fixed transforms: gripper->camera_link (joint_cam), camera_link->pico (joint_pico)
        T_g_cam = joint_T(root, c["joint_cam"])
        T_cam_pico = joint_T(root, c["joint_pico"])
        T_cam_g = np.linalg.inv(T_g_cam)
        T_pico_g = np.linalg.inv(T_g_cam @ T_cam_pico)

        K, dist = load_intrinsics(episode, c["cam_role"])
        poses = np.asarray(pico[c["pose"]], float)
        valid = np.asarray(pico[c["valid"]]).astype(bool)
        i0 = int(np.argmax(valid))
        iN = len(valid) - 1 - int(np.argmax(valid[::-1]))

        got = {}
        for end, i in [("first", i0), ("last", iN)]:
            img = read_frame(episode / "cameras" / c["cam"] / "color.mp4", end)
            if img is None:
                print(f"[{arm}] {end}: could not read frame")
                continue
            res = solve_tag(img, K, dist, dict_id, tag_id, marker_size)
            if res is None:
                want = "any tag" if tag_id is None else f"tag {tag_id}"
                print(f"[{arm}] {end}: {want} not detected in {c['cam']}")
                continue
            T_oc_tag, det_id = res
            print(f"[{arm}] {end}: tag id {det_id}")
            T_tag_g_vis = np.linalg.inv(T_oc_tag) @ T_cam_g          # optical==camera_link
            T_pw_g_pico = pose7_to_T(poses[i]) @ T_pico_g
            got[end] = (T_tag_g_vis, T_pw_g_pico)

        if "first" not in got or "last" not in got:
            print(f"[{arm}] need both first & last tag detections; skipping\n")
            continue
        (Tv0, Tp0), (TvN, TpN) = got["first"], got["last"]
        T_tag_pw = Tv0 @ np.linalg.inv(Tp0)                          # anchor worlds at t0
        moved = np.linalg.norm(TvN[:3, 3] - Tv0[:3, 3]) * 1000       # how far the gripper moved
        drift_pos = pos_err_mm(TvN, T_tag_pw @ TpN)
        drift_rot = rot_angle(TvN, T_tag_pw @ TpN)
        print(f"[{arm}] gripper moved {moved:.0f}mm start->end")
        print(f"[{arm}] t0 anchor residual : {pos_err_mm(Tv0, T_tag_pw @ Tp0):.2f}mm "
              f"{rot_angle(Tv0, T_tag_pw @ Tp0):.2f}deg  (should be ~0)")
        print(f"[{arm}] tN Pico drift vs vision : {drift_pos:.1f}mm  {drift_rot:.2f}deg\n")


if __name__ == "__main__":
    main()
