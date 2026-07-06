"""Phase 3: Eye-in-Hand calibration data collection (RealSense + Pico controller).

Why a *warm-started* multiprocessing camera process
---------------------------------------------------
A RealSense pipeline costs ~1-2 s to (re)open and its first ~30 frames have unsettled
auto-exposure / auto-white-balance. Opening the camera per capture would re-pay that
cost and re-introduce a timestamp skew on every sample.

Instead a dedicated child process (``_camera_worker``) opens the camera **once**,
discards the warm-up frames, signals readiness, and then streams forever, copying each
new frame into a ``shared_memory`` buffer and bumping a monotonic sequence counter.
The main loop therefore always sees the freshest frame with zero latency -- the camera
is permanently "hot". This also decouples the ~30 Hz camera cadence from the ~200 Hz
Pico cadence: because the operator is told to **hold still for ~1 s** before sampling,
the (frame, pose) pair is aligned to well within the static-pose tolerance regardless
of the rate mismatch.

The tear-free read uses a tiny seqlock: the writer bumps the counter *after* the pixel
+ timestamp stores; the reader retries if the counter changed mid-copy. No lock object
is held during the pixel copy, so preview stays smooth.

Capture triggers (either one captures a sample)
    SPACE / p       keyboard capture
    Pico trigger    pull the controller trigger past --trigger-press. Only the *rising
                    edge* fires (holding it does not auto-repeat); the trigger must drop
                    below --trigger-release to re-arm. Auto-paired to --controller
                    (right_controller -> right_trigger, left -> left).

Keys (with the preview window focused)
    d           delete the last sample
    q / ESC     quit and write dataset.json

Run example
    python scripts/calibration/handeye_capture.py \\
        --record-dir calibration_data/exp01 \\
        --controller right_controller --square-size 0.025 \\
        --pattern 11 8 --serial <realsense-serial>
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Event, Process, Queue, Value, shared_memory
from pathlib import Path

import cv2
import numpy as np

from xrobotoolkit_teleop.common.xr_client import XrClient

from handeye_common import detect_chessboard, load_sample, make_object_points, write_dataset_config, write_sample

# Maps a Pico controller to its index-finger trigger key (XrClient.get_key_value_by_name).
_TRIGGER_FOR_CONTROLLER = {
    "left_controller": "left_trigger",
    "right_controller": "right_trigger",
}


# =====================================================================================
# Warm-started camera worker -- module-level so it is picklable under Windows "spawn"
# =====================================================================================


def _camera_worker(
    serial: str,
    width: int,
    height: int,
    fps: int,
    shm_name: str,
    seq: "Value",
    ts_us: "Value",
    ready_event,
    msg_queue: "Queue",
    cmd_queue: "Queue",
) -> None:
    """Child process: open RealSense once, warm up, then stream into shared memory."""
    import pyrealsense2 as rs  # imported in the child for spawn safety

    config = rs.config()
    if serial:
        config.enable_device(serial)
    # bgr8 so the shared buffer is directly usable by cv2.imshow / cv2.imwrite
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline = rs.pipeline()
    try:
        pipeline.start(config)
    except Exception as exc:
        msg_queue.put(
            {
                "error": (
                    f"Failed to start RealSense color stream "
                    f"(serial={serial or '<first device>'}, {width}x{height}@{fps}): {exc}"
                )
            }
        )
        ready_event.set()
        return

    # --- report factory intrinsics back to the parent before signalling ready ---
    profile = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
    intr = profile.get_intrinsics()
    K = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # RealSense coeffs are Brown-Conrady [k1, k2, p1, p2, k3]; pad if fewer.
    coeffs = list(intr.coeffs) + [0.0] * max(0, 5 - len(intr.coeffs))
    dist = np.array(coeffs[:5], dtype=np.float64)
    msg_queue.put({"K": K, "dist": dist})

    # --- warm-up: let auto-exposure / AWB settle ---
    for _ in range(30):
        pipeline.wait_for_frames()
    ready_event.set()

    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=100)
            color = frames.get_color_frame()
            if not color:
                continue
            arr = np.asanyarray(color.get_data())
            # seqlock write: store pixels + timestamp first, then bump the counter.
            buf[:] = arr
            ts_us.value = int(color.get_timestamp())
            seq.value += 1

            # drain commands without ever blocking the live stream
            while not cmd_queue.empty():
                try:
                    cmd = cmd_queue.get_nowait()
                except Exception:
                    break
                if cmd == "stop":
                    return
    finally:
        shm.close()
        try:
            pipeline.stop()
        except Exception:
            pass


# =====================================================================================
# Tear-free frame reader (seqlock) -- called from the parent process
# =====================================================================================


def read_latest_frame(buf_view: np.ndarray, seq: "Value", ts_us: "Value") -> tuple[np.ndarray, int]:
    """Copy the newest complete frame. Retries if the writer was mid-update."""
    for _ in range(16):
        s1 = seq.value
        snap = buf_view.copy()
        t = ts_us.value
        s2 = seq.value
        if s1 == s2:
            return snap, t
        time.sleep(0.0005)
    # extremely unlikely (writer idle): return whatever we last saw
    return buf_view.copy(), int(ts_us.value)


# =====================================================================================
# Main capture loop
# =====================================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Eye-in-Hand hand-eye calibration samples.")
    p.add_argument("--record-dir", type=Path, required=True, help="output directory for this dataset.")
    p.add_argument("--controller", default="right_controller", choices=["left_controller", "right_controller", "headset"])
    p.add_argument("--serial", default="", help="RealSense serial number (empty = first device).")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--square-size", type=float, required=True, help="chessboard square edge length in metres.")
    p.add_argument(
        "--pattern",
        type=int,
        nargs=2,
        default=[11, 8],
        metavar=("COLS", "ROWS"),
        help="number of INNER corners, e.g. --pattern 11 8.",
    )
    p.add_argument(
        "--axis-remap",
        type=float,
        nargs=9,
        default=None,
        help="optional 3x3 (row-major) to fold the Phase-2 sanity-check axis flip into the base frame.",
    )
    p.add_argument("--preview-every", type=int, default=1, help="render preview every Nth loop iteration (1 = full rate).")
    p.add_argument("--min-still-s", type=float, default=0.6, help="min seconds the hand must be still before a capture is accepted.")
    p.add_argument("--still-pos-m", type=float, default=0.003, help="max pose drift during the still window, in metres.")
    p.add_argument("--still-rot-deg", type=float, default=1.0, help="max rotation drift during the still window, in degrees.")
    p.add_argument("--max-reproj-px", type=float, default=0.8, help="max chessboard reprojection RMS allowed for a saved sample.")
    p.add_argument(
        "--pico-trigger",
        default="auto",
        choices=["auto", "left_trigger", "right_trigger", "none"],
        help="capture on the Pico trigger rising edge (auto = paired to --controller; headset has no trigger).",
    )
    p.add_argument("--trigger-press", type=float, default=0.5, help="trigger value above this counts as pressed (0-1).")
    p.add_argument("--trigger-release", type=float, default=0.2, help="trigger must drop below this to re-arm (hysteresis, 0-1).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    record_dir: Path = args.record_dir
    record_dir.mkdir(parents=True, exist_ok=True)

    pattern_size = (int(args.pattern[0]), int(args.pattern[1]))
    axis_remap = None if args.axis_remap is None else np.array(args.axis_remap, dtype=np.float64).reshape(3, 3)

    trigger_name = _resolve_trigger_name(args.controller, args.pico_trigger)
    capture_hint = "SPACE" if trigger_name is None else f"SPACE / {trigger_name}"
    win_title = f"handeye capture ({capture_hint}=capture, d=undo, q=quit)"

    # --- shared-memory frame buffer + sync primitives ---
    shm_name = f"handeye_cam_{os.getpid()}"
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=args.width * args.height * 3)
    seq = Value("Q", 0)  # monotonic write counter (uint64)
    ts_us = Value("Q", 0)  # latest RealSense color-frame timestamp (microseconds)
    ready = Event()
    msg_q: Queue = Queue()  # child -> parent (intrinsics)
    cmd_q: Queue = Queue()  # parent -> child (commands)

    cam_proc = Process(
        target=_camera_worker,
        args=(args.serial, args.width, args.height, args.fps, shm_name, seq, ts_us, ready, msg_q, cmd_q),
        daemon=True,
    )
    cam_proc.start()

    buf_view = np.ndarray((args.height, args.width, 3), dtype=np.uint8, buffer=shm.buf)

    try:
        print("Warming up camera (auto-exposure settling)...", flush=True)
        ready.wait(timeout=20.0)
        if not ready.is_set():
            raise RuntimeError("Camera warm-up timed out.")
        intr_msg = msg_q.get(timeout=5.0)
        if "error" in intr_msg:
            raise RuntimeError(intr_msg["error"])
        K, dist = intr_msg["K"], intr_msg["dist"]
        objp = make_object_points(args.square_size, pattern_size)
        print("Camera is hot. Intrinsics:\n", K)
        print("Distortion:", dist.ravel())

        # --- XR (Pico) side ---
        xr = XrClient()
        trig_msg = f", capture on {trigger_name} (pull past {args.trigger_press})" if trigger_name else ", capture on SPACE"
        print(
            f"Reading Pico pose from '{args.controller}'{trig_msg}.\n"
            f"Hold still >= {args.min_still_s:.2f}s before capturing.\n"
        )

        samples = list(_existing_samples(record_dir))  # resume support
        next_index = (max((int(p.stem.split("_")[1]) for p in samples), default=-1) + 1) if samples else 0
        if samples:
            print(f"Resuming: {len(samples)} existing sample(s) in {record_dir}; next index = {next_index}.")

        still_ref_pose = None
        still_since = None
        loop_i = 0
        trig_armed = True  # rising-edge / hysteresis state for the Pico trigger
        next_invalid_warn = 0.0

        try:
            while True:
                loop_i += 1
                pose = xr.get_pose_by_name(args.controller)  # [x,y,z,qx,qy,qz,qw]
                frame_bgr, cam_ts = read_latest_frame(buf_view, seq, ts_us)
                if not _pose_has_valid_quat(pose):
                    now = time.monotonic()
                    if now >= next_invalid_warn:
                        print(f"  [wait] {args.controller} pose is not valid yet (zero quaternion).")
                        next_invalid_warn = now + 2.0
                    if loop_i % max(1, args.preview_every) == 0:
                        cv2.imshow(win_title, _draw_waiting_pose(frame_bgr, args.controller))
                    key = cv2.waitKey(50) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    continue

                # Stillness detection against a fixed reference pose for the whole
                # still window, not just consecutive frame deltas.
                if still_ref_pose is None or not _pose_within_tolerance(
                    still_ref_pose, pose, args.still_pos_m, args.still_rot_deg
                ):
                    still_ref_pose = pose
                    still_since = time.monotonic()
                still_ok = still_since is not None and (time.monotonic() - still_since) >= args.min_still_s

                if loop_i % max(1, args.preview_every) == 0:
                    preview = _draw_preview(frame_bgr, pattern_size, still_ok, len(samples), capture_hint)
                    cv2.imshow(win_title, preview)

                # Pico trigger: rising edge only (hold-to-repeat disabled via hysteresis).
                trigger_rising = False
                if trigger_name is not None:
                    tv = float(xr.get_key_value_by_name(trigger_name))
                    if trig_armed and tv > args.trigger_press:
                        trigger_rising = True
                        trig_armed = False
                    elif not trig_armed and tv < args.trigger_release:
                        trig_armed = True

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord(" "), ord("p")) or trigger_rising:
                    if not still_ok:
                        print(f"  [skip] hand not still enough -- hold steady for ~1 s, then {capture_hint} again.")
                        continue
                    # re-grab the freshest frame + pose at the same instant to minimise skew
                    frame_bgr, cam_ts = read_latest_frame(buf_view, seq, ts_us)
                    pose_now = xr.get_pose_by_name(args.controller)
                    if not _pose_has_valid_quat(pose_now):
                        print(f"  [skip] {args.controller} pose became invalid -- wake/track the controller and retry.")
                        continue
                    board_ok, reproj_rms = _chessboard_reprojection_rms(frame_bgr, pattern_size, objp, K, dist)
                    if not board_ok:
                        print("  [skip] chessboard not detected in the saved frame -- re-aim and retry.")
                        continue
                    if reproj_rms > args.max_reproj_px:
                        print(
                            f"  [skip] chessboard reprojection RMS {reproj_rms:.2f}px > "
                            f"{args.max_reproj_px:.2f}px -- avoid blur/edge views."
                        )
                        continue
                    if samples and _pose_matches_last_sample(record_dir, samples[-1], pose_now):
                        print(
                            f"  [skip] {args.controller} pose is unchanged from the previous sample -- "
                            "move/retrack the controller before capturing."
                        )
                        continue
                    src = "trigger" if trigger_rising else "SPACE"
                    meta = {
                        "source": src,
                        "trigger_value": float(tv) if trigger_name is not None else None,
                        "camera_ts_us": int(cam_ts),
                        "pico_ts_ns": int(xr.get_timestamp_ns()),
                        "capture_mono_ns": time.monotonic_ns(),
                        "n_corners": int(corners.shape[0]),
                    }
                    write_sample(record_dir, next_index, frame_bgr, pose_now, meta)
                    print(f"  [ok] sample_{next_index:03d} saved via {src}. total = {len(samples) + 1}")
                    samples.append(record_dir / f"sample_{next_index:03d}.json")
                    next_index += 1
                elif key == ord("d"):
                    if samples:
                        last = samples.pop()
                        for suffix in (".png", ".json"):
                            f = record_dir / (last.stem + suffix)
                            if f.exists():
                                f.unlink()
                        next_index = int(last.stem.split("_")[1])
                        print(f"  [undo] removed {last.stem}. total = {len(samples)}")
        except KeyboardInterrupt:
            print("\nInterrupted; writing dataset.json before shutdown.")

        # --- persist dataset config (intrinsics + chessboard params) ---
        write_dataset_config(
            record_dir=record_dir,
            camera_matrix=K,
            dist_coeffs=dist,
            square_size_m=args.square_size,
            pattern_size=pattern_size,
            axis_remap=axis_remap,
            image_size_wh=(args.width, args.height),
        )
        print(f"\nDone. {len(samples)} sample(s) written to {record_dir}.")
        print("Next: run handeye_solve.py on this record-dir.")

    finally:
        cv2.destroyAllWindows()
        cmd_q.put("stop")
        cam_proc.join(timeout=5.0)
        if cam_proc.is_alive():
            cam_proc.terminate()
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        try:
            xr.close()
        except NameError:
            pass


# =====================================================================================
# Small helpers
# =====================================================================================


def _resolve_trigger_name(controller: str, choice: str) -> str | None:
    """Resolve --pico-trigger to an XrClient key name, or None if disabled / unavailable."""
    if choice == "none":
        return None
    if choice == "auto":
        return _TRIGGER_FOR_CONTROLLER.get(controller)  # None for headset (no trigger)
    return choice


def _existing_samples(record_dir: Path):
    return sorted(Path(record_dir).glob("sample_*.json"))


def _pose_has_valid_quat(pose) -> bool:
    """XR can return a zero quaternion while tracking is not ready."""
    if pose is None:
        return False
    arr = np.asarray(pose, dtype=np.float64)
    return arr.shape[0] >= 7 and np.linalg.norm(arr[3:7]) > 1e-8


def _pose_within_tolerance(pose_a, pose_b, pos_tol_m: float, rot_tol_deg: float) -> bool:
    """True when two Pico poses are within a small translation/rotation window."""
    if pose_a is None or pose_b is None or not _pose_has_valid_quat(pose_a) or not _pose_has_valid_quat(pose_b):
        return False
    a = np.asarray(pose_a, dtype=np.float64)
    b = np.asarray(pose_b, dtype=np.float64)
    if np.linalg.norm(a[:3] - b[:3]) > pos_tol_m:
        return False
    qa = a[3:] / np.linalg.norm(a[3:])
    qb = b[3:] / np.linalg.norm(b[3:])
    dot = np.clip(abs(float(np.dot(qa, qb))), -1.0, 1.0)
    rot_deg = float(np.degrees(2.0 * np.arccos(dot)))
    return rot_deg <= rot_tol_deg


def _chessboard_reprojection_rms(
    frame_bgr: np.ndarray,
    pattern_size,
    objp: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[bool, float]:
    """Detect the board on the exact saved frame and reject bad corner fits."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    ok, corners = detect_chessboard(gray, pattern_size)
    if not ok or corners is None:
        return False, float("inf")
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return False, float("inf")
    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    return True, float(np.sqrt(np.mean(err * err)))


def _pose_matches_last_sample(record_dir: Path, last_sample: Path, pose, pos_tol_m=0.001, rot_tol=1e-5) -> bool:
    """True when the new pose is effectively a duplicate of the last saved sample."""
    try:
        prev = load_sample(record_dir / last_sample.name if not last_sample.is_absolute() else last_sample)[
            "pico_pose_xyzq"
        ]
    except Exception:
        return False
    if not _pose_has_valid_quat(prev) or not _pose_has_valid_quat(pose):
        return False
    a = np.asarray(prev, dtype=np.float64)
    b = np.asarray(pose, dtype=np.float64)
    if np.linalg.norm(a[:3] - b[:3]) > pos_tol_m:
        return False
    dot = abs(float(np.dot(a[3:] / np.linalg.norm(a[3:]), b[3:] / np.linalg.norm(b[3:]))))
    return dot > 1.0 - rot_tol


def _draw_waiting_pose(frame_bgr: np.ndarray, controller: str) -> np.ndarray:
    out = frame_bgr.copy()
    cv2.putText(
        out,
        f"waiting for valid {controller} pose...",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2,
    )
    return out


def _draw_preview(
    frame_bgr: np.ndarray, pattern_size, still_ok: bool, n_samples: int, capture_hint: str = "SPACE"
) -> np.ndarray:
    out = frame_bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    ok, corners = detect_chessboard(gray, pattern_size)
    if ok and corners is not None:
        cv2.drawChessboardCorners(out, pattern_size, corners, ok)
    if not ok:
        status, color = "no board", (0, 0, 255)
    elif still_ok:
        status, color = f"STILL -> {capture_hint}", (0, 255, 0)
    else:
        status, color = "keep still...", (0, 128, 255)
    cv2.putText(out, f"samples={n_samples}  {status}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


if __name__ == "__main__":
    main()
