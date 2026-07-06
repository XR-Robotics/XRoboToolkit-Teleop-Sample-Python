"""Estimate a 6x8 ArUco GridBoard pose from an Intel RealSense color stream.

Default board:
    markers: 6 columns x 8 rows
    square pitch: 32 mm
    marker side: 24 mm

OpenCV returns the board pose in the camera optical frame:
    x right, y down, z forward, units in metres.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is not installed. Install an environment with cv2 available."
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise SystemExit(
            "This OpenCV build has no cv2.aruco module. Install opencv-contrib-python "
            "or an OpenCV build that includes ArUco."
        )
    return cv2


def _import_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed in this Python environment."
        ) from exc
    return rs


def _aruco_dictionary(cv2, name: str):
    aruco = cv2.aruco
    if name.isdigit():
        dict_id = int(name)
    else:
        if not name.startswith("DICT_"):
            name = f"DICT_{name}"
        if not hasattr(aruco, name):
            available = sorted(k for k in dir(aruco) if k.startswith("DICT_"))
            raise SystemExit(
                f"Unknown ArUco dictionary {name!r}. Available examples: "
                f"{', '.join(available[:12])}"
            )
        dict_id = getattr(aruco, name)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dict_id)
    return aruco.Dictionary_get(dict_id)


def _make_grid_board(
    cv2,
    *,
    markers_x: int,
    markers_y: int,
    marker_length_m: float,
    marker_separation_m: float,
    dictionary,
    first_marker_id: int,
):
    aruco = cv2.aruco
    marker_count = markers_x * markers_y
    ids = np.arange(first_marker_id, first_marker_id + marker_count, dtype=np.int32)
    if hasattr(aruco, "GridBoard_create"):
        try:
            return aruco.GridBoard_create(
                markers_x,
                markers_y,
                marker_length_m,
                marker_separation_m,
                dictionary,
                first_marker_id,
            )
        except TypeError:
            return aruco.GridBoard_create(
                markers_x,
                markers_y,
                marker_length_m,
                marker_separation_m,
                dictionary,
            )
    try:
        return aruco.GridBoard(
            (markers_x, markers_y),
            marker_length_m,
            marker_separation_m,
            dictionary,
            ids,
        )
    except TypeError:
        return aruco.GridBoard(
            (markers_x, markers_y),
            marker_length_m,
            marker_separation_m,
            dictionary,
        )


def _make_detector(cv2, dictionary, *, detect_inverted: bool):
    aruco = cv2.aruco
    if hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
    else:
        params = aruco.DetectorParameters_create()
    if hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = bool(detect_inverted)
    if hasattr(params, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, params), params
    return None, params


def _detect_markers(cv2, image_gray: np.ndarray, dictionary, detector, params):
    aruco = cv2.aruco
    if detector is not None:
        return detector.detectMarkers(image_gray)
    return aruco.detectMarkers(image_gray, dictionary, parameters=params)


def _refine_markers(cv2, image_gray, board, corners, ids, rejected, camera_matrix, dist_coeffs):
    if ids is None or len(corners) == 0 or not hasattr(cv2.aruco, "refineDetectedMarkers"):
        return corners, ids, rejected
    try:
        result = cv2.aruco.refineDetectedMarkers(
            image_gray,
            board,
            corners,
            ids,
            rejected,
            camera_matrix,
            dist_coeffs,
        )
    except Exception:
        return corners, ids, rejected
    if isinstance(result, tuple) and len(result) >= 3:
        return result[0], result[1], result[2]
    return corners, ids, rejected


def _estimate_pose(cv2, board, corners, ids, camera_matrix, dist_coeffs):
    if ids is None or len(ids) == 0:
        return None
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    try:
        result = cv2.aruco.estimatePoseBoard(
            corners, ids, board, camera_matrix, dist_coeffs, rvec, tvec
        )
    except Exception:
        result = cv2.aruco.estimatePoseBoard(
            corners, ids, board, camera_matrix, dist_coeffs, None, None
        )
    if isinstance(result, tuple):
        markers_used, rvec, tvec = result[:3]
    else:
        markers_used = result
    if int(markers_used) <= 0:
        return None
    return np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64), int(markers_used)


def _board_marker_map(board) -> dict[int, np.ndarray]:
    if hasattr(board, "getIds"):
        ids = np.asarray(board.getIds()).reshape(-1)
    else:
        ids = np.asarray(board.ids).reshape(-1)
    if hasattr(board, "getObjPoints"):
        obj_points = board.getObjPoints()
    else:
        obj_points = board.objPoints
    return {
        int(marker_id): np.asarray(obj_points[index], dtype=np.float64).reshape(-1, 3)
        for index, marker_id in enumerate(ids)
    }


def _reprojection_error_px(cv2, board_points, corners, ids, rvec, tvec, camera_matrix, dist_coeffs):
    if ids is None:
        return None
    errors = []
    for corner, marker_id in zip(corners, ids.reshape(-1)):
        obj = board_points.get(int(marker_id))
        if obj is None:
            continue
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        diff = projected.reshape(-1, 2) - np.asarray(corner).reshape(-1, 2)
        errors.extend(np.linalg.norm(diff, axis=1).tolist())
    if not errors:
        return None
    return float(np.mean(errors))


def _intrinsics_to_opencv(intrinsics):
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(intrinsics.coeffs, dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def _pose_payload(
    cv2,
    *,
    serial: str | None,
    frame_timestamp_ms: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    markers_used: int,
    reprojection_error_px: float | None,
):
    rotation_matrix = cv2.Rodrigues(rvec)[0]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = tvec.reshape(3)
    return {
        "timestamp_host_s": time.time(),
        "timestamp_realsense_ms": float(frame_timestamp_ms),
        "serial_number": serial,
        "markers_used": int(markers_used),
        "reprojection_error_px": reprojection_error_px,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "rvec_board_to_camera": rvec.reshape(3).tolist(),
        "t_board_to_camera_m": tvec.reshape(3).tolist(),
        "R_board_to_camera": rotation_matrix.tolist(),
        "T_board_to_camera": transform.tolist(),
    }


def _draw_text(image, cv2, lines: list[str]) -> None:
    x, y = 12, 24
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24


def _draw_axes(cv2, image, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m: float) -> None:
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m)
        return
    if hasattr(cv2.aruco, "drawAxis"):
        cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m)


def _dictionary_scan(cv2, image_gray: np.ndarray, *, detect_inverted: bool) -> list[tuple[str, int]]:
    names = sorted(
        name
        for name in dir(cv2.aruco)
        if name.startswith("DICT_") and name not in {"DICT_ARUCO_ORIGINAL"}
    )
    counts = []
    for name in names:
        dictionary = _aruco_dictionary(cv2, name)
        detector, params = _make_detector(cv2, dictionary, detect_inverted=detect_inverted)
        _, ids, _ = _detect_markers(cv2, image_gray, dictionary, detector, params)
        counts.append((name, 0 if ids is None else len(ids)))
    return sorted(counts, key=lambda item: item[1], reverse=True)


def _list_devices(rs) -> None:
    devices = list(rs.context().query_devices())
    if not devices:
        print("No RealSense devices detected.")
        return
    for index, dev in enumerate(devices):
        fields = {}
        for name in ("name", "serial_number", "product_line", "firmware_version"):
            try:
                fields[name] = dev.get_info(getattr(rs.camera_info, name))
            except Exception:
                pass
        print(f"{index}: {fields}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None, help="RealSense serial number. Defaults to the first device.")
    parser.add_argument("--list-devices", action="store_true", help="Print RealSense devices and exit.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--markers-x", type=int, default=6, help="Number of marker columns.")
    parser.add_argument("--markers-y", type=int, default=8, help="Number of marker rows.")
    parser.add_argument("--square-mm", type=float, default=32.0, help="Marker pitch / square side in millimetres.")
    parser.add_argument("--marker-mm", type=float, default=24.0, help="Black marker side length in millimetres.")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="ArUco dictionary name or numeric OpenCV id.")
    parser.add_argument("--first-marker-id", type=int, default=0)
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path, default=None, help="Continuously overwrite with the latest pose.")
    parser.add_argument("--jsonl", type=Path, default=None, help="Append every valid pose to this JSONL log.")
    parser.add_argument("--snapshot-dir", type=Path, default=None, help="Press 's' to save annotated frames here.")
    parser.add_argument("--no-window", action="store_true", help="Run headless and print poses only.")
    parser.add_argument("--duration-s", type=float, default=None, help="Optional run duration in seconds.")
    parser.add_argument("--print-every", type=int, default=15, help="Print one pose every N valid frames.")
    parser.add_argument("--print-detected-ids", action="store_true", help="Print marker IDs detected in each frame.")
    parser.add_argument("--min-markers-for-pose", type=int, default=4, help="Minimum board markers before drawing/saving pose.")
    parser.add_argument("--no-detect-inverted", action="store_true", help="Disable white-on-black marker detection.")
    parser.add_argument("--scan-dictionaries", action="store_true", help="Print which ArUco dictionaries detect the most markers.")
    parser.add_argument("--no-refine", action="store_true", help="Disable cv2.aruco.refineDetectedMarkers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2 = _import_cv2()
    rs = _import_realsense()

    if args.list_devices:
        _list_devices(rs)
        return

    marker_length_m = args.marker_mm / 1000.0
    square_length_m = args.square_mm / 1000.0
    marker_separation_m = square_length_m - marker_length_m
    if marker_separation_m <= 0:
        raise SystemExit("--square-mm must be larger than --marker-mm.")

    dictionary = _aruco_dictionary(cv2, args.dictionary)
    board = _make_grid_board(
        cv2,
        markers_x=args.markers_x,
        markers_y=args.markers_y,
        marker_length_m=marker_length_m,
        marker_separation_m=marker_separation_m,
        dictionary=dictionary,
        first_marker_id=args.first_marker_id,
    )
    board_points = _board_marker_map(board)
    detect_inverted = not args.no_detect_inverted
    detector, detector_params = _make_detector(cv2, dictionary, detect_inverted=detect_inverted)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    active_serial = args.serial
    try:
        if active_serial is None:
            try:
                active_serial = profile.get_device().get_info(rs.camera_info.serial_number)
            except Exception:
                active_serial = None
        stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        camera_matrix, dist_coeffs = _intrinsics_to_opencv(stream_profile.get_intrinsics())
        print(
            "[ARUCO] Started RealSense "
            f"serial={active_serial} {args.width}x{args.height}@{args.fps}; "
            f"board={args.markers_x}x{args.markers_y}, marker={marker_length_m:.3f}m, "
            f"separation={marker_separation_m:.3f}m, dictionary={args.dictionary}"
        )
        print("[ARUCO] Pose is T_board_to_camera. Press q/esc to quit, s to save snapshot.")

        for _ in range(max(args.warmup_frames, 0)):
            pipeline.wait_for_frames(timeout_ms=5000)

        valid_frame_count = 0
        end_time = None if args.duration_s is None else time.time() + args.duration_s
        while True:
            if end_time is not None and time.time() >= end_time:
                break
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if args.scan_dictionaries:
                top = _dictionary_scan(cv2, gray, detect_inverted=detect_inverted)[:8]
                print("[ARUCO] dictionary scan: " + ", ".join(f"{name}={count}" for name, count in top))
                break
            corners, ids, rejected = _detect_markers(cv2, gray, dictionary, detector, detector_params)
            if not args.no_refine:
                corners, ids, rejected = _refine_markers(
                    cv2, gray, board, corners, ids, rejected, camera_matrix, dist_coeffs
                )

            annotated = image.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
                if args.print_detected_ids:
                    print(f"[ARUCO] detected ids: {ids.reshape(-1).astype(int).tolist()}")

            payload = None
            pose = _estimate_pose(cv2, board, corners, ids, camera_matrix, dist_coeffs)
            if pose is not None:
                rvec, tvec, markers_used = pose
                if markers_used >= args.min_markers_for_pose:
                    reproj = _reprojection_error_px(
                        cv2, board_points, corners, ids, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    _draw_axes(cv2, annotated, camera_matrix, dist_coeffs, rvec, tvec, args.axis_length_m)
                    payload = _pose_payload(
                        cv2,
                        serial=active_serial,
                        frame_timestamp_ms=color_frame.get_timestamp(),
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                        rvec=rvec,
                        tvec=tvec,
                        markers_used=markers_used,
                        reprojection_error_px=reproj,
                    )
                    valid_frame_count += 1
                    if args.output_json is not None:
                        args.output_json.parent.mkdir(parents=True, exist_ok=True)
                        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    if args.jsonl is not None:
                        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                        with args.jsonl.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(payload) + "\n")
                    if args.print_every > 0 and valid_frame_count % args.print_every == 0:
                        t = payload["t_board_to_camera_m"]
                        print(
                            "[ARUCO] "
                            f"markers={markers_used:02d} "
                            f"t=[{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}] m "
                            f"reproj={reproj if reproj is not None else float('nan'):.2f}px"
                        )

            detected_count = 0 if ids is None else len(ids)
            lines = [
                f"Detected markers: {detected_count}",
                (
                    f"No board pose (need {args.min_markers_for_pose})"
                    if payload is None
                    else f"Used: {payload['markers_used']}  z: {payload['t_board_to_camera_m'][2]:.3f} m"
                ),
            ]
            if payload is not None and payload["reprojection_error_px"] is not None:
                lines.append(f"Reproj: {payload['reprojection_error_px']:.2f} px")
            _draw_text(annotated, cv2, lines)

            if args.no_window:
                continue
            cv2.imshow("RealSense ArUco board pose", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and args.snapshot_dir is not None:
                args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                stem = time.strftime("aruco_%Y%m%d_%H%M%S")
                image_path = args.snapshot_dir / f"{stem}.png"
                cv2.imwrite(str(image_path), annotated)
                if payload is not None:
                    (args.snapshot_dir / f"{stem}.json").write_text(
                        json.dumps(payload, indent=2), encoding="utf-8"
                    )
                print(f"[ARUCO] saved {image_path}")
    finally:
        pipeline.stop()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
