# OpenCV ArUco Board Pose

Live pose estimation for a RealSense D405 looking at an ArUco GridBoard.

Default board geometry:

- 6 marker columns x 8 marker rows
- 32 mm square pitch
- 24 mm black marker side
- 8 mm marker separation, computed as `32 - 24`

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.opencv.aruco_board_pose_realsense --serial 260322277506 --dictionary DICT_ARUCO_MIP_36H12
```

If more than one RealSense is connected:

```powershell
.\.venv\Scripts\python.exe -m scripts.opencv.aruco_board_pose_realsense --list-devices
.\.venv\Scripts\python.exe -m scripts.opencv.aruco_board_pose_realsense --serial YOUR_D405_SERIAL --dictionary DICT_ARUCO_MIP_36H12
```

Save the latest pose continuously:

```powershell
.\.venv\Scripts\python.exe -m scripts.opencv.aruco_board_pose_realsense --serial 260322277506 --dictionary DICT_ARUCO_MIP_36H12 --output-json data/aruco/latest_pose.json
```

Short headless smoke test:

```powershell
.\.venv\Scripts\python.exe -m scripts.opencv.aruco_board_pose_realsense --serial 260322277506 --dictionary DICT_ARUCO_MIP_36H12 --no-window --duration-s 3 --print-every 1
```

Find the matching dictionary for an unknown board:

```powershell
.\.venv\Scripts\python.exe -m scripts.opencv.aruco_board_pose_realsense --serial 260322277506 --scan-dictionaries --no-window
```

Print marker IDs to verify the board layout:

```powershell
.\.venv\Scripts\python.exe -m scripts.opencv.aruco_board_pose_realsense --serial 260322277506 --dictionary DICT_ARUCO_MIP_36H12 --no-window --duration-s 2 --print-detected-ids --min-markers-for-pose 99
```

The output JSON contains `T_board_to_camera`, `R_board_to_camera`, and
`t_board_to_camera_m`. This is the pose of the board frame in the RealSense color
camera optical frame, using OpenCV convention: x right, y down, z forward.

Important: the `--dictionary` and `--first-marker-id` must match the board you
printed. If detection works but the full board pose is unstable, try the exact
dictionary used to generate the board, for example `DICT_5X5_100` or
`DICT_6X6_250`. This board appears to use `DICT_ARUCO_MIP_36H12`.

If `cv2.aruco` is missing, install an OpenCV build with ArUco support, usually
`opencv-contrib-python`, in the environment used to run this script.
