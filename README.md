# rgbw_cube_detection

HSV-based RGB(W) cube detection for a ZED 2i stream.

This repo contains two main ways to run detection:

- `main.py`: **Direct ZED SDK** (`pyzed`) pipeline (opens the camera itself).  
  Use this when you are *not* running `zed_wrapper`.
- `ros2_cube_detector_node.py`: **ROS2-compatible** pipeline that subscribes to `zed_wrapper` topics (does **not** open the camera).  
  Use this when `zed_wrapper` is already running and owns the camera.

## ROS2: Run cube detection alongside `zed_wrapper`

### 1) Start the camera driver (Terminal 1)

```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

### 2) Run the cube detector node (Terminal 2)

From this repo folder:

```bash
python3 ros2_cube_detector_node.py
```

- **Console output**: prints per-cube XYZ in **camera frame** (meters) when detections are present.
- **Live view**: shows an OpenCV window named **`ZED Cube Detector`** with bounding boxes and labels.
  - Press **`q`** in the window to quit.

#### Headless / no GUI

If you are on SSH or you don’t have a working GUI backend for OpenCV, run:

```bash
python3 ros2_cube_detector_node.py --no-display
```

## ROS2 topics used

Configured in `config.py`:

- **RGB**: `/zed/zed_node/rgb/color/rect/image`
- **Depth**: `/zed/zed_node/depth/depth_registered`
- **Camera info**: `/zed/zed_node/rgb/color/rect/camera_info`

## QoS note (common “topic exists but echo shows nothing” issue)

ZED image topics typically publish with **SensorData QoS** (best-effort).

If you use `ros2 topic echo` with default QoS, it may print nothing even though data is flowing.
Use:

```bash
ros2 topic hz /zed/zed_node/rgb/color/rect/image --qos-reliability best_effort
ros2 topic echo /zed/zed_node/rgb/color/rect/image --qos-reliability best_effort
```

`ros2_cube_detector_node.py` already subscribes with sensor QoS (`qos_profile_sensor_data`).

## Output coordinates

The node prints 3D points in the **camera optical frame** derived from:

- pixel center of the detected bbox
- depth at that pixel (median in a small window)
- intrinsics from `CameraInfo.K`

Unprojection:

\[
x = (u - c_x) \\cdot z / f_x,\n+\\quad y = (v - c_y) \\cdot z / f_y,\n+\\quad z = z
\]

## Troubleshooting

### “OpenCV: The function is not implemented … cvWaitKey / cvDestroyAllWindows”

Your OpenCV build has no GUI backend. Use `--no-display`, or install an OpenCV wheel with GUI support.

### NumPy 2.x vs ROS2 `cv_bridge`

This repo’s ROS2 node **does not use `cv_bridge`** on purpose, because ROS2 Humble `cv_bridge` is often built against NumPy 1.x and will crash if your environment has NumPy 2.x.

### Nothing prints, window stays black

Check that images are actually publishing:

```bash
ros2 topic hz /zed/zed_node/rgb/color/rect/image --qos-reliability best_effort
ros2 topic hz /zed/zed_node/depth/depth_registered --qos-reliability best_effort
```

If rates are zero, `zed_wrapper` is not fully streaming yet (or there is a ROS domain / networking mismatch).

