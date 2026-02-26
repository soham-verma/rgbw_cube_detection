# RGBW Cube Detection

Detect and map RGB cubes on sand using a ZED 2i camera. Uses positional tracking and color-based detection to build a global map of cube locations as the camera moves.

## Features

- **Color detection** for red, green, blue, and white cubes (HSV segmentation)
- **Cube-shaped filtering** to reject non-cube RGB objects (aspect ratio, solidity, extent)
- **3D positions** in world coordinates via ZED custom object detection
- **Live top-down map** of cube locations and camera trajectory
- **ROS2 publishing** to `/rgbw_cube_detection/cubes` (JSON via std_msgs/String)
- **JSON export** on keypress

## Prerequisites

- **ZED SDK** from [stereolabs.com/developers](https://www.stereolabs.com/developers/)
- **ZED 2i** camera (or ZED 2 / ZED Mini)
- Python 3.10+

## Setup

1. Install ZED SDK and pyzed:
   ```bash
   cd /usr/local/zed/
   python3 get_python_api.py
   ```

2. Create venv and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. For ROS2 publishing (optional):
   ```bash
   source /opt/ros/humble/setup.bash  # or foxy, iron, etc.
   ```

## Usage

```bash
source venv/bin/activate
python main.py
```

- **s** – Save cube map to `cube_map.json`
- **q** or **ESC** – Quit

Move the camera around the sand to detect and map cubes. Positions are in the ZED world frame (right-handed, Y-up, meters).

## ROS2

If ROS2 is sourced, cube coordinates are published to `/rgbw_cube_detection/cubes` at ~10 Hz:

```bash
ros2 topic echo --full-length /rgbw_cube_detection/cubes
```

Message format (std_msgs/String, JSON):
```json
{
  "header": {"frame_id": "zed_world"},
  "cubes": [
    {"id": 1, "color": "red", "position": [x, y, z]},
    {"id": 2, "color": "green", "position": [x, y, z]}
  ]
}
```

## Configuration

Edit `config.py` to adjust:

- HSV color ranges
- Min/max contour area
- Cube shape filters (aspect ratio, extent, solidity)
- ROS2 topic name

## Project Structure

```
rgbw_cube_detection/
├── main.py           # Entry point, display, save
├── camera.py         # ZED init, positional tracking, object detection
├── cube_detector.py  # HSV color + cube-shape detection
├── mapper.py         # Grab loop, ingest, map update
├── ros2_publisher.py # ROS2 topic publisher
├── config.py         # HSV ranges, thresholds
├── scripts/
│   └── echo_cubes.py # ROS2 subscriber for full JSON output
└── requirements.txt
```
