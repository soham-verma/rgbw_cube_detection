"""
ZED 2i RGB Cube Mapper - Entry point.
Live view with annotated camera feed and top-down map. Press 's' to save, 'q' or ESC to quit.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

from camera import (
    create_zed_camera,
    get_object_detection_runtime_params,
    get_runtime_params,
)
from config import MAP_SCALE, MAP_SIZE
from mapper import MappingState, run_mapping_step
from ros2_publisher import CubeMapPublisher


def _is_valid_pos(pos: list[float]) -> bool:
    """Check that position has no NaN or Inf."""
    return all(np.isfinite(pos))


def draw_annotated_frame(image: np.ndarray, objects: sl.Objects | None) -> np.ndarray:
    """Draw bounding boxes and labels on the camera image."""
    out = image.copy()
    if objects is None:
        return out

    for obj in objects.object_list:
        if obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
            continue
        bbox = obj.bounding_box_2d
        pts = None
        if bbox is not None:
            bbox_np = np.array(bbox, dtype=np.int32)
            if bbox_np.size >= 8:
                pts = bbox_np.reshape((-1, 2))
        if pts is not None and len(pts) >= 4:
            cv2.polylines(out, [pts], True, (0, 255, 0), 2)
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        else:
            cx, cy = 0, 0
        pos = obj.position
        pos_arr = np.array(pos) if hasattr(pos, "__len__") else np.array([pos[0], pos[1], pos[2]])
        dist = float(np.sqrt(np.sum(pos_arr**2)))
        label = f"#{obj.id} {dist:.2f}m"
        cv2.putText(out, label, (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def draw_top_down_map(state: MappingState) -> np.ndarray:
    """Draw top-down 2D map (X-Z plane, Y up)."""
    map_img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    map_img[:] = (40, 40, 40)

    center = MAP_SIZE // 2
    # Grid
    for i in range(-5, 6):
        x = center + i * MAP_SCALE
        if 0 <= x < MAP_SIZE:
            cv2.line(map_img, (x, 0), (x, MAP_SIZE), (60, 60, 60), 1)
        z = center + i * MAP_SCALE
        if 0 <= z < MAP_SIZE:
            cv2.line(map_img, (0, z), (MAP_SIZE, z), (60, 60, 60), 1)

    # Origin
    cv2.circle(map_img, (center, center), 4, (128, 128, 128), -1)
    cv2.putText(map_img, "0", (center + 8, center - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "white": (200, 200, 200),
    }

    # Camera trajectory
    for cam_pos in state.camera_trajectory:
        if not _is_valid_pos(cam_pos):
            continue
        mx = int(center + cam_pos[0] * MAP_SCALE)
        mz = int(center - cam_pos[2] * MAP_SCALE)  # Flip Z for display
        if 0 <= mx < MAP_SIZE and 0 <= mz < MAP_SIZE:
            cv2.circle(map_img, (mx, mz), 1, (100, 200, 255), -1)

    # Cubes
    for cube in state.cube_map.values():
        if not _is_valid_pos(cube.position):
            continue
        x, y, z = cube.position
        mx = int(center + x * MAP_SCALE)
        mz = int(center - z * MAP_SCALE)
        if 0 <= mx < MAP_SIZE and 0 <= mz < MAP_SIZE:
            col = color_map.get(cube.color, (255, 255, 255))
            cv2.circle(map_img, (mx, mz), 8, col, -1)
            cv2.circle(map_img, (mx, mz), 8, (255, 255, 255), 1)
            cv2.putText(map_img, cube.color[0], (mx - 4, mz + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Current camera position (last in trajectory)
    if state.camera_trajectory:
        cam_pos = state.camera_trajectory[-1]
        if _is_valid_pos(cam_pos):
            mx = int(center + cam_pos[0] * MAP_SCALE)
            mz = int(center - cam_pos[2] * MAP_SCALE)
            if 0 <= mx < MAP_SIZE and 0 <= mz < MAP_SIZE:
                cv2.circle(map_img, (mx, mz), 12, (0, 255, 255), 2)

    return map_img


def save_map_json(state: MappingState, filepath: Path) -> None:
    """Save cube map and camera trajectory to JSON."""
    cubes = [
        {
            "id": c.id,
            "color": c.color,
            "position": c.position,
            "last_seen": datetime.fromtimestamp(c.last_seen).isoformat(),
        }
        for c in state.cube_map.values()
    ]
    data = {
        "cubes": cubes,
        "camera_trajectory": state.camera_trajectory,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved map to {filepath}")


def main() -> int:
    print("Opening ZED 2i camera...")
    try:
        zed = create_zed_camera()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    runtime_params = get_runtime_params()
    obj_runtime_params = get_object_detection_runtime_params()
    left_image = sl.Mat()
    state = MappingState()
    ros2_pub = CubeMapPublisher()

    print("Running. Press 's' to save map, 'q' or ESC to quit.")
    print("Move the camera around the sand to detect and map RGB cubes.")

    while True:
        img_bgr, objects, success = run_mapping_step(
            zed, left_image, state, runtime_params, obj_runtime_params
        )
        if not success or img_bgr is None:
            continue

        ros2_pub.publish(state)

        annotated = draw_annotated_frame(img_bgr, objects)
        map_img = draw_top_down_map(state)

        # Resize for display if needed
        h, w = annotated.shape[:2]
        max_h = 720
        if h > max_h:
            scale = max_h / h
            annotated = cv2.resize(annotated, (int(w * scale), max_h))
        map_img = cv2.resize(map_img, (annotated.shape[1], annotated.shape[0]))

        combined = np.hstack([annotated, map_img])
        cv2.putText(
            combined,
            "Camera | Map | 's'=save 'q'=quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.imshow("ZED RGB Cube Mapper", combined)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("s"):
            out_path = Path("cube_map.json")
            save_map_json(state, out_path)

    ros2_pub.shutdown()
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
