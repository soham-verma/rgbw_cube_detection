#!/usr/bin/env python3
"""
ROS2 node that detects colored cubes from ZED wrapper topics.

Subscribes to RGB image, depth image, and camera_info from the ZED wrapper,
runs HSV-based cube detection, unprojects 2D detections to 3D using depth
and camera intrinsics, prints XYZ positions to the console, and optionally
shows a live annotated video window.

Usage:
    # Terminal 1 - ZED wrapper must be running:
    ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i

    # Terminal 2 - Run this node (with live view):
    python ros2_cube_detector_node.py

    # Without the live window:
    python ros2_cube_detector_node.py --no-display
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ZED_CAMERA_INFO_TOPIC,
    ZED_DEPTH_TOPIC,
    ZED_IMAGE_TOPIC,
)
from cube_detector import detect as detect_cubes

PRINT_RATE_HZ = 2.0
MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 10.0
DEPTH_SAMPLE_RADIUS = 3

# BGR colors for each cube color label
_COLOR_BGR = {
    "red":   (0,   0,   255),
    "green": (0,   200, 0),
    "blue":  (255, 80,  0),
    "white": (220, 220, 220),
}


def _ros_image_to_numpy(msg: Image) -> np.ndarray | None:
    """Convert a sensor_msgs/Image to a numpy array without cv_bridge."""
    enc = msg.encoding.lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("bgr8", "rgb8"):
        img = raw.reshape((msg.height, msg.width, 3))
        if enc == "rgb8":
            img = img[:, :, ::-1].copy()
        return img

    if enc in ("bgra8", "rgba8"):
        img = raw.reshape((msg.height, msg.width, 4))
        if enc == "rgba8":
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if enc == "32fc1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))

    if enc == "16uc1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))

    if enc in ("mono8", "8uc1"):
        return raw.reshape((msg.height, msg.width))

    return None


class CubeDetectorNode(Node):
    def __init__(self, show_display: bool = True) -> None:
        super().__init__("cube_detector_node")
        self._show_display = show_display
        self._camera_info: CameraInfo | None = None
        self._last_print_time = 0.0

        # ZED topics typically use SensorDataQoS (best-effort). If we subscribe
        # with the default reliable QoS, we may receive nothing.
        self.create_subscription(
            CameraInfo, ZED_CAMERA_INFO_TOPIC, self._camera_info_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, ZED_IMAGE_TOPIC, self._image_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, ZED_DEPTH_TOPIC, self._depth_cb, qos_profile_sensor_data
        )

        self._latest_rgb: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._rgb_stamp = None
        self._depth_stamp = None

        self._timer = self.create_timer(1.0 / PRINT_RATE_HZ, self._process)

        self.get_logger().info(f"Subscribing to RGB:   {ZED_IMAGE_TOPIC}")
        self.get_logger().info(f"Subscribing to Depth: {ZED_DEPTH_TOPIC}")
        self.get_logger().info(f"Subscribing to Info:  {ZED_CAMERA_INFO_TOPIC}")
        if show_display:
            self.get_logger().info("Live view enabled  (press 'q' in the window to quit)")
        self.get_logger().info("Waiting for messages...")

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _image_cb(self, msg: Image) -> None:
        img = _ros_image_to_numpy(msg)
        if img is None:
            self.get_logger().warn(f"Unsupported RGB encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return
        self._latest_rgb = img
        self._rgb_stamp = msg.header.stamp

    def _depth_cb(self, msg: Image) -> None:
        depth = _ros_image_to_numpy(msg)
        if depth is None:
            self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        self._latest_depth = depth
        self._depth_stamp = msg.header.stamp

    def _sample_depth(self, depth_img: np.ndarray, cx: int, cy: int) -> float:
        """Sample depth around (cx, cy) with a small window, returning median of valid values."""
        h, w = depth_img.shape[:2]
        r = DEPTH_SAMPLE_RADIUS
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        patch = depth_img[y0:y1, x0:x1].flatten()
        valid = patch[np.isfinite(patch) & (patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)]
        if len(valid) == 0:
            return float("nan")
        return float(np.median(valid))

    def _process(self) -> None:
        if self._latest_rgb is None or self._latest_depth is None or self._camera_info is None:
            if self._show_display:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.get_logger().info("Quit requested — shutting down.")
                    rclpy.shutdown()
            return

        rgb = self._latest_rgb
        depth = self._latest_depth

        if rgb.shape[:2] != depth.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        K = np.array(self._camera_info.k).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx_cam, cy_cam = K[0, 2], K[1, 2]

        if fx == 0 or fy == 0:
            return

        detections = detect_cubes(rgb)

        # Build annotated frame for display (always, so the window stays live)
        if self._show_display:
            frame = rgb.copy()

        now = time.monotonic()
        throttled = (now - self._last_print_time) >= 1.0 / PRINT_RATE_HZ
        if throttled:
            self._last_print_time = now

        lines: list[str] = []
        for det in detections:
            x, y, w, h = det["bbox_xywh"]
            px = x + w // 2
            py = y + h // 2

            z = self._sample_depth(depth, px, py)
            color = det["color_name"]
            bgr = _COLOR_BGR.get(color, (255, 255, 255))

            if self._show_display:
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
                if np.isfinite(z):
                    label = f"{color} {z:.2f}m"
                else:
                    label = color
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), bgr, -1)
                cv2.putText(frame, label, (x + 2, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            if not np.isfinite(z):
                continue

            x_cam = (px - cx_cam) * z / fx
            y_cam = (py - cy_cam) * z / fy

            if throttled:
                lines.append(
                    f"  {color:>6s}: x={x_cam:+.3f}  y={y_cam:+.3f}  z={z:.3f} m  "
                    f"(px {px},{py}  conf {det['probability']:.0%})"
                )

        if throttled and lines:
            self.get_logger().info("--- Detected cubes (camera frame) ---")
            for line in lines:
                self.get_logger().info(line)

        if self._show_display:
            cv2.imshow("ZED Cube Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().info("Quit requested — shutting down.")
                cv2.destroyAllWindows()
                rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="ZED cube detector ROS2 node")
    parser.add_argument("--no-display", action="store_true", help="Disable the live OpenCV window")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args if ros_args else None)
    node = CubeDetectorNode(show_display=not args.no_display)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
