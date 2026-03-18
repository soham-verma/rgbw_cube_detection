#!/usr/bin/env python3
"""
ROS2 node for cube detection using zed_wrapper streams.
Subscribes to ZED image, depth, and odometry from zed_wrapper.
Publishes cube positions to /rgbw_cube_detection/cubes.

Run with zed_wrapper active: the ZED camera must be opened by zed_wrapper
(for rtabmap, etc.); this node only consumes the published topics.

Usage:
  source /opt/ros/humble/setup.bash
  # Launch zed_wrapper first, then:
  cd ~/basestation/rgbw_cube_detection && python3 nodes/cube_detection_node.py
"""

from __future__ import annotations

import json
import math
import sys
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener

from config import (
    ROS2_CUBES_TOPIC,
    ZED_IMAGE_TOPIC,
    ZED_DEPTH_TOPIC,
    ZED_CAMERA_INFO_TOPIC,
)
from cube_detector import detect as detect_cubes


# Defaults from config (override via ROS params)
class CubeDetectionNode(Node):
    """Subscribe to zed_wrapper, detect cubes, publish 3D positions."""

    def __init__(self) -> None:
        super().__init__("cube_detection_node")
        self.declare_parameter("image_topic", ZED_IMAGE_TOPIC)
        self.declare_parameter("depth_topic", ZED_DEPTH_TOPIC)
        self.declare_parameter("camera_info_topic", ZED_CAMERA_INFO_TOPIC)
        self.declare_parameter("cubes_topic", ROS2_CUBES_TOPIC)
        self.declare_parameter("frame_id", "odom")  # Target frame for published cube positions
        self.declare_parameter("merge_dist_m", 0.15)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_image: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._latest_camera_info: CameraInfo | None = None
        self._cube_map: dict[int, dict] = {}
        self._next_id = 0
        self._merge_dist = self.get_parameter("merge_dist_m").value
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        image_topic = self.get_parameter("image_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        cubes_topic = self.get_parameter("cubes_topic").value

        self._sub_image = self.create_subscription(
            Image, image_topic, self._cb_image, 10
        )
        self._sub_depth = self.create_subscription(
            Image, depth_topic, self._cb_depth, 10
        )
        self._sub_camera_info = self.create_subscription(
            CameraInfo, camera_info_topic, self._cb_camera_info, 10
        )
        self._pub = self.create_publisher(String, cubes_topic, 10)
        self._timer = self.create_timer(0.1, self._timer_cb)

        self.get_logger().info(
            f"Subscribing to {image_topic}, {depth_topic}, {camera_info_topic}"
        )
        self.get_logger().info(f"Publishing to {cubes_topic}")

    def _cb_image(self, msg: Image) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                self._latest_image = img
        except Exception as e:
            self.get_logger().warn(f"Image decode: {e}")

    def _cb_depth(self, msg: Image) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            with self._lock:
                self._latest_depth = depth
        except Exception as e:
            self.get_logger().warn(f"Depth decode: {e}")

    def _cb_camera_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_camera_info = msg

    def _unproject(self, u: float, v: float, z: float, K: list[float]) -> list[float]:
        """Unproject pixel (u,v) with depth z to 3D in camera optical frame."""
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return [x, y, z]

    def _transform_point(self, p: list[float], cam_frame: str, odom_frame: str) -> list[float] | None:
        """Transform point from camera optical frame to odom using TF."""
        try:
            pt = PointStamped()
            pt.header.frame_id = cam_frame
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.point.x, pt.point.y, pt.point.z = p[0], p[1], p[2]
            trans = self._tf_buffer.transform(pt, odom_frame)
            return [trans.point.x, trans.point.y, trans.point.z]
        except Exception:
            return None

    def _merge_cube(self, color: str, position: list[float]) -> int:
        """Add or merge cube into map, return id."""
        for cid, entry in list(self._cube_map.items()):
            dx = position[0] - entry["position"][0]
            dy = position[1] - entry["position"][1]
            dz = position[2] - entry["position"][2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) < self._merge_dist:
                entry["position"] = position
                entry["color"] = color
                return cid
        cid = self._next_id
        self._next_id += 1
        self._cube_map[cid] = {"id": cid, "color": color, "position": position}
        return cid

    def _timer_cb(self) -> None:
        with self._lock:
            img = self._latest_image
            depth = self._latest_depth
            cam_info = self._latest_camera_info
        if img is None or depth is None or cam_info is None:
            return

        if img.shape[:2] != depth.shape[:2]:
            return

        target_frame = self.get_parameter("frame_id").value
        cam_frame = cam_info.header.frame_id

        K = cam_info.k
        detections = detect_cubes(img)

        for d in detections:
            x, y, w, h = d["bbox_xywh"]
            cx, cy = x + w // 2, y + h // 2
            if cy < 0 or cy >= depth.shape[0] or cx < 0 or cx >= depth.shape[1]:
                continue
            z = float(np.nanmedian(depth[max(0, y) : y + h, max(0, x) : x + w]))
            if not np.isfinite(z) or z <= 0 or z > 20:
                continue
            p_cam = self._unproject(float(cx), float(cy), z, K)
            p_odom = self._transform_point(p_cam, cam_frame, target_frame)
            if p_odom is None or not all(np.isfinite(p_odom)):
                continue
            self._merge_cube(d["color_name"], p_odom)

        data = {
            "header": {"frame_id": self.get_parameter("frame_id").value},
            "cubes": list(self._cube_map.values()),
        }
        msg = String()
        msg.data = json.dumps(data)
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = CubeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
