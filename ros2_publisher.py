"""
ROS2 publisher for cube coordinates.
Publishes to /rgbw_cube_detection/cubes as std_msgs/String (JSON).

Requires: source /opt/ros/<distro>/setup.bash before running.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mapper import MappingState

_ros2_available = False
_rclpy = None
_std_msgs = None

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    _ros2_available = True
except ImportError:
    pass


try:
    from config import ROS2_CUBES_TOPIC
    DEFAULT_TOPIC = ROS2_CUBES_TOPIC
except ImportError:
    DEFAULT_TOPIC = "/rgbw_cube_detection/cubes"
DEFAULT_FRAME_ID = "zed_world"
PUB_RATE_HZ = 10.0  # Throttle publishes to avoid flooding


def _cube_map_to_json(state: "MappingState") -> str:
    """Serialize cube map to JSON for ROS2 message."""
    cubes = [
        {
            "id": c.id,
            "color": c.color,
            "position": c.position,
        }
        for c in state.cube_map.values()
    ]
    return json.dumps({"header": {"frame_id": DEFAULT_FRAME_ID}, "cubes": cubes})


class CubeMapPublisher:
    """ROS2 publisher for the cube map. Optional - app runs without ROS2."""

    def __init__(self, topic: str = DEFAULT_TOPIC) -> None:
        self._topic = topic
        self._node: Node | None = None
        self._publisher = None
        self._last_pub_time = 0.0
        self._enabled = False

        if not _ros2_available:
            print("ROS2 not available (rclpy not found). Cube coordinates will not be published.")
            return

        try:
            rclpy.init()
            self._node = Node("rgbw_cube_detection_publisher")
            self._publisher = self._node.create_publisher(String, topic, 10)
            self._enabled = True
            print(f"ROS2: Publishing cube map to {topic}")
        except Exception as e:
            print(f"ROS2 init failed: {e}. Cube coordinates will not be published.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def publish(self, state: "MappingState") -> None:
        """Publish cube map (throttled to PUB_RATE_HZ)."""
        if not self._enabled or self._node is None or self._publisher is None:
            return
        now = time.monotonic()
        if now - self._last_pub_time < 1.0 / PUB_RATE_HZ:
            return
        self._last_pub_time = now

        msg = String()
        msg.data = _cube_map_to_json(state)
        self._publisher.publish(msg)
        rclpy.spin_once(self._node, timeout_sec=0)

    def shutdown(self) -> None:
        """Clean up ROS2 resources."""
        if not _ros2_available:
            return
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        try:
            rclpy.shutdown()
        except Exception:
            pass
