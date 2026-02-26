#!/usr/bin/env python3
"""Subscribe and print full cube map JSON (no truncation)."""
import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    rclpy.init()
    node = Node("echo_cubes")

    def callback(msg):
        try:
            data = json.loads(msg.data)
            print(json.dumps(data, indent=2))
            print("---")
        except json.JSONDecodeError:
            print(msg.data)

    node.create_subscription(String, "/rgbw_cube_detection/cubes", callback, 10)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
