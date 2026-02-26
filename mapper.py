"""
Main mapping loop: grab frames, detect cubes, ingest into ZED, retrieve 3D objects,
and maintain a global cube map in world coordinates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import pyzed.sl as sl

from camera import (
    create_zed_camera,
    get_object_detection_runtime_params,
    get_runtime_params,
)
from config import COLOR_LABELS
from cube_detector import detect as detect_cubes


@dataclass
class CubeEntry:
    """A cube in the global map."""

    id: int
    color: str
    position: list[float]
    last_seen: float
    tracking_state: str = "OK"


@dataclass
class MappingState:
    """State for the cube mapping session."""

    cube_map: dict[int, CubeEntry] = field(default_factory=dict)
    camera_trajectory: list[list[float]] = field(default_factory=list)


def detections_to_custom_boxes(detections: list[dict]) -> list:
    """Convert our detections to sl.CustomBoxObjectData for ZED ingest."""
    objects_in = []
    for d in detections:
        tmp = sl.CustomBoxObjectData()
        tmp.unique_object_id = sl.generate_unique_id()
        tmp.probability = d["probability"]
        tmp.label = d["label"]
        tmp.bounding_box_2d = d["bbox"]
        tmp.is_grounded = True
        tmp.is_static = True  # Cubes on sand don't move
        objects_in.append(tmp)
    return objects_in


def run_mapping_step(
    zed: sl.Camera,
    left_image: sl.Mat,
    state: MappingState,
    runtime_params: sl.RuntimeParameters,
    obj_runtime_params: sl.ObjectDetectionRuntimeParameters,
) -> tuple[np.ndarray | None, sl.Objects | None, bool]:
    """
    Run one step of the mapping pipeline.

    Returns:
        (image_np, objects, success)
    """
    err = zed.grab(runtime_params)
    if err != sl.ERROR_CODE.SUCCESS:
        return None, None, False

    # Retrieve left image
    zed.retrieve_image(left_image, sl.VIEW.LEFT)

    # Convert to numpy BGR for our detector (ZED returns BGRA)
    img_data = left_image.get_data()
    if img_data is not None:
        if len(img_data.shape) == 3 and img_data.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        else:
            img_bgr = img_data.copy()
    else:
        return None, None, False

    # Run color-based cube detection
    detections = detect_cubes(img_bgr)

    # Ingest into ZED for 3D + tracking
    if detections:
        objects_in = detections_to_custom_boxes(detections)
        zed.ingest_custom_box_objects(objects_in)

    # Retrieve 3D objects (in WORLD frame via runtime_params)
    objects = sl.Objects()
    zed.retrieve_objects(objects, obj_runtime_params)

    # Update cube map from retrieved objects
    for obj in objects.object_list:
        if obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
            continue
        pos = obj.position
        pos_arr = np.array(pos) if hasattr(pos, "__len__") else np.array([pos[0], pos[1], pos[2]])
        pos_list = [float(pos_arr[0]), float(pos_arr[1]), float(pos_arr[2])]
        if not np.all(np.isfinite(pos_arr)):
            continue  # Skip invalid (NaN/Inf) positions
        color_name = (
            COLOR_LABELS[obj.raw_label]
            if 0 <= obj.raw_label < len(COLOR_LABELS)
            else "unknown"
        )
        state.cube_map[obj.id] = CubeEntry(
            id=obj.id,
            color=color_name,
            position=pos_list,
            last_seen=time.time(),
            tracking_state="OK",
        )

    # Get camera pose for trajectory
    pose = sl.Pose()
    zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
    py_translation = sl.Translation()
    trans = pose.get_translation(py_translation)
    trans_vals = trans.get() if hasattr(trans, "get") else [trans[0], trans[1], trans[2]]
    cam_pos = [float(trans_vals[0]), float(trans_vals[1]), float(trans_vals[2])]
    if np.all(np.isfinite(cam_pos)):
        state.camera_trajectory.append(cam_pos)

    return img_bgr, objects, True
