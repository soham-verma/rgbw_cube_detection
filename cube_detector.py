"""
HSV color-based cube detection for RGB cubes on sand.
Returns 2D bounding boxes for ingestion into ZED custom object detection.
"""

import cv2
import numpy as np

from config import (
    COLOR_RANGES,
    COLOR_LABELS,
    LABEL_TO_INDEX,
    MIN_CONTOUR_AREA,
    MAX_CONTOUR_AREA,
    MORPH_KERNEL_SIZE,
    ASPECT_RATIO_MIN,
    ASPECT_RATIO_MAX,
    EXTENT_MIN,
    SOLIDITY_MIN,
    CONTOUR_APPROX_EPSILON,
)


def _is_cube_shape(contour: np.ndarray, x: int, y: int, w: int, h: int, area: float) -> bool:
    """Check if contour has cube-like shape (roughly square face)."""
    if w <= 0 or h <= 0:
        return False
    aspect = w / h
    if aspect < ASPECT_RATIO_MIN or aspect > ASPECT_RATIO_MAX:
        return False
    rect_area = w * h
    if rect_area <= 0:
        return False
    extent = area / rect_area
    if extent < EXTENT_MIN:
        return False
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False
    solidity = area / hull_area
    if solidity < SOLIDITY_MIN:
        return False
    # Approx polygon: square has 4 vertices
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, CONTOUR_APPROX_EPSILON * peri, True)
    if len(approx) < 3 or len(approx) > 6:
        return False
    return True


def detect(image: np.ndarray) -> list[dict]:
    """
    Detect colored cubes in a BGR image using HSV segmentation.

    Args:
        image: BGR image from ZED left camera (numpy array)

    Returns:
        List of dicts: [{bbox, label, color_name, probability}, ...]
        bbox: list of 4 corners [[x,y], ...] for sl.Rect, or [x, y, w, h]
    """
    if image is None or image.size == 0:
        return []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detections = []

    for color_name, ranges in COLOR_RANGES.items():
        mask_combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            lower_arr = np.array(lower, dtype=np.uint8)
            upper_arr = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_arr, upper_arr)
            mask_combined = cv2.bitwise_or(mask_combined, mask)

        # Morphological ops to clean noise
        kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if not _is_cube_shape(contour, x, y, w, h, area):
                continue
            # ZED expects bounding_box_2d as 4 points: top_left, top_right, bottom_right, bottom_left (clockwise)
            bbox_corners = np.array(
                [
                    [x, y],           # top_left (A)
                    [x + w, y],       # top_right (B)
                    [x + w, y + h],   # bottom_right (C)
                    [x, y + h],       # bottom_left (D)
                ],
                dtype=np.int32,
            )
            # Simple confidence from area (larger = more likely real)
            probability = min(0.99, 0.5 + (area / 1000) * 0.1)

            detections.append({
                "bbox": bbox_corners,
                "bbox_xywh": (x, y, w, h),
                "label": LABEL_TO_INDEX[color_name],
                "color_name": color_name,
                "probability": float(probability),
            })

    return detections
