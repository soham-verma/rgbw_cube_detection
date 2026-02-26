"""
ZED 2i camera setup with positional tracking and custom object detection.
"""

import pyzed.sl as sl


def create_zed_camera(resolution=sl.RESOLUTION.HD720) -> sl.Camera:
    """Create and configure ZED 2i camera with positional tracking and custom OD."""
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution
    init_params.camera_fps = 30
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.sdk_verbose = False

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")

    # Enable positional tracking (required for world-frame objects and object tracking)
    tracking_params = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError(f"Failed to enable positional tracking: {err}")

    # Enable custom object detection (CUSTOM_BOX_OBJECTS)
    detection_params = sl.ObjectDetectionParameters()
    detection_params.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    detection_params.enable_tracking = True
    detection_params.enable_segmentation = False  # We only need positions, not masks

    err = zed.enable_object_detection(detection_params)
    if err != sl.ERROR_CODE.SUCCESS:
        zed.disable_positional_tracking()
        zed.close()
        raise RuntimeError(f"Failed to enable object detection: {err}")

    return zed


def get_runtime_params() -> sl.RuntimeParameters:
    """Runtime params with WORLD reference frame for 3D positions."""
    params = sl.RuntimeParameters()
    params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    return params


def get_object_detection_runtime_params() -> sl.ObjectDetectionRuntimeParameters:
    """Object detection runtime params."""
    return sl.ObjectDetectionRuntimeParameters()
