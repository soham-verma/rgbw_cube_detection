"""
Configuration for RGB cube detection.
HSV ranges tuned for colored cubes on sand (beige/tan background).
Adjust these values based on your lighting and cube colors.
"""

# HSV color ranges for cube detection (OpenCV: H 0-180, S 0-255, V 0-255)
# Each entry: (color_name, (lower_hsv), (upper_hsv))
# Red wraps around hue - use two ranges
COLOR_RANGES = {
    "red": [
        ((0, 100, 100), (10, 255, 255)),   # Lower red
        ((170, 100, 100), (180, 255, 255)),  # Upper red
    ],
    "green": [
        ((35, 80, 80), (85, 255, 255)),
    ],
    "blue": [
        ((100, 80, 80), (130, 255, 255)),
    ],
    # "white": [
    #     ((0, 0, 180), (180, 50, 255)),  # Low saturation, high value
    # ],
}

# Label indices for ZED custom box objects (must match order)
COLOR_LABELS = ["red", "green", "blue", "white"]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(COLOR_LABELS)}

# Minimum contour area in pixels (filter out noise, shadows, sand texture)
MIN_CONTOUR_AREA = 100

# Maximum contour area (reject very large blobs that are not cubes)
MAX_CONTOUR_AREA = 50000

# Cube shape filters (reject non-cube RGB blobs)
# Aspect ratio w/h: cube faces are roughly square; allow 0.5-2.0 for perspective
ASPECT_RATIO_MIN = 0.5
ASPECT_RATIO_MAX = 2.0
# Extent: area / bounding_rect_area; squares fill ~0.78+ of their bbox
EXTENT_MIN = 0.6
# Solidity: area / convex_hull_area; solid cubes have high solidity
SOLIDITY_MIN = 0.7
# Approx polygon: cube face has ~4 corners (square)
CONTOUR_APPROX_EPSILON = 0.04  # relative to perimeter

# Morphological kernel size for noise cleanup
MORPH_KERNEL_SIZE = (5, 5)

# Map visualization settings
MAP_SCALE = 200  # Pixels per meter for top-down view
MAP_SIZE = 800   # Size of map panel in pixels

# ROS2 topic for publishing cube coordinates (std_msgs/String, JSON)
ROS2_CUBES_TOPIC = "/rgbw_cube_detection/cubes"

# zed_wrapper topics (for cube_detection_node when using zed_wrapper + rtabmap)
# Adjust namespace if your zed_wrapper uses different naming
ZED_IMAGE_TOPIC = "/zed/zed_node/rgb/color/rect/image"
ZED_DEPTH_TOPIC = "/zed/zed_node/depth/depth_registered"
ZED_CAMERA_INFO_TOPIC = "/zed/zed_node/rgb/color/rect/camera_info"
