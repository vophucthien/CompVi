"""Lane detection using Canny edge detection and Hough transform."""

from pathlib import Path

import cv2
import numpy as np


IMAGE_DIR = Path(__file__).resolve().parent / "Images"
OUTPUT_DIR = Path(__file__).resolve().parent / "Outputs"
DEBUG_DIR = Path(__file__).resolve().parent / "Debug"
SAVE_DEBUG_IMAGES = False
DRAW_LANE_CENTER = True

# Parameter set used for all input images.
# These values were tuned once and reused for every frame.
GAUSSIAN_KERNEL = (3, 3)
CANNY_LOW_THRESHOLD = 95
CANNY_HIGH_THRESHOLD = 285
HOUGH_RHO = 1.5
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 55
HOUGH_MIN_LINE_LENGTH = 28
HOUGH_MAX_LINE_GAP = 38
MIN_ABS_SLOPE = 0.25
MAX_ABS_SLOPE = 5.0
MIN_VERTICAL_DELTA = 10
LEFT_MAX_MID_X_RATIO = 0.45
RIGHT_MIN_MID_X_RATIO = 0.55
LEFT_MAX_X_MIN_RATIO = 0.40
RIGHT_MIN_X_MAX_RATIO = 0.68
ROI_LEFT_TOP_Y_RATIO = 0.80
ROI_TOP_LEFT_X_RATIO = 0.30
ROI_TOP_Y_RATIO = 0.50
ROI_TOP_RIGHT_X_RATIO = 0.60
ROI_RIGHT_TOP_Y_RATIO = 0.75
LINE_FIT_TOP_Y_RATIO = 0.50
CENTER_TOP_SAMPLE_Y_RATIO = 0.56
CENTER_BOTTOM_SAMPLE_Y_RATIO = 0.88


def apply_roi_mask(image, vertices):
    # Keep only the polygon area (road region) and suppress the rest.
    mask = np.zeros_like(image)
    fill_value = (255,) * image.shape[2] if image.ndim == 3 else 255
    cv2.fillPoly(mask, vertices, fill_value)
    return cv2.bitwise_and(image, mask)


def build_roi_vertices(width, height):
    # ROI polygon defined by width/height ratios so it scales with image size.
    return np.array(
        [[
            (0, height - 1),
            (0, int(ROI_LEFT_TOP_Y_RATIO * height)),
            (int(ROI_TOP_LEFT_X_RATIO * width), int(ROI_TOP_Y_RATIO * height)),
            (int(ROI_TOP_RIGHT_X_RATIO * width), int(ROI_TOP_Y_RATIO * height)),
            (width - 1, int(ROI_RIGHT_TOP_Y_RATIO * height)),
            (width - 1, height - 1),
        ]],
        dtype=np.int32,
    )


def detect_edges(image):
    # Standard preprocessing: grayscale -> blur -> Canny edges.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, GAUSSIAN_KERNEL, 0)
    edge_image = cv2.Canny(blurred_image, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    return gray_image, edge_image


def detect_hough_segments(masked_edges):
    # Probabilistic Hough Transform returns candidate line segments.
    return cv2.HoughLinesP(
        masked_edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )


def collect_drawable_segments(lines, image_shape):
    # Filter Hough segments into likely left and right lane boundaries.
    if lines is None:
        return [], []

    _, width = image_shape[:2]
    left_segments = []
    right_segments = []

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, line)

        if x1 == x2 or abs(y2 - y1) < MIN_VERTICAL_DELTA:
            continue

        slope = (y2 - y1) / (x2 - x1)
        midpoint_x = (x1 + x2) / 2
        min_x = min(x1, x2)
        max_x = max(x1, x2)

        if (
            -MAX_ABS_SLOPE <= slope <= -MIN_ABS_SLOPE
            and midpoint_x < LEFT_MAX_MID_X_RATIO * width
            and min_x < LEFT_MAX_X_MIN_RATIO * width
        ):
            left_segments.append((x1, y1, x2, y2))
        elif (
            MIN_ABS_SLOPE <= slope <= 0.60
            and midpoint_x > RIGHT_MIN_MID_X_RATIO * width
            and max_x > RIGHT_MIN_X_MAX_RATIO * width
        ):
            right_segments.append((x1, y1, x2, y2))

    return left_segments, right_segments


def fit_lane_line_unclipped(segments, image_shape, top_y_ratio=LINE_FIT_TOP_Y_RATIO):
    # Fit one representative line from many segments using weighted least squares.
    # Keep float coordinates here to avoid bias from early clipping.
    if not segments:
        return None

    height, _ = image_shape[:2]
    y_bottom = height - 1
    y_top = int(top_y_ratio * height)

    xs = []
    ys = []
    weights = []
    for x1, y1, x2, y2 in segments:
        length = float(np.hypot(x2 - x1, y2 - y1))
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        weights.extend([length, length])

    slope_xy, intercept_xy = np.polyfit(ys, xs, 1, w=weights)
    x_bottom = float(slope_xy * y_bottom + intercept_xy)
    x_top = float(slope_xy * y_top + intercept_xy)
    return (x_bottom, y_bottom, x_top, y_top)


def x_at_y_float(line, y_value):
    # Compute x-coordinate where a line intersects a chosen y row.
    x1, y1, x2, y2 = line
    if y1 == y2:
        return float(x1)
    return float(x1 + (y_value - y1) * (x2 - x1) / (y2 - y1))


def compute_lane_center_line(image, left_segments, right_segments):
    # Build centerline by midpoint-of-boundaries at two horizontal sample rows.
    left_line = fit_lane_line_unclipped(left_segments, image.shape)
    right_line = fit_lane_line_unclipped(right_segments, image.shape)
    if left_line is None or right_line is None:
        return None

    height, width = image.shape[:2]
    # Sample two scanlines and connect their lane midpoints.
    y_top = int(CENTER_TOP_SAMPLE_Y_RATIO * height)
    y_bottom = int(CENTER_BOTTOM_SAMPLE_Y_RATIO * height)

    left_top_x = x_at_y_float(left_line, y_top)
    right_top_x = x_at_y_float(right_line, y_top)
    left_bottom_x = x_at_y_float(left_line, y_bottom)
    right_bottom_x = x_at_y_float(right_line, y_bottom)

    if right_top_x <= left_top_x or right_bottom_x <= left_bottom_x:
        return None

    top_center_x = int(np.clip((left_top_x + right_top_x) / 2, 0, width - 1))
    bottom_center_x = int(np.clip((left_bottom_x + right_bottom_x) / 2, 0, width - 1))
    return (bottom_center_x, y_bottom, top_center_x, y_top)


def draw_lane_lines(image, left_segments, right_segments, center_line=None, color=(0, 0, 255), thickness=5):
    # Draw detected lane boundaries and optional centerline on the original image.
    line_image = np.zeros_like(image)

    for segments in (left_segments, right_segments):
        for x1, y1, x2, y2 in segments:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    output = cv2.addWeighted(image, 1.0, line_image, 1.0, 0)
    if center_line is not None:
        x1, y1, x2, y2 = center_line
        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 255), 4)

    return output


def save_debug_images(base_name, gray_image, edges, masked_edges, left_segments, right_segments, center_line, original):
    # Optional intermediate outputs for report screenshots and parameter debugging.
    if not SAVE_DEBUG_IMAGES:
        return

    DEBUG_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(DEBUG_DIR / f"01_gray_{base_name}"), gray_image)
    cv2.imwrite(str(DEBUG_DIR / f"02_canny_{base_name}"), edges)
    cv2.imwrite(str(DEBUG_DIR / f"03_masked_edges_{base_name}"), masked_edges)

    debug_lines = np.zeros_like(original)
    for segments in (left_segments, right_segments):
        for x1, y1, x2, y2 in segments:
            cv2.line(debug_lines, (x1, y1), (x2, y2), (0, 0, 255), 5)
    if center_line is not None:
        x1, y1, x2, y2 = center_line
        cv2.line(debug_lines, (x1, y1), (x2, y2), (255, 0, 255), 4)
    cv2.imwrite(str(DEBUG_DIR / f"04_lane_lines_{base_name}"), debug_lines)


def process_image(image_path):
    # Full pipeline for one image.
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray_image, edges = detect_edges(image)
    height, width = image.shape[:2]
    vertices = build_roi_vertices(width, height)
    masked_edges = apply_roi_mask(edges, vertices)

    lines = detect_hough_segments(masked_edges)
    left_segments, right_segments = collect_drawable_segments(lines, image.shape)
    center_line = compute_lane_center_line(image, left_segments, right_segments) if DRAW_LANE_CENTER else None
    result = draw_lane_lines(image, left_segments, right_segments, center_line)

    save_debug_images(
        image_path.name,
        gray_image,
        edges,
        masked_edges,
        left_segments,
        right_segments,
        center_line,
        image,
    )

    return result, len(left_segments) + len(right_segments), center_line is not None


def main():
    # Batch-process all supported images in IMAGE_DIR with one shared pipeline.
    image_paths = sorted(
        [
            *IMAGE_DIR.glob("*.jpg"),
            *IMAGE_DIR.glob("*.jpeg"),
            *IMAGE_DIR.glob("*.png"),
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Lane detection parameters")
    print(f"  Gaussian kernel: {GAUSSIAN_KERNEL}")
    print(f"  Canny thresholds: ({CANNY_LOW_THRESHOLD}, {CANNY_HIGH_THRESHOLD})")
    print(
        "  Hough parameters: "
        f"rho={HOUGH_RHO}, "
        f"theta={HOUGH_THETA:.5f}, "
        f"threshold={HOUGH_THRESHOLD}, "
        f"minLineLength={HOUGH_MIN_LINE_LENGTH}, "
        f"maxLineGap={HOUGH_MAX_LINE_GAP}"
    )
    print()

    for image_path in image_paths:
        result, segment_count, center_drawn = process_image(image_path)
        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), result)
        print(
            f"{image_path.name}: "
            f"drawn segments={segment_count}, "
            f"center line={'yes' if center_drawn else 'no'}, "
            f"saved={output_path.name}"
        )


if __name__ == "__main__":
    main()
