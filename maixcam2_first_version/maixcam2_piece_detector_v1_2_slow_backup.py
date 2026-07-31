#!/usr/bin/env python3
"""
MaixCAM 2 first-version detector for the 2026 electronic-design puzzle task.

What this version does:
  * reads the camera through the official MaixPy API;
  * rectifies an A4 sheet using a fixed on-screen four-corner guide;
  * detects up to four bright puzzle pieces in the upper half;
  * fits/refines 3-to-5-edge polygons;
  * shows piece ID, edge count, area and edge lengths;
  * supports touch buttons for freeze/resume and saving debug images.

Scene requirement:
  * camera fixed vertically above the sheet;
  * dark matte A4 working sheet, bright/white covered puzzle pieces;
  * align the four A4 corners with the green guide before detection.

No Shapely, SciPy or PyQt is required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time as pytime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# -------------------------- adjustable parameters ---------------------------

APP_VERSION = "1.2"
CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 1440
CAMERA_FPS = 30

# Mount the camera body 90 degrees clockwise relative to its previous
# landscape orientation. Rotating the sensor frame clockwise restores an
# upright 1440x2560 portrait image whose long side follows the A4 long side.
CAMERA_ROTATION = cv2.ROTATE_90_CLOCKWISE

# Rectified A4: 60 pixels per centimetre. This preserves substantially more
# detail from the portrait-mounted 2560x1440 sensor.
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
PX_PER_CM = 60.0
RECTIFIED_WIDTH = int(round(A4_WIDTH_CM * PX_PER_CM))
RECTIFIED_HEIGHT = int(round(A4_HEIGHT_CM * PX_PER_CM))

# The A4 guide occupies this fraction of camera image height.
A4_GUIDE_HEIGHT_RATIO = 0.90

# Pieces are initially in the upper half. Avoid the paper border and separator.
WORK_X0 = 0.02
WORK_X1 = 0.98
WORK_Y0 = 0.02
WORK_Y1 = 0.485

# Segmentation automatically supports bright pieces on a dark sheet and
# darker test pieces on a white sheet.
MIN_BRIGHT_THRESHOLD = 125
MAX_BRIGHT_THRESHOLD = 235
MIN_PIECE_AREA_CM2 = 2.5
MAX_PIECE_AREA_CM2 = 65.0
MAX_PIECES = 4

# Polygon fitting. Raise EPSILON_RATIO if a straight edge is split by noise;
# lower it if a real corner is being removed.
EPSILON_RATIO = 0.012
MIN_LEGAL_EDGE_CM = 1.0
EDGE_WARNING_MARGIN_CM = 0.15

SAVE_DIR = "/root/puzzle_piece_debug"

COLORS = [
    (0, 220, 255),
    (80, 230, 80),
    (255, 150, 40),
    (220, 80, 220),
]


@dataclass
class Piece:
    piece_id: int
    contour: np.ndarray
    polygon: np.ndarray
    centroid: tuple[float, float]
    area_cm2: float
    edge_lengths_cm: list[float]
    valid: bool

    def serializable(self) -> dict:
        return {
            "id": self.piece_id,
            "edge_count": len(self.polygon),
            "centroid_cm": [
                round(self.centroid[0] / PX_PER_CM, 3),
                round(self.centroid[1] / PX_PER_CM, 3),
            ],
            "area_cm2": round(self.area_cm2, 3),
            "edge_lengths_cm": [
                round(length, 3) for length in self.edge_lengths_cm
            ],
            "valid": self.valid,
            "vertices_cm": [
                [round(float(x) / PX_PER_CM, 3),
                 round(float(y) / PX_PER_CM, 3)]
                for x, y in self.polygon
            ],
        }


def a4_guide_corners(image_width: int, image_height: int) -> np.ndarray:
    """Return the expected A4 corner positions in the camera image."""
    guide_height = image_height * A4_GUIDE_HEIGHT_RATIO
    guide_width = guide_height * A4_WIDTH_CM / A4_HEIGHT_CM
    if guide_width > image_width * 0.96:
        guide_width = image_width * 0.96
        guide_height = guide_width * A4_HEIGHT_CM / A4_WIDTH_CM
    x0 = (image_width - guide_width) * 0.5
    y0 = (image_height - guide_height) * 0.5
    return np.asarray(
        [
            [x0, y0],
            [x0 + guide_width, y0],
            [x0 + guide_width, y0 + guide_height],
            [x0, y0 + guide_height],
        ],
        dtype=np.float32,
    )


def rectify_a4(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = a4_guide_corners(frame.shape[1], frame.shape[0])
    destination = np.asarray(
        [
            [0, 0],
            [RECTIFIED_WIDTH - 1, 0],
            [RECTIFIED_WIDTH - 1, RECTIFIED_HEIGHT - 1],
            [0, RECTIFIED_HEIGHT - 1],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    rectified = cv2.warpPerspective(
        frame,
        homography,
        (RECTIFIED_WIDTH, RECTIFIED_HEIGHT),
        flags=cv2.INTER_LINEAR,
    )
    return rectified, source


def work_bounds() -> tuple[int, int, int, int]:
    return (
        int(round(RECTIFIED_WIDTH * WORK_X0)),
        int(round(RECTIFIED_HEIGHT * WORK_Y0)),
        int(round(RECTIFIED_WIDTH * WORK_X1)),
        int(round(RECTIFIED_HEIGHT * WORK_Y1)),
    )


def distance_point_to_segment(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    vector = second - first
    denominator = float(np.dot(vector, vector))
    if denominator < 1e-9:
        return np.linalg.norm(points - first, axis=1)
    factors = np.clip(((points - first) @ vector) / denominator, 0.0, 1.0)
    projections = first + factors[:, None] * vector
    return np.linalg.norm(points - projections, axis=1)


def line_intersection(
    point_a: np.ndarray,
    direction_a: np.ndarray,
    point_b: np.ndarray,
    direction_b: np.ndarray,
) -> np.ndarray | None:
    cross = (
        float(direction_a[0] * direction_b[1])
        - float(direction_a[1] * direction_b[0])
    )
    if abs(cross) < 1e-5:
        return None
    difference = point_b - point_a
    factor = (
        float(difference[0] * direction_b[1])
        - float(difference[1] * direction_b[0])
    ) / cross
    return point_a + factor * direction_a


def refine_polygon_with_lines(
    contour: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Fit each polygon side to contour points and intersect adjacent lines."""
    contour_points = contour.reshape(-1, 2).astype(np.float32)
    vertices = polygon.reshape(-1, 2).astype(np.float32)
    side_distances = []
    for index in range(len(vertices)):
        side_distances.append(
            distance_point_to_segment(
                contour_points,
                vertices[index],
                vertices[(index + 1) % len(vertices)],
            )
        )
    assignments = np.argmin(np.stack(side_distances, axis=1), axis=1)

    fitted_lines: list[tuple[np.ndarray, np.ndarray]] = []
    for side_id in range(len(vertices)):
        points = contour_points[assignments == side_id]
        if len(points) < 6:
            return vertices
        vx, vy, x0, y0 = cv2.fitLine(
            points.reshape(-1, 1, 2),
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        ).reshape(-1)
        fitted_lines.append(
            (
                np.asarray([x0, y0], dtype=np.float32),
                np.asarray([vx, vy], dtype=np.float32),
            )
        )

    refined = []
    maximum_shift = 0.06 * cv2.arcLength(contour, True)
    for vertex_id in range(len(vertices)):
        previous = fitted_lines[(vertex_id - 1) % len(vertices)]
        current = fitted_lines[vertex_id]
        intersection = line_intersection(
            previous[0],
            previous[1],
            current[0],
            current[1],
        )
        if intersection is None:
            return vertices
        if float(np.linalg.norm(intersection - vertices[vertex_id])) > maximum_shift:
            return vertices
        refined.append(intersection)

    result = np.asarray(refined, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        return vertices
    return result


def polygon_contour_error(
    contour: np.ndarray,
    polygon: np.ndarray,
) -> float:
    points = contour.reshape(-1, 2).astype(np.float32)
    distances = []
    for index in range(len(polygon)):
        distances.append(
            distance_point_to_segment(
                points,
                polygon[index],
                polygon[(index + 1) % len(polygon)],
            )
        )
    return float(np.mean(np.min(np.stack(distances, axis=1), axis=1)))


def remove_impossible_short_edges(
    contour: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Remove burr/shadow corners that create an edge illegal for this task."""
    vertices = polygon.reshape(-1, 2).astype(np.float32)
    minimum_pixels = (
        MIN_LEGAL_EDGE_CM - EDGE_WARNING_MARGIN_CM
    ) * PX_PER_CM
    while len(vertices) > 3:
        lengths = np.asarray(
            [
                np.linalg.norm(
                    vertices[(index + 1) % len(vertices)] - vertices[index]
                )
                for index in range(len(vertices))
            ]
        )
        short_id = int(np.argmin(lengths))
        if lengths[short_id] >= minimum_pixels:
            break

        # A short fitted side is impossible because the task guarantees every
        # physical edge is at least 1 cm. It normally comes from a cardboard
        # burr or the thin side-wall shadow. Try removing either endpoint and
        # retain the candidate that best follows the original contour.
        first = np.delete(vertices, short_id, axis=0)
        second = np.delete(
            vertices,
            (short_id + 1) % len(vertices),
            axis=0,
        )
        vertices = min(
            (first, second),
            key=lambda candidate: polygon_contour_error(contour, candidate),
        )
    return vertices


def fit_polygon(contour: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    # Use one predictable epsilon first. Selecting the smallest area error
    # from many epsilons tends to preserve tiny raster/noise corners and can
    # turn a real triangle into a quadrilateral.
    ratios = [EPSILON_RATIO]
    ratios.extend(
        EPSILON_RATIO * factor
        for factor in (0.85, 1.15, 0.70, 1.30, 0.55, 1.50, 1.80)
    )
    approximation = None
    for ratio in ratios:
        approximation = cv2.approxPolyDP(contour, ratio * perimeter, True)
        edge_count = len(approximation)
        if 3 <= edge_count <= 5:
            break
        approximation = None

    if approximation is None:
        return None
    cleaned = remove_impossible_short_edges(contour, approximation)
    refined = refine_polygon_with_lines(contour, cleaned)

    # Preserve contour order and reject a degenerate refinement.
    if abs(cv2.contourArea(refined.astype(np.float32))) < 1.0:
        return approximation.reshape(-1, 2).astype(np.float32)
    return refined


def segment_bright_pieces(work_image: np.ndarray) -> tuple[np.ndarray, int]:
    gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold, _ = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    threshold = int(
        np.clip(otsu_threshold, MIN_BRIGHT_THRESHOLD, MAX_BRIGHT_THRESHOLD)
    )
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Support both task-board arrangements:
    #   1. bright/white pieces on a dark matte A4 sheet;
    #   2. darker coloured test pieces on a white A4 sheet.
    # In the second arrangement the threshold image is mostly white and the
    # pieces are black holes. Invert it so pieces are always the white
    # foreground consumed by findContours().
    if cv2.countNonZero(binary) > binary.size * 0.55:
        binary = cv2.bitwise_not(binary)
    return binary, threshold


def polygon_edge_lengths(polygon: np.ndarray) -> list[float]:
    return [
        float(
            np.linalg.norm(
                polygon[(index + 1) % len(polygon)] - polygon[index]
            )
        ) / PX_PER_CM
        for index in range(len(polygon))
    ]


def detect_pieces(rectified: np.ndarray) -> tuple[list[Piece], np.ndarray, int]:
    x0, y0, x1, y1 = work_bounds()
    work_image = rectified[y0:y1, x0:x1]
    binary, threshold = segment_bright_pieces(work_image)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    detections = []
    minimum_area_px = MIN_PIECE_AREA_CM2 * PX_PER_CM * PX_PER_CM
    maximum_area_px = MAX_PIECE_AREA_CM2 * PX_PER_CM * PX_PER_CM
    work_height, work_width = binary.shape

    for contour in contours:
        area_px = cv2.contourArea(contour)
        if not minimum_area_px <= area_px <= maximum_area_px:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        # Reject an object clipped by the recognition-region boundary.
        if bx <= 1 or by <= 1 or bx + bw >= work_width - 1 or by + bh >= work_height - 1:
            continue

        polygon = fit_polygon(contour)
        if polygon is None:
            continue
        polygon[:, 0] += x0
        polygon[:, 1] += y0
        contour_global = contour.copy()
        contour_global[:, 0, 0] += x0
        contour_global[:, 0, 1] += y0

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            continue
        centroid = (
            moments["m10"] / moments["m00"] + x0,
            moments["m01"] / moments["m00"] + y0,
        )
        lengths = polygon_edge_lengths(polygon)
        valid = (
            3 <= len(polygon) <= 5
            and min(lengths) >= MIN_LEGAL_EDGE_CM - EDGE_WARNING_MARGIN_CM
        )
        detections.append(
            {
                "contour": contour_global,
                "polygon": polygon,
                "centroid": centroid,
                "area_cm2": area_px / (PX_PER_CM * PX_PER_CM),
                "edge_lengths_cm": lengths,
                "valid": valid,
            }
        )

    # Stable numbering: top-to-bottom, then left-to-right.
    detections.sort(
        key=lambda item: (
            round(item["centroid"][1] / (4.0 * PX_PER_CM)),
            item["centroid"][0],
        )
    )
    detections = detections[:MAX_PIECES]
    pieces = [
        Piece(piece_id=index + 1, **detection)
        for index, detection in enumerate(detections)
    ]
    return pieces, binary, threshold


def draw_detection(
    rectified: np.ndarray,
    pieces: Iterable[Piece],
    threshold: int,
) -> np.ndarray:
    canvas = rectified.copy()
    x0, y0, x1, y1 = work_bounds()
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 3)

    pieces = list(pieces)
    for piece in pieces:
        color = COLORS[(piece.piece_id - 1) % len(COLORS)]
        if not piece.valid:
            color = (0, 0, 255)
        polygon_int = np.round(piece.polygon).astype(np.int32)
        cv2.polylines(canvas, [polygon_int], True, color, 5, cv2.LINE_AA)

        for vertex_id, point in enumerate(polygon_int):
            cv2.circle(canvas, tuple(point), 8, (255, 255, 255), -1)
            cv2.circle(canvas, tuple(point), 8, color, 2)
            next_point = polygon_int[(vertex_id + 1) % len(polygon_int)]
            midpoint = ((point + next_point) * 0.5).astype(int)
            text = f"{piece.edge_lengths_cm[vertex_id]:.1f}"
            cv2.putText(
                canvas,
                text,
                tuple(midpoint),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                tuple(midpoint),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        label = (
            f"P{piece.piece_id}  E={len(piece.polygon)}  "
            f"A={piece.area_cm2:.1f}cm2"
        )
        position = (
            int(piece.centroid[0]) - 80,
            int(piece.centroid[1]),
        )
        cv2.putText(
            canvas,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    status = (
        f"v{APP_VERSION}  pieces={len(pieces)}/4  threshold={threshold}"
    )
    cv2.putText(
        canvas,
        status,
        (20, RECTIFIED_HEIGHT - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        status,
        (20, RECTIFIED_HEIGHT - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def compose_screen(
    preview_source: np.ndarray,
    mode: str,
    message: str = "",
) -> np.ndarray:
    screen = np.zeros((480, 640, 3), dtype=np.uint8)
    scale = min(640 / preview_source.shape[1], 360 / preview_source.shape[0])
    preview_size = (
        max(1, int(preview_source.shape[1] * scale)),
        max(1, int(preview_source.shape[0] * scale)),
    )
    preview = cv2.resize(
        preview_source,
        preview_size,
        interpolation=cv2.INTER_AREA,
    )
    px = (640 - preview.shape[1]) // 2
    py = (360 - preview.shape[0]) // 2
    screen[py:py + preview.shape[0], px:px + preview.shape[1]] = preview

    left_color = (0, 150, 255) if mode == "frozen" else (0, 180, 0)
    cv2.rectangle(screen, (10, 375), (310, 465), left_color, -1)
    cv2.rectangle(screen, (330, 375), (630, 465), (180, 90, 0), -1)
    left_text = {
        "align": "DETECT",
        "live": "FREEZE",
        "frozen": "RESUME",
    }[mode]
    cv2.putText(
        screen, left_text, (70, 435),
        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3, cv2.LINE_AA,
    )
    cv2.putText(
        screen, "SAVE", (420, 435),
        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3, cv2.LINE_AA,
    )
    if message:
        cv2.putText(
            screen, message[:42], (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
        )
    return screen


def save_debug(
    raw: np.ndarray,
    rectified: np.ndarray,
    binary: np.ndarray,
    annotated: np.ndarray,
    pieces: list[Piece],
    save_dir: str,
) -> str:
    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = pytime.strftime("%Y%m%d_%H%M%S")
    prefix = directory / f"pieces_{stamp}"
    cv2.imwrite(str(prefix) + "_raw.jpg", raw)
    cv2.imwrite(str(prefix) + "_rectified.jpg", rectified)
    cv2.imwrite(str(prefix) + "_binary.png", binary)
    cv2.imwrite(str(prefix) + "_result.jpg", annotated)
    Path(str(prefix) + "_result.json").write_text(
        json.dumps(
            {"piece_count": len(pieces),
             "pieces": [piece.serializable() for piece in pieces]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(prefix)


def draw_camera_guide(frame: np.ndarray) -> np.ndarray:
    guide = frame.copy()
    corners = np.round(
        a4_guide_corners(frame.shape[1], frame.shape[0])
    ).astype(np.int32)
    cv2.polylines(guide, [corners], True, (0, 255, 0), 7, cv2.LINE_AA)
    return guide


def run_pc_image(image_path: str, output_path: str) -> int:
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    rectified, _ = rectify_a4(frame)
    pieces, binary, threshold = detect_pieces(rectified)
    annotated = draw_detection(rectified, pieces, threshold)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated):
        raise RuntimeError(f"Cannot write result: {output}")
    cv2.imwrite(str(output.with_name(output.stem + "_binary.png")), binary)
    print(json.dumps(
        {"piece_count": len(pieces),
         "pieces": [piece.serializable() for piece in pieces]},
        ensure_ascii=False,
        indent=2,
    ))
    print(f"Result saved: {output.resolve()}")
    return 0


def run_maix() -> int:
    # Import only on the device so the same detector can be regression-tested
    # against still images on a PC.
    from maix import app, camera, display, image, touchscreen

    cam = camera.Camera(
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        image.Format.FMT_BGR888,
        fps=CAMERA_FPS,
    )
    cam.skip_frames(12)
    disp = display.Display()
    touch = touchscreen.TouchScreen()
    touch.clear()

    mode = "align"
    pressed_before = False
    last_touch = (0, 0)
    last_raw = None
    last_rectified = None
    last_binary = None
    last_annotated = None
    last_pieces: list[Piece] = []
    message = "Align A4 to green guide, then tap DETECT"

    while not app.need_exit():
        if mode != "frozen" or last_raw is None:
            maix_frame = cam.read()
            frame = image.image2cv(
                maix_frame,
                ensure_bgr=False,
                copy=True,
            )
            frame = cv2.rotate(frame, CAMERA_ROTATION)
            last_raw = frame
            if mode == "align":
                preview_source = draw_camera_guide(frame)
            else:
                rectified, _ = rectify_a4(frame)
                pieces, binary, threshold = detect_pieces(rectified)
                annotated = draw_detection(rectified, pieces, threshold)
                last_rectified = rectified
                last_binary = binary
                last_annotated = annotated
                last_pieces = pieces
                preview_source = annotated
        else:
            preview_source = last_annotated

        screen = compose_screen(preview_source, mode, message)
        disp.show(image.cv2image(screen, bgr=True, copy=True))

        if touch.available(0):
            x, y, pressed = touch.read()
            if pressed:
                last_touch = (x, y)
                pressed_before = True
            elif pressed_before:
                pressed_before = False
                x, y = last_touch
                if 10 <= x <= 310 and 365 <= y <= 475:
                    if mode == "align":
                        mode = "live"
                        message = "Live detection"
                    elif mode == "live":
                        mode = "frozen"
                        message = f"Frozen: {len(last_pieces)} pieces"
                        print(json.dumps(
                            {"piece_count": len(last_pieces),
                             "pieces": [
                                 piece.serializable()
                                 for piece in last_pieces
                             ]},
                            ensure_ascii=False,
                            indent=2,
                        ))
                    else:
                        mode = "live"
                        message = "Live detection"
                elif 330 <= x <= 630 and 365 <= y <= 475:
                    if last_annotated is None:
                        message = "Tap DETECT before saving"
                    else:
                        prefix = save_debug(
                            last_raw,
                            last_rectified,
                            last_binary,
                            last_annotated,
                            last_pieces,
                            SAVE_DIR,
                        )
                        message = f"Saved: {os.path.basename(prefix)}"
                        print(f"Saved debug files: {prefix}_*")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        help="PC test mode: process one camera image instead of using Maix API",
    )
    parser.add_argument(
        "--output",
        default="maix_piece_result.jpg",
        help="PC test-mode result path",
    )
    args = parser.parse_args()
    if args.image:
        return run_pc_image(args.image, args.output)
    return run_maix()


if __name__ == "__main__":
    raise SystemExit(main())
