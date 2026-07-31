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
  * approximately uniform A4 background contrasting with the pieces;
  * plain cardboard or printed playing-card pieces are supported;
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

APP_VERSION = "1.13-near-limit-edge-preserve"
CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 1440
CAMERA_FPS = 30
BUTTON_DETECT_RECT = (10, 375, 310, 465)
BUTTON_SAVE_RECT = (330, 375, 630, 465)

# Rectified A4: 40 pixels per centimetre.
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
PX_PER_CM = 40.0
RECTIFIED_WIDTH = int(round(A4_WIDTH_CM * PX_PER_CM))
RECTIFIED_HEIGHT = int(round(A4_HEIGHT_CM * PX_PER_CM))

# The A4 guide occupies this fraction of camera image height.
A4_GUIDE_HEIGHT_RATIO = 0.90

# Pieces are initially in the upper half. Avoid the paper border and separator.
WORK_X0 = 0.02
WORK_X1 = 0.98
WORK_Y0 = 0.02
WORK_Y1 = 0.50

# Segmentation automatically supports bright pieces on a dark sheet and
# darker test pieces on a white sheet.
MIN_BRIGHT_THRESHOLD = 125
MAX_BRIGHT_THRESHOLD = 235
MIN_COLOR_DISTANCE_THRESHOLD = 14
MAX_COLOR_DISTANCE_THRESHOLD = 90
MIN_PIECE_AREA_CM2 = 2.5
MAX_PIECE_AREA_CM2 = 65.0
MAX_PIECES = 4

# Polygon fitting. Raise EPSILON_RATIO if a straight edge is split by noise;
# lower it if a real corner is being removed.
EPSILON_RATIO = 0.012
MIN_LEGAL_EDGE_CM = 1.0
EDGE_WARNING_MARGIN_CM = 0.15
# A new polygon vertex must improve the mean contour fit by at least this
# amount. This suppresses threshold-sensitive bevels while preserving a real
# 1 cm edge when its two corners measurably change the outline.
VERTEX_COMPLEXITY_PENALTY_CM = 0.03
# A measured 1 cm cut can fluctuate below 1 cm after rectification and line
# fitting. Preserve a near-limit edge when both endpoints are genuine corners;
# otherwise the generic extra-vertex penalty tends to collapse it into a
# triangle.
SHORT_EDGE_EVIDENCE_MAX_CM = MIN_LEGAL_EDGE_CM + EDGE_WARNING_MARGIN_CM
SHORT_EDGE_ENDPOINT_MAX_TURN_DEG = 150.0
SHORT_EDGE_PRESERVATION_BONUS_CM = 0.04
# Thresholding can insert a vertex along an otherwise straight cut edge.
# The tolerated deviation scales with the full candidate side: perspective and
# blur can move a point by a few millimetres on a 10 cm edge, but must not
# erase a visibly bent short edge.
COLLINEAR_VERTEX_MIN_TURN_DEG = 165.0
COLLINEAR_VERTEX_MAX_DEVIATION_CM = 0.20
COLLINEAR_VERTEX_SPAN_RATIO = 0.05
COLLINEAR_VERTEX_DEVIATION_CAP_CM = 0.45
# Printed card artwork can meet a cut edge and make the segmented component
# look like it has a narrow physical notch.  Only repair a component when the
# normal polygon fit has already failed and its area is still close to its
# convex hull.  This avoids unconditional hole filling and preserves genuine
# large concavities.
MIN_PRINT_NOTCH_SOLIDITY = 0.88

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


def remove_nearly_collinear_vertices(
    polygon: np.ndarray,
) -> np.ndarray:
    """Collapse threshold-induced vertices lying on a straight physical edge."""
    vertices = polygon.reshape(-1, 2).astype(np.float32)
    while len(vertices) > 3:
        removable = []
        for index in range(len(vertices)):
            previous = vertices[index - 1]
            current = vertices[index]
            following = vertices[(index + 1) % len(vertices)]
            incoming = current - previous
            outgoing = following - current
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            if incoming_length < 1e-6 or outgoing_length < 1e-6:
                removable.append((0.0, index))
                continue
            cosine = float(
                np.clip(
                    np.dot(incoming, outgoing)
                    / (incoming_length * outgoing_length),
                    -1.0,
                    1.0,
                )
            )
            turn_degrees = 180.0 - math.degrees(math.acos(cosine))
            deviation = float(
                distance_point_to_segment(
                    current.reshape(1, 2),
                    previous,
                    following,
                )[0]
            )
            endpoint_span_cm = float(
                np.linalg.norm(following - previous)
            ) / PX_PER_CM
            maximum_deviation_cm = max(
                COLLINEAR_VERTEX_MAX_DEVIATION_CM,
                min(
                    COLLINEAR_VERTEX_DEVIATION_CAP_CM,
                    COLLINEAR_VERTEX_SPAN_RATIO * endpoint_span_cm,
                ),
            )
            if (
                turn_degrees >= COLLINEAR_VERTEX_MIN_TURN_DEG
                and deviation
                <= maximum_deviation_cm * PX_PER_CM
            ):
                removable.append((deviation, index))
        if not removable:
            break
        _, remove_index = min(removable)
        vertices = np.delete(vertices, remove_index, axis=0)
    return vertices


def has_supported_near_limit_edge(polygon: np.ndarray) -> bool:
    """Return whether a near-1 cm edge has two non-collinear endpoints."""
    vertices = polygon.reshape(-1, 2).astype(np.float32)
    minimum_length_px = (
        MIN_LEGAL_EDGE_CM - EDGE_WARNING_MARGIN_CM
    ) * PX_PER_CM
    maximum_length_px = SHORT_EDGE_EVIDENCE_MAX_CM * PX_PER_CM
    for index in range(len(vertices)):
        current = vertices[index]
        following = vertices[(index + 1) % len(vertices)]
        edge_length = float(np.linalg.norm(following - current))
        if not minimum_length_px <= edge_length <= maximum_length_px:
            continue
        endpoint_turns = []
        for vertex_id in (index, (index + 1) % len(vertices)):
            previous = vertices[vertex_id - 1]
            vertex = vertices[vertex_id]
            next_vertex = vertices[(vertex_id + 1) % len(vertices)]
            incoming = vertex - previous
            outgoing = next_vertex - vertex
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            if incoming_length < 1e-6 or outgoing_length < 1e-6:
                endpoint_turns.append(180.0)
                continue
            cosine = float(
                np.clip(
                    np.dot(incoming, outgoing)
                    / (incoming_length * outgoing_length),
                    -1.0,
                    1.0,
                )
            )
            endpoint_turns.append(
                180.0 - math.degrees(math.acos(cosine))
            )
        if max(endpoint_turns) <= SHORT_EDGE_ENDPOINT_MAX_TURN_DEG:
            return True
    return False


def fit_polygon(contour: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    # Compare all legal approximations. Pure contour error always prefers more
    # vertices, so charge a small physical-scale cost for each extra corner.
    # This makes the result stable when a nearly straight edge alternates
    # between four and five vertices across adjacent camera frames.
    ratios = [EPSILON_RATIO]
    ratios.extend(
        EPSILON_RATIO * factor
        for factor in (0.85, 1.15, 0.70, 1.30, 0.55, 1.50, 1.80)
    )
    candidates: dict[tuple, tuple[float, np.ndarray]] = {}
    for ratio in ratios:
        approximation = cv2.approxPolyDP(contour, ratio * perimeter, True)
        edge_count = len(approximation)
        if not 3 <= edge_count <= 5:
            continue
        cleaned = remove_nearly_collinear_vertices(approximation)
        cleaned = remove_impossible_short_edges(contour, cleaned)
        if not 3 <= len(cleaned) <= 5:
            continue
        signature = tuple(
            tuple(int(round(value)) for value in point)
            for point in cleaned.reshape(-1, 2)
        )
        contour_error_cm = (
            polygon_contour_error(contour, cleaned) / PX_PER_CM
        )
        score = (
            contour_error_cm
            + VERTEX_COMPLEXITY_PENALTY_CM * (len(cleaned) - 3)
        )
        if has_supported_near_limit_edge(cleaned):
            score -= SHORT_EDGE_PRESERVATION_BONUS_CM
        old = candidates.get(signature)
        if old is None or score < old[0]:
            candidates[signature] = (score, cleaned)

    if not candidates:
        return None
    _, cleaned = min(
        candidates.values(),
        key=lambda item: (item[0], len(item[1])),
    )
    refined = refine_polygon_with_lines(contour, cleaned)

    # Preserve contour order and reject a degenerate refinement.
    if abs(cv2.contourArea(refined.astype(np.float32))) < 1.0:
        return cleaned.reshape(-1, 2).astype(np.float32)
    return refined


def fill_external_components(binary: np.ndarray) -> np.ndarray:
    """Fill card markings/printing while discarding tiny colour noise."""
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    filled = np.zeros_like(binary)
    minimum_noise_area = (
        0.35 * PX_PER_CM * PX_PER_CM
    )
    for contour in contours:
        if cv2.contourArea(contour) < minimum_noise_area:
            continue
        cv2.drawContours(
            filled,
            [contour],
            -1,
            255,
            cv2.FILLED,
        )
    return filled


def repair_print_notches(binary: np.ndarray) -> np.ndarray:
    """Repair shallow edge notches caused by printed playing-card artwork."""
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    repaired = binary.copy()
    minimum_area = (
        MIN_PIECE_AREA_CM2 * PX_PER_CM * PX_PER_CM
    )
    maximum_area = (
        MAX_PIECE_AREA_CM2 * PX_PER_CM * PX_PER_CM
    )
    height, width = binary.shape
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not minimum_area <= area <= maximum_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if (
            x <= 1
            or y <= 1
            or x + w >= width - 1
            or y + h >= height - 1
        ):
            continue
        # A successful direct fit already represents the physical boundary;
        # do not alter it merely because it is mildly concave.
        if fit_polygon(contour) is not None:
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0.0 or hull_area > maximum_area:
            continue
        solidity = area / hull_area
        if solidity < MIN_PRINT_NOTCH_SOLIDITY:
            continue
        # The hull must itself remain a legal 3--5 edge task polygon.
        if fit_polygon(hull) is None:
            continue
        cv2.drawContours(
            repaired,
            [hull],
            -1,
            255,
            cv2.FILLED,
        )
    return repaired


def evaluate_mask_piece(
    binary: np.ndarray,
) -> tuple[
    tuple[int, int, int, float, float, float],
    np.ndarray,
]:
    """Rank masks by fitted pieces, components, clipping and total area.

    Counting connected components alone is insufficient for printed cards:
    artwork that reaches a cut edge can leave a deep open notch. Such a
    component passes the area test but later fails polygon fitting. Evaluate
    the same repair-and-fit path used by detection so threshold selection
    does not stop on an unusable four-component mask.
    """
    scored_binary = repair_print_notches(binary)
    contours, _ = cv2.findContours(
        scored_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    minimum_area = (
        MIN_PIECE_AREA_CM2 * PX_PER_CM * PX_PER_CM
    )
    maximum_area = (
        MAX_PIECE_AREA_CM2 * PX_PER_CM * PX_PER_CM
    )
    height, width = binary.shape
    plausible = []
    fragment_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not minimum_area <= area <= maximum_area:
            if 0.35 * PX_PER_CM * PX_PER_CM <= area < minimum_area:
                fragment_area += area
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if (
            x <= 1
            or y <= 1
            or x + w >= width - 1
            or y + h >= height - 1
        ):
            continue
        plausible.append((float(area), contour))

    plausible.sort(key=lambda item: item[0], reverse=True)
    selected = plausible[:MAX_PIECES]
    fitted_count = 0
    shape_error = 0.0
    for contour_area, contour in selected:
        polygon = fit_polygon(contour)
        if polygon is None:
            shape_error += 10.0
            continue
        fitted_count += 1
        contour_error_cm = (
            polygon_contour_error(contour, polygon) / PX_PER_CM
        )
        polygon_area = abs(float(cv2.contourArea(polygon)))
        relative_area_error = abs(
            polygon_area - contour_area
        ) / max(contour_area, 1.0)
        shape_error += contour_error_cm + relative_area_error

    extra_components = max(0, len(plausible) - MAX_PIECES)
    return (
        (
            fitted_count,
            len(selected),
            -extra_components,
            -shape_error,
            -fragment_area / (PX_PER_CM * PX_PER_CM),
            float(sum(area for area, _ in selected)),
        ),
        scored_binary,
    )


def mask_piece_score(
    binary: np.ndarray,
) -> tuple[int, int, int, float, float, float]:
    score, _ = evaluate_mask_piece(binary)
    return score


def fast_mask_piece_score(
    binary: np.ndarray,
) -> tuple[int, int, float, float]:
    """Cheap threshold pre-rank without polygon fitting or notch repair."""
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    minimum_area = (
        MIN_PIECE_AREA_CM2 * PX_PER_CM * PX_PER_CM
    )
    maximum_area = (
        MAX_PIECE_AREA_CM2 * PX_PER_CM * PX_PER_CM
    )
    fragment_minimum = 0.35 * PX_PER_CM * PX_PER_CM
    height, width = binary.shape
    plausible_areas = []
    fragment_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not minimum_area <= area <= maximum_area:
            if fragment_minimum <= area < minimum_area:
                fragment_area += area
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if (
            x <= 1
            or y <= 1
            or x + w >= width - 1
            or y + h >= height - 1
        ):
            continue
        plausible_areas.append(area)
    plausible_areas.sort(reverse=True)
    selected = plausible_areas[:MAX_PIECES]
    return (
        len(selected),
        -max(0, len(plausible_areas) - MAX_PIECES),
        -fragment_area / (PX_PER_CM * PX_PER_CM),
        float(sum(selected)),
    )


def clean_segmentation_mask(
    binary: np.ndarray,
    invert_bright_majority: bool = False,
) -> np.ndarray:
    """Apply the common low-cost cleanup used by all threshold candidates."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )
    if (
        invert_bright_majority
        and cv2.countNonZero(binary) > binary.size * 0.55
    ):
        binary = cv2.bitwise_not(binary)
    return fill_external_components(binary)


def select_threshold_candidate(
    measurement: np.ndarray,
    thresholds: Iterable[int],
    preferred_threshold: int,
    invert_bright_majority: bool = False,
) -> tuple[
    np.ndarray,
    int,
    tuple[int, int, int, float, float, float],
    np.ndarray,
]:
    """Choose the threshold that produces the most plausible piece mask.

    Otsu is retained as one candidate, but it can no longer unilaterally
    discard a low-contrast white card on a pale coloured sheet.  Candidate
    count and clipping are primary; closeness to the robust background-noise
    estimate breaks ties before contour area.
    """
    candidates = []
    ordered_thresholds = [int(preferred_threshold)]
    ordered_thresholds.extend(int(value) for value in thresholds)
    seen_thresholds = set()
    for threshold in ordered_thresholds:
        if threshold in seen_thresholds:
            continue
        seen_thresholds.add(threshold)
        _, binary = cv2.threshold(
            measurement,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )
        binary = clean_segmentation_mask(
            binary,
            invert_bright_majority=invert_bright_majority,
        )
        fast_score = fast_mask_piece_score(binary)
        fast_key = (
            fast_score[0],
            fast_score[1],
            fast_score[2],
            -abs(threshold - preferred_threshold),
            fast_score[3],
        )
        candidates.append((fast_key, threshold, binary))

    # Polygon fitting is the dominant Maix cost. Four finalists retain
    # nearby exposure alternatives without repeating that work for every
    # threshold in the colour-distance sweep.
    candidates.sort(key=lambda item: item[0], reverse=True)
    finalists = candidates[:4]
    best_binary = np.zeros_like(measurement)
    best_threshold = int(preferred_threshold)
    best_key = None
    best_score = (0, 0, 0, float("-inf"), float("-inf"), 0.0)
    best_repaired = np.zeros_like(measurement)
    for _, threshold, binary in finalists:
        score, repaired = evaluate_mask_piece(binary)
        key = (
            score[0],
            score[1],
            score[2],
            score[3],
            score[4],
            -abs(threshold - preferred_threshold),
            score[5],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_binary = binary
            best_threshold = threshold
            best_score = score
            best_repaired = repaired
    return best_binary, best_threshold, best_score, best_repaired


def segment_intensity_pieces(
    work_image: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    tuple[int, int, int, float, float, float],
    np.ndarray,
]:
    gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold, _ = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    otsu_threshold = int(
        np.clip(otsu_threshold, MIN_BRIGHT_THRESHOLD, MAX_BRIGHT_THRESHOLD)
    )
    thresholds = [
        int(
            np.clip(
                otsu_threshold + offset,
                MIN_BRIGHT_THRESHOLD,
                MAX_BRIGHT_THRESHOLD,
            )
        )
        for offset in (-24, -12, 0, 12, 24)
    ]
    # Supports bright cards on black/dark boards and darker pieces on white
    # sheets.  A majority-bright result is inverted for the latter case.
    return select_threshold_candidate(
        gray,
        thresholds,
        otsu_threshold,
        invert_bright_majority=True,
    )


def segment_colour_difference(
    work_image: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    tuple[int, int, int, float, float, float],
    np.ndarray,
]:
    """Separate pieces from an arbitrary, approximately uniform sheet."""
    blurred = cv2.GaussianBlur(work_image, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    # The rectified edge can contain the white physical A4 border, so edge
    # samples are not a reliable sheet-colour estimate.  Puzzle pieces occupy
    # a minority of the work area; a sparse whole-area median therefore
    # estimates the actual blue/coloured sheet much more robustly.
    samples = lab[::6, ::6].reshape(-1, 3)
    background = np.median(samples, axis=0)
    difference = cv2.absdiff(
        lab,
        (
            float(background[0]),
            float(background[1]),
            float(background[2]),
            0.0,
        ),
    )
    first, second, third = cv2.split(difference)
    distance = cv2.max(first, cv2.max(second, third))
    otsu_threshold, _ = cv2.threshold(
        distance,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    otsu_threshold = int(
        np.clip(
            otsu_threshold,
            MIN_COLOR_DISTANCE_THRESHOLD,
            MAX_COLOR_DISTANCE_THRESHOLD,
        )
    )
    # The 70th percentile still belongs to the sheet when four separated
    # pieces occupy a minority of the work area.  Moving just above it gives
    # a robust estimate of sheet illumination/print variation.  In capture
    # 131923 this is 22 instead of Otsu's incorrect 49.
    background_threshold = int(
        np.clip(
            np.percentile(distance, 70.0) + 3.0,
            MIN_COLOR_DISTANCE_THRESHOLD,
            40,
        )
    )
    thresholds = [
        background_threshold + offset
        for offset in (-8, -4, 0, 4, 8, 12)
    ]
    thresholds.extend(
        (
            MIN_COLOR_DISTANCE_THRESHOLD,
            18,
            22,
            26,
            30,
            34,
            38,
            otsu_threshold,
        )
    )
    thresholds = [
        int(
            np.clip(
                value,
                MIN_COLOR_DISTANCE_THRESHOLD,
                MAX_COLOR_DISTANCE_THRESHOLD,
            )
        )
        for value in thresholds
    ]
    return select_threshold_candidate(
        distance,
        thresholds,
        background_threshold,
    )


def segment_bright_pieces(
    work_image: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Choose a fast primary mode and run the other only as fallback."""
    samples = work_image[::6, ::6].reshape(-1, 3)
    background_bgr = np.median(samples, axis=0)
    background_chroma = float(
        np.max(background_bgr) - np.min(background_bgr)
    )
    if background_chroma >= 25.0:
        primary = segment_colour_difference
        secondary = segment_intensity_pieces
    else:
        primary = segment_intensity_pieces
        secondary = segment_colour_difference

    (
        primary_binary,
        primary_threshold,
        primary_score,
        primary_repaired,
    ) = primary(work_image)
    if (
        primary_score[0] == MAX_PIECES
        and primary_score[1] == MAX_PIECES
        and primary_score[2] == 0
    ):
        return primary_binary, primary_threshold, primary_repaired

    (
        secondary_binary,
        secondary_threshold,
        secondary_score,
        secondary_repaired,
    ) = secondary(work_image)
    if secondary_score > primary_score:
        return (
            secondary_binary,
            secondary_threshold,
            secondary_repaired,
        )
    return primary_binary, primary_threshold, primary_repaired


def normalize_piece_components(binary: np.ndarray) -> np.ndarray:
    """Keep real pieces and render their fitted straight-edge silhouettes.

    Printed artwork can create detached islands or open notches in the raw
    colour mask. Once a component has passed the task's area, clipping and
    polygon checks, the fitted 3--5-edge polygon is the more faithful physical
    boundary for geometry and debug output.
    """
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    minimum_area = MIN_PIECE_AREA_CM2 * PX_PER_CM * PX_PER_CM
    maximum_area = MAX_PIECE_AREA_CM2 * PX_PER_CM * PX_PER_CM
    height, width = binary.shape
    fitted_components = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not minimum_area <= area <= maximum_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if (
            x <= 1
            or y <= 1
            or x + w >= width - 1
            or y + h >= height - 1
        ):
            continue
        polygon = fit_polygon(contour)
        if polygon is None:
            continue
        fitted_components.append((area, polygon))

    fitted_components.sort(key=lambda item: item[0], reverse=True)
    normalized = np.zeros_like(binary)
    for _, polygon in fitted_components[:MAX_PIECES]:
        polygon_i = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(
            normalized,
            [polygon_i],
            -1,
            255,
            cv2.FILLED,
        )
    return normalized


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
    _, threshold, repaired_binary = segment_bright_pieces(work_image)
    binary = normalize_piece_components(repaired_binary)
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

    if mode in ("ready", "frozen"):
        button_color = (0, 180, 0) if mode == "ready" else (0, 150, 255)
        button_text = "START" if mode == "ready" else "START NEXT"
        cv2.rectangle(screen, (10, 375), (310, 465), button_color, -1)
        cv2.rectangle(screen, (330, 375), (630, 465), (180, 90, 0), -1)
        cv2.putText(
            screen,
            button_text,
            (30 if mode == "frozen" else 70, 435),
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


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


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
            last_raw = frame
            if mode == "align":
                preview_source = draw_camera_guide(frame)
            else:
                # Preview only. The expensive detector runs once on FREEZE.
                preview_source = draw_camera_guide(frame)
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
                if point_in_rect(x, y, BUTTON_DETECT_RECT):
                    if mode == "align":
                        mode = "live"
                        message = "Live preview - tap FREEZE"
                    elif mode == "live":
                        mode = "frozen"
                        last_rectified, _ = rectify_a4(last_raw)
                        (
                            last_pieces,
                            last_binary,
                            threshold,
                        ) = detect_pieces(last_rectified)
                        last_annotated = draw_detection(
                            last_rectified,
                            last_pieces,
                            threshold,
                        )
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
                        message = "Live preview - tap FREEZE"
                elif point_in_rect(x, y, BUTTON_SAVE_RECT):
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
