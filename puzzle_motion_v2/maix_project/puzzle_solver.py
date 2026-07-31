"""OpenCV/Numpy-only geometric puzzle solver for MaixCAM 2.

Input and output coordinates use centimetres in the rectified A4 frame:
origin at A4 top-left, +x right, +y down. Positive angles are clockwise.
"""

from __future__ import annotations

import math
import time as pytime
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


EDGE_ABS_TOLERANCE_CM = 0.20
EDGE_REL_TOLERANCE = 0.05
# Questions 1 and 2 permit physical edges down to 1 cm.  Split seams must
# retain those short legal segments instead of pruning them as noise.
MIN_SPLIT_SEGMENT_CM = 1.00
SPLIT_CLOSURE_ABS_TOLERANCE_CM = 0.35
SPLIT_CLOSURE_REL_TOLERANCE = 0.10
INTERVAL_EPSILON = 1e-4
# Camera perspective and polygon fitting can make two physically mating
# cardboard contours overlap slightly. These relaxed competition settings
# still reject the earlier 87--89% false layouts.
MAX_OVERLAP_AREA_CM2 = 0.50
MIN_RECTANGULARITY = 0.92
EARLY_ACCEPT_RECTANGULARITY = 0.925
MAX_SEARCH_NODES = 180_000
MAX_SEARCH_SECONDS = 3.00
OVERLAP_RASTER_PX_PER_CM = 20.0

# Questions 1 and 2 require a 5--9 cm short side and a 9--12 cm long side.
TARGET_MIN_SHORT_CM = 5.00
TARGET_MAX_SHORT_CM = 9.00
TARGET_MIN_LONG_CM = 9.00
TARGET_MAX_LONG_CM = 12.00

A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
A4_HALF_HEIGHT_CM = A4_HEIGHT_CM * 0.5

_LAST_SEARCH_TIMED_OUT = False


@dataclass
class Placement:
    piece_id: int
    vertices: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    used_edges: frozenset[int]
    # For every polygon edge, store the occupied native parameter ranges.
    # An empty edge is (), a fully occupied edge is ((0.0, 1.0),).
    # Keeping ranges instead of one boolean is what permits T-junctions:
    # two short edges can consume the two halves of one long edge.
    edge_coverage: tuple[tuple[tuple[float, float], ...], ...] = ()


@dataclass
class Solution:
    placements: list[Placement]
    rectangularity: float
    width_cm: float
    height_cm: float
    search_nodes: int = 0


def last_search_timed_out() -> bool:
    return _LAST_SEARCH_TIMED_OUT


def polygon_area_signed(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_area(vertices: np.ndarray) -> float:
    return abs(polygon_area_signed(vertices))


def counter_clockwise(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    # In an image coordinate system (+y down), visually counter-clockwise
    # polygons have a negative shoelace area.
    if polygon_area_signed(vertices) > 0:
        return vertices[::-1].copy()
    return vertices.copy()


def polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    signed_area = polygon_area_signed(vertices)
    if abs(signed_area) < 1e-9:
        return np.mean(vertices, axis=0)
    cross = (
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - np.roll(vertices[:, 0], -1) * vertices[:, 1]
    )
    cx = np.sum(
        (vertices[:, 0] + np.roll(vertices[:, 0], -1)) * cross
    ) / (6.0 * signed_area)
    cy = np.sum(
        (vertices[:, 1] + np.roll(vertices[:, 1], -1)) * cross
    ) / (6.0 * signed_area)
    return np.asarray([cx, cy], dtype=np.float64)


def polygon_edges(
    vertices: np.ndarray,
) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    for index in range(len(vertices)):
        yield index, vertices[index], vertices[(index + 1) % len(vertices)]


def edge_length(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(second - first))


def lengths_match(first: float, second: float) -> bool:
    tolerance = max(
        EDGE_ABS_TOLERANCE_CM,
        EDGE_REL_TOLERANCE * max(first, second),
    )
    return abs(first - second) <= tolerance


def split_closure_matches(first: float, second: float) -> bool:
    """Use a wider tolerance only when closing an already split edge."""
    tolerance = max(
        SPLIT_CLOSURE_ABS_TOLERANCE_CM,
        SPLIT_CLOSURE_REL_TOLERANCE * max(first, second),
    )
    return abs(first - second) <= tolerance


def empty_edge_coverage(
    vertices: np.ndarray,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(() for _ in range(len(vertices)))


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    cleaned = sorted(
        (
            max(0.0, min(1.0, float(start))),
            max(0.0, min(1.0, float(end))),
        )
        for start, end in intervals
        if end - start > INTERVAL_EPSILON
    )
    merged: list[list[float]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1] + INTERVAL_EPSILON:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def add_edge_coverage(
    coverage: tuple[tuple[tuple[float, float], ...], ...],
    edge_id: int,
    start: float,
    end: float,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    updated = list(coverage)
    updated[edge_id] = merge_intervals(
        tuple(updated[edge_id]) + ((min(start, end), max(start, end)),)
    )
    return tuple(updated)


def uncovered_intervals(
    coverage: tuple[tuple[tuple[float, float], ...], ...],
    edge_id: int,
) -> tuple[tuple[float, float], ...]:
    available = []
    cursor = 0.0
    for start, end in coverage[edge_id]:
        if start > cursor + INTERVAL_EPSILON:
            available.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < 1.0 - INTERVAL_EPSILON:
        available.append((cursor, 1.0))
    return tuple(available)


def coverage_used_edges(
    coverage: tuple[tuple[tuple[float, float], ...], ...],
) -> frozenset[int]:
    return frozenset(
        edge_id
        for edge_id, intervals in enumerate(coverage)
        if (
            intervals
            and intervals[0][0] <= INTERVAL_EPSILON
            and intervals[-1][1] >= 1.0 - INTERVAL_EPSILON
            and sum(end - start for start, end in intervals)
            >= 1.0 - INTERVAL_EPSILON
        )
    )


def edge_point(
    first: np.ndarray,
    second: np.ndarray,
    parameter: float,
) -> np.ndarray:
    return first + (second - first) * parameter


def segment_contact_candidates(
    source_length: float,
    target_length: float,
    target_start: float,
    target_end: float,
    target_is_remainder: bool,
) -> list[tuple[float, float, float, float]]:
    """Return source/target parameter spans that can form one contact.

    The source belongs to an unplaced piece and is initially the complete
    [0, 1] edge. The target span is one currently uncovered interval.
    A shorter edge may occupy either end of a longer interval. If the target
    is the remainder of a previous split, the final closure receives a
    slightly wider tolerance to absorb camera/polygon-fitting error.
    """
    target_fraction = target_end - target_start
    available_target_length = target_length * target_fraction
    close_match = lengths_match(source_length, available_target_length)
    if target_is_remainder:
        close_match = close_match or split_closure_matches(
            source_length,
            available_target_length,
        )
    if close_match:
        return [(0.0, 1.0, target_start, target_end)]

    candidates: list[tuple[float, float, float, float]] = []
    if source_length < available_target_length:
        remainder = available_target_length - source_length
        if (
            source_length >= MIN_SPLIT_SEGMENT_CM
            and remainder >= MIN_SPLIT_SEGMENT_CM
        ):
            target_piece_fraction = source_length / target_length
            candidates.extend(
                [
                    (
                        0.0,
                        1.0,
                        target_start,
                        target_start + target_piece_fraction,
                    ),
                    (
                        0.0,
                        1.0,
                        target_end - target_piece_fraction,
                        target_end,
                    ),
                ]
            )
    else:
        remainder = source_length - available_target_length
        if (
            available_target_length >= MIN_SPLIT_SEGMENT_CM
            and remainder >= MIN_SPLIT_SEGMENT_CM
        ):
            source_piece_fraction = available_target_length / source_length
            candidates.extend(
                [
                    (
                        0.0,
                        source_piece_fraction,
                        target_start,
                        target_end,
                    ),
                    (
                        1.0 - source_piece_fraction,
                        1.0,
                        target_start,
                        target_end,
                    ),
                ]
            )
    return candidates


def remainder_can_be_filled(
    remainder_cm: float,
    future_piece_ids: set[int],
    piece_edge_lengths: list[tuple[float, ...]],
) -> bool:
    """Conservative pruning for an uncovered remainder of a split target."""
    if remainder_cm < MIN_SPLIT_SEGMENT_CM:
        return False
    future_edges = [
        (piece_id, length)
        for piece_id in future_piece_ids
        for length in piece_edge_lengths[piece_id]
    ]
    if any(
        split_closure_matches(remainder_cm, length)
        for _, length in future_edges
    ):
        return True
    for first_index, (first_id, first_length) in enumerate(future_edges):
        for second_id, second_length in future_edges[first_index + 1:]:
            if first_id == second_id:
                continue
            if split_closure_matches(
                remainder_cm,
                first_length + second_length,
            ):
                return True
    return False


def edge_transform(
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map source A->B onto the reversed target B->A."""
    source_vector = source_b - source_a
    target_vector = target_a - target_b
    angle = math.atan2(target_vector[1], target_vector[0]) - math.atan2(
        source_vector[1], source_vector[0]
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    source_midpoint = (source_a + source_b) * 0.5
    target_midpoint = (target_a + target_b) * 0.5
    translation = target_midpoint - rotation @ source_midpoint
    return rotation, translation


def apply_transform(
    vertices: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return vertices @ rotation.T + translation


def legal_target_dimensions(width: float, height: float) -> bool:
    short_side, long_side = sorted((width, height))
    return (
        TARGET_MIN_SHORT_CM <= short_side <= TARGET_MAX_SHORT_CM
        and TARGET_MIN_LONG_CM <= long_side <= TARGET_MAX_LONG_CM
    )


def overlap_area_raster(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_min = np.min(first, axis=0)
    first_max = np.max(first, axis=0)
    second_min = np.min(second, axis=0)
    second_max = np.max(second, axis=0)
    common_min = np.maximum(first_min, second_min)
    common_max = np.minimum(first_max, second_max)
    if np.any(common_max <= common_min):
        return 0.0

    # Camera-fitted contest pieces are normally convex. Native OpenCV
    # intersection avoids allocating two raster masks for every branch.
    first_float = np.asarray(first, dtype=np.float32)
    second_float = np.asarray(second, dtype=np.float32)
    if (
        hasattr(cv2, "intersectConvexConvex")
        and cv2.isContourConvex(first_float)
        and cv2.isContourConvex(second_float)
    ):
        intersection_area, _ = cv2.intersectConvexConvex(
            first_float,
            second_float,
        )
        return float(intersection_area)

    all_min = np.minimum(first_min, second_min) - 0.2
    all_max = np.maximum(first_max, second_max) + 0.2
    size = np.ceil(
        (all_max - all_min) * OVERLAP_RASTER_PX_PER_CM
    ).astype(int) + 3
    if np.any(size <= 0) or np.any(size > 1200):
        return float("inf")

    first_mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    second_mask = np.zeros_like(first_mask)

    def raster(vertices: np.ndarray) -> np.ndarray:
        return np.round(
            (vertices - all_min) * OVERLAP_RASTER_PX_PER_CM
        ).astype(np.int32)

    cv2.fillPoly(first_mask, [raster(first)], 255)
    cv2.fillPoly(second_mask, [raster(second)], 255)
    # Removing one boundary pixel prevents a shared legal edge from being
    # counted as an overlap merely because raster polygons include borders.
    kernel = np.ones((3, 3), dtype=np.uint8)
    first_mask = cv2.erode(first_mask, kernel)
    second_mask = cv2.erode(second_mask, kernel)
    intersection_pixels = cv2.countNonZero(
        cv2.bitwise_and(first_mask, second_mask)
    )
    return intersection_pixels / (
        OVERLAP_RASTER_PX_PER_CM * OVERLAP_RASTER_PX_PER_CM
    )


def minimum_rectangle(
    placements: list[Placement],
) -> tuple[np.ndarray, float, float, float]:
    points = np.concatenate(
        [placement.vertices for placement in placements],
        axis=0,
    ).astype(np.float32)
    rectangle = cv2.minAreaRect(points)
    box = cv2.boxPoints(rectangle).astype(np.float64)
    side_lengths = [
        edge_length(box[index], box[(index + 1) % 4])
        for index in range(4)
    ]
    width = min(side_lengths)
    height = max(side_lengths)
    return box, width, height, width * height


def evaluate_complete(placements: list[Placement]) -> Solution | None:
    _, width, height, rectangle_area = minimum_rectangle(placements)
    if rectangle_area <= 1e-6:
        return None
    total_area = sum(
        polygon_area(placement.vertices) for placement in placements
    )
    rectangularity = total_area / rectangle_area
    if (
        rectangularity >= MIN_RECTANGULARITY
        and legal_target_dimensions(width, height)
    ):
        return Solution(
            placements=list(placements),
            rectangularity=rectangularity,
            width_cm=width,
            height_cm=height,
        )
    return None


def solve_geometry(
    pieces: list[np.ndarray],
    max_nodes: int = MAX_SEARCH_NODES,
    max_seconds: float | None = MAX_SEARCH_SECONDS,
) -> tuple[Solution | None, int]:
    """Search edge-compatible, non-overlapping placements for 2--4 pieces."""
    global _LAST_SEARCH_TIMED_OUT
    _LAST_SEARCH_TIMED_OUT = False
    if not 1 <= len(pieces) <= 4:
        return None, 0
    deadline = (
        pytime.monotonic() + max_seconds
        if max_seconds is not None and max_seconds > 0
        else None
    )
    normalized = [
        counter_clockwise(np.asarray(piece, dtype=np.float64))
        for piece in pieces
    ]
    piece_edge_lengths = [
        tuple(
            edge_length(first, second)
            for _, first, second in polygon_edges(piece)
        )
        for piece in normalized
    ]
    order = sorted(
        range(len(normalized)),
        key=lambda piece_id: polygon_area(normalized[piece_id]),
        reverse=True,
    )
    anchor_id = order[0]
    identity = np.eye(2, dtype=np.float64)
    zero = np.zeros(2, dtype=np.float64)
    anchor = Placement(
        piece_id=anchor_id,
        vertices=normalized[anchor_id],
        rotation=identity,
        translation=zero,
        used_edges=frozenset(),
        edge_coverage=empty_edge_coverage(normalized[anchor_id]),
    )
    if len(normalized) == 1:
        solution = evaluate_complete([anchor])
        if solution is not None:
            solution.search_nodes = 1
        return solution, 1
    remaining = set(range(len(normalized))) - {anchor_id}
    visited: set[tuple] = set()
    best_solution: Solution | None = None
    node_count = 0
    timed_out = False

    def time_expired() -> bool:
        nonlocal timed_out
        if deadline is not None and pytime.monotonic() >= deadline:
            timed_out = True
        return timed_out

    def state_key(placements: list[Placement]) -> tuple:
        items = []
        for placement in sorted(
            placements,
            key=lambda item: item.piece_id,
        ):
            centroid = polygon_centroid(placement.vertices)
            first_edge = placement.vertices[1] - placement.vertices[0]
            angle = math.atan2(first_edge[1], first_edge[0])
            items.append(
                (
                    placement.piece_id,
                    round(float(centroid[0]), 2),
                    round(float(centroid[1]), 2),
                    round(angle, 2),
                    tuple(
                        tuple(
                            (round(start, 2), round(end, 2))
                            for start, end in intervals
                        )
                        for intervals in placement.edge_coverage
                    ),
                )
            )
        return tuple(items)

    def recurse(
        placements: list[Placement],
        unplaced: set[int],
    ) -> bool:
        nonlocal best_solution, node_count
        if time_expired():
            return False
        node_count += 1
        if node_count > max_nodes:
            return False
        key = state_key(placements)
        if key in visited:
            return False
        visited.add(key)

        if not unplaced:
            solution = evaluate_complete(placements)
            if solution is not None and (
                best_solution is None
                or solution.rectangularity > best_solution.rectangularity
            ):
                solution.search_nodes = node_count
                best_solution = solution
                if (
                    solution.rectangularity
                    >= EARLY_ACCEPT_RECTANGULARITY
                ):
                    return True
            return False

        candidate_ids = sorted(
            unplaced,
            key=lambda piece_id: max(
                edge_length(a, b)
                for _, a, b in polygon_edges(normalized[piece_id])
            ),
            reverse=True,
        )
        for candidate_id in candidate_ids:
            if time_expired():
                return False
            source_vertices = normalized[candidate_id]
            for source_edge_id, source_a, source_b in polygon_edges(
                source_vertices
            ):
                source_length = edge_length(source_a, source_b)
                for target_index, target in enumerate(placements):
                    for target_edge_id, target_a, target_b in polygon_edges(
                        target.vertices
                    ):
                        target_length = edge_length(target_a, target_b)
                        target_has_coverage = bool(
                            target.edge_coverage[target_edge_id]
                        )
                        for target_start, target_end in uncovered_intervals(
                            target.edge_coverage,
                            target_edge_id,
                        ):
                            contacts = segment_contact_candidates(
                                source_length,
                                target_length,
                                target_start,
                                target_end,
                                target_has_coverage,
                            )
                            available_target_length = (
                                target_length
                                * (target_end - target_start)
                            )
                            if (
                                source_length
                                < available_target_length
                                and not lengths_match(
                                    source_length,
                                    available_target_length,
                                )
                                and not remainder_can_be_filled(
                                    available_target_length - source_length,
                                    unplaced - {candidate_id},
                                    piece_edge_lengths,
                                )
                            ):
                                continue
                            for (
                                source_start,
                                source_end,
                                contact_target_start,
                                contact_target_end,
                            ) in contacts:
                                if time_expired():
                                    return False
                                contact_source_a = edge_point(
                                    source_a,
                                    source_b,
                                    source_start,
                                )
                                contact_source_b = edge_point(
                                    source_a,
                                    source_b,
                                    source_end,
                                )
                                contact_target_a = edge_point(
                                    target_a,
                                    target_b,
                                    contact_target_start,
                                )
                                contact_target_b = edge_point(
                                    target_a,
                                    target_b,
                                    contact_target_end,
                                )
                                rotation, translation = edge_transform(
                                    contact_source_a,
                                    contact_source_b,
                                    contact_target_a,
                                    contact_target_b,
                                )
                                transformed = apply_transform(
                                    source_vertices,
                                    rotation,
                                    translation,
                                )
                                if polygon_area(transformed) <= 0.1:
                                    continue
                                if any(
                                    overlap_area_raster(
                                        transformed,
                                        placed.vertices,
                                    ) > MAX_OVERLAP_AREA_CM2
                                    for placed in placements
                                ):
                                    continue

                                target_coverage = add_edge_coverage(
                                    target.edge_coverage,
                                    target_edge_id,
                                    contact_target_start,
                                    contact_target_end,
                                )
                                source_coverage = add_edge_coverage(
                                    empty_edge_coverage(source_vertices),
                                    source_edge_id,
                                    source_start,
                                    source_end,
                                )
                                updated_target = Placement(
                                    piece_id=target.piece_id,
                                    vertices=target.vertices,
                                    rotation=target.rotation,
                                    translation=target.translation,
                                    used_edges=coverage_used_edges(
                                        target_coverage
                                    ),
                                    edge_coverage=target_coverage,
                                )
                                next_placements = list(placements)
                                next_placements[target_index] = updated_target
                                next_placements.append(
                                    Placement(
                                        piece_id=candidate_id,
                                        vertices=transformed,
                                        rotation=rotation,
                                        translation=translation,
                                        used_edges=coverage_used_edges(
                                            source_coverage
                                        ),
                                        edge_coverage=source_coverage,
                                    )
                                )
                                if recurse(
                                    next_placements,
                                    unplaced - {candidate_id},
                                ):
                                    return True
        return False

    recurse([anchor], remaining)
    _LAST_SEARCH_TIMED_OUT = timed_out
    if best_solution is not None:
        best_solution.search_nodes = node_count
    return best_solution, node_count


def canonical_target_solution(
    solution: Solution,
    top_margin_cm: float = 0.6,
) -> Solution:
    """Rotate and translate a solved rectangle into the lower A4 half."""
    box, _, _, _ = minimum_rectangle(solution.placements)
    vectors = [
        box[(index + 1) % 4] - box[index]
        for index in range(4)
    ]
    long_vector = max(vectors, key=lambda vector: np.linalg.norm(vector))
    long_angle = math.atan2(long_vector[1], long_vector[0])
    cosine = math.cos(-long_angle)
    sine = math.sin(-long_angle)
    global_rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )

    rotated_points = np.concatenate(
        [
            placement.vertices @ global_rotation.T
            for placement in solution.placements
        ],
        axis=0,
    )
    minimum = np.min(rotated_points, axis=0)
    maximum = np.max(rotated_points, axis=0)
    size = maximum - minimum
    if size[1] > size[0]:
        quarter_turn = np.asarray(
            [[0.0, -1.0], [1.0, 0.0]],
            dtype=np.float64,
        )
        global_rotation = quarter_turn @ global_rotation
        rotated_points = np.concatenate(
            [
                placement.vertices @ global_rotation.T
                for placement in solution.placements
            ],
            axis=0,
        )
        minimum = np.min(rotated_points, axis=0)
        maximum = np.max(rotated_points, axis=0)
        size = maximum - minimum

    target_center = np.asarray(
        [
            A4_WIDTH_CM * 0.5,
            A4_HALF_HEIGHT_CM + top_margin_cm + size[1] * 0.5,
        ],
        dtype=np.float64,
    )
    current_center = (minimum + maximum) * 0.5
    global_translation = target_center - current_center

    final_placements = []
    for placement in solution.placements:
        final_rotation = global_rotation @ placement.rotation
        final_translation = (
            global_rotation @ placement.translation + global_translation
        )
        final_placements.append(
            Placement(
                piece_id=placement.piece_id,
                vertices=apply_transform(
                    placement.vertices,
                    global_rotation,
                    global_translation,
                ),
                rotation=final_rotation,
                translation=final_translation,
                used_edges=placement.used_edges,
                edge_coverage=placement.edge_coverage,
            )
        )
    final_points = np.concatenate(
        [placement.vertices for placement in final_placements],
        axis=0,
    )
    final_size = np.max(final_points, axis=0) - np.min(
        final_points,
        axis=0,
    )
    width = float(final_size[0])
    height = float(final_size[1])
    rectangle_area = width * height
    total_area = sum(
        polygon_area(placement.vertices)
        for placement in final_placements
    )
    return Solution(
        placements=final_placements,
        rectangularity=total_area / rectangle_area,
        width_cm=width,
        height_cm=height,
        search_nodes=solution.search_nodes,
    )
