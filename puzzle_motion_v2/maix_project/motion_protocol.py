"""Pickup-point planning and shared motion-command JSON protocol."""

from __future__ import annotations

import json
import math
import os
import struct
import time
from pathlib import Path

import cv2
import numpy as np

from puzzle_solver import (
    A4_HEIGHT_CM,
    A4_HALF_HEIGHT_CM,
    A4_WIDTH_CM,
    TARGET_REGION_CENTERLINE_CLEARANCE_CM,
    Solution,
    apply_transform,
    overlap_area_raster,
    polygon_area,
    polygon_centroid,
)


PROTOCOL_NAME = "maixcam2-puzzle-motion-v1"
STM32_FRAME_PROTOCOL = "maixcam2-stm32-motion-v3-binary"
# Each fixed-size frame is: A5 5A, pick_x, pick_y, place_x, place_y,
# angle, CRC8. Positions are signed millimetres and angle is signed whole deg.
STM32_FRAME_HEADER = b"\xA5\x5A"
STM32_FRAME_FORMAT = "<2s5hB"
STM32_FRAME_SIZE = struct.calcsize(STM32_FRAME_FORMAT)
PICKUP_RASTER_PX_PER_CM = 40.0
MAGNET_RADIUS_CM = 0.50
PICKUP_MARGIN_CM = 0.20
# Mechanical compensation belongs to the actuator plan, not the geometric
# solver.  Keep every solved rotation and polygon unchanged, but move piece
# centres slightly away from the completed rectangle centre.  This creates a
# small assembly gap for rail positioning error without weakening topology,
# card-aspect, border or texture validation.
# Keep a 3 mm final assembly seam.  This leaves room for angular placement
# error without spreading the completed rectangle more than necessary.
ASSEMBLY_MIN_SEAM_GAP_CM = 0.30
ASSEMBLY_CLEARANCE_SCALE = 1.000
# The workspace boundary normally stops expansion first.  This only prevents
# a malformed layout with coincident centroids from looping forever.
ASSEMBLY_CLEARANCE_SEARCH_LIMIT = 3.000
ASSEMBLY_CLEARANCE_SCALE_STEP = 0.005
ASSEMBLY_MAX_OVERLAP_RATIO = 0.001
ASSEMBLY_WORKSPACE_EDGE_MARGIN_CM = 0.50


def safe_pickup_point(
    vertices_cm: np.ndarray,
    magnet_radius_cm: float = MAGNET_RADIUS_CM,
    margin_cm: float = PICKUP_MARGIN_CM,
) -> tuple[np.ndarray, float, bool]:
    vertices = np.asarray(vertices_cm, dtype=np.float64)
    minimum = np.min(vertices, axis=0) - 0.3
    maximum = np.max(vertices, axis=0) + 0.3
    size = np.ceil(
        (maximum - minimum) * PICKUP_RASTER_PX_PER_CM
    ).astype(int) + 3
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    raster = np.round(
        (vertices - minimum) * PICKUP_RASTER_PX_PER_CM
    ).astype(np.int32)
    cv2.fillPoly(mask, [raster], 255)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, maximum_distance_px, _, maximum_location = cv2.minMaxLoc(distance)
    required_px = (
        magnet_radius_cm + margin_cm
    ) * PICKUP_RASTER_PX_PER_CM

    centroid = polygon_centroid(vertices)
    centroid_px = np.round(
        (centroid - minimum) * PICKUP_RASTER_PX_PER_CM
    ).astype(int)
    centroid_is_safe = (
        0 <= centroid_px[0] < mask.shape[1]
        and 0 <= centroid_px[1] < mask.shape[0]
        and distance[centroid_px[1], centroid_px[0]] >= required_px
    )
    if centroid_is_safe:
        selected_px = centroid_px
    else:
        feasible_y, feasible_x = np.where(distance >= required_px)
        if len(feasible_x):
            candidates = np.column_stack((feasible_x, feasible_y))
            selected_px = candidates[
                np.argmin(
                    np.sum((candidates - centroid_px) ** 2, axis=1)
                )
            ]
        else:
            selected_px = np.asarray(maximum_location, dtype=int)

    selected = (
        selected_px.astype(np.float64) / PICKUP_RASTER_PX_PER_CM
        + minimum
    )
    clearance_cm = float(
        distance[selected_px[1], selected_px[0]]
        / PICKUP_RASTER_PX_PER_CM
    )
    return selected, clearance_cm, clearance_cm >= (
        magnet_radius_cm + margin_cm
    )


def normalize_angle_degrees(angle: float) -> float:
    angle = (angle + 180.0) % 360.0 - 180.0
    if abs(angle + 180.0) < 1e-9:
        return 180.0
    return angle


def encode_stm32_frame(command: dict) -> bytes:
    """Encode one fixed-size raw UART frame for the STM32F103.

    The received frame order is the execution order, so no step or piece ID
    is transmitted.  A CRC8/XOR covers the header and five signed int16
    values, allowing the controller to discard a corrupted frame quickly.
    Positions retain integer-millimetre precision while rotation is rounded to
    a whole degree, so no decimal component is sent to the controller.
    """
    values = (
        int(round(float(command["pick_x_cm"]) * 10.0)),
        int(round(float(command["pick_y_cm"]) * 10.0)),
        int(round(float(command["place_x_cm"]) * 10.0)),
        int(round(float(command["place_y_cm"]) * 10.0)),
        int(round(float(command["rotate_deg_clockwise"]))),
    )
    if any(value < -32768 or value > 32767 for value in values):
        raise ValueError("STM32 motion field exceeds int16 range")
    payload = struct.pack("<2s5h", STM32_FRAME_HEADER, *values)
    checksum = 0
    for value in payload:
        checksum ^= value
    return payload + bytes((checksum,))


def format_stm32_frame(command: dict) -> str:
    """Return a hex representation for debug files without changing UART data."""
    return encode_stm32_frame(command).hex().upper()


def send_stm32_frames(
    commands: list[dict],
    device_path: str,
) -> tuple[bool, str]:
    """Write raw fixed-size motion frames to a UART character device."""
    payload = b"".join(encode_stm32_frame(command) for command in commands)
    try:
        descriptor = os.open(
            device_path,
            os.O_WRONLY | os.O_NOCTTY | os.O_NONBLOCK,
        )
    except OSError as error:
        return False, "{}: {}".format(device_path, error)

    try:
        sent = 0
        while sent < len(payload):
            written = os.write(descriptor, payload[sent:])
            if written <= 0:
                raise OSError("serial write returned no data")
            sent += written
    except OSError as error:
        return False, "{}: {}".format(device_path, error)
    finally:
        os.close(descriptor)
    return True, "{} binary frames ({} bytes) -> {}".format(
        len(commands),
        len(payload),
        device_path,
    )


def point_to_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    point_x = float(point[0])
    point_y = float(point[1])
    start_x = float(start[0])
    start_y = float(start[1])
    direction_x = float(end[0]) - start_x
    direction_y = float(end[1]) - start_y
    denominator = (
        direction_x * direction_x
        + direction_y * direction_y
    )
    if denominator <= 1e-12:
        return math.hypot(point_x - start_x, point_y - start_y)
    fraction = (
        (point_x - start_x) * direction_x
        + (point_y - start_y) * direction_y
    ) / denominator
    fraction = min(1.0, max(0.0, fraction))
    closest_x = start_x + fraction * direction_x
    closest_y = start_y + fraction * direction_y
    return math.hypot(point_x - closest_x, point_y - closest_y)


def segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    """Return whether two closed 2-D segments touch or cross."""
    def cross(origin: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        left_x = float(left[0]) - float(origin[0])
        left_y = float(left[1]) - float(origin[1])
        right_x = float(right[0]) - float(origin[0])
        right_y = float(right[1]) - float(origin[1])
        return left_x * right_y - left_y * right_x

    first_left = cross(first_start, first_end, second_start)
    first_right = cross(first_start, first_end, second_end)
    second_left = cross(second_start, second_end, first_start)
    second_right = cross(second_start, second_end, first_end)
    tolerance = 1e-9
    if (
        abs(first_left) <= tolerance
        and min(first_start[0], first_end[0]) - tolerance <= second_start[0]
        <= max(first_start[0], first_end[0]) + tolerance
        and min(first_start[1], first_end[1]) - tolerance <= second_start[1]
        <= max(first_start[1], first_end[1]) + tolerance
    ):
        return True
    if (
        abs(first_right) <= tolerance
        and min(first_start[0], first_end[0]) - tolerance <= second_end[0]
        <= max(first_start[0], first_end[0]) + tolerance
        and min(first_start[1], first_end[1]) - tolerance <= second_end[1]
        <= max(first_start[1], first_end[1]) + tolerance
    ):
        return True
    if (
        abs(second_left) <= tolerance
        and min(second_start[0], second_end[0]) - tolerance <= first_start[0]
        <= max(second_start[0], second_end[0]) + tolerance
        and min(second_start[1], second_end[1]) - tolerance <= first_start[1]
        <= max(second_start[1], second_end[1]) + tolerance
    ):
        return True
    if (
        abs(second_right) <= tolerance
        and min(second_start[0], second_end[0]) - tolerance <= first_end[0]
        <= max(second_start[0], second_end[0]) + tolerance
        and min(second_start[1], second_end[1]) - tolerance <= first_end[1]
        <= max(second_start[1], second_end[1]) + tolerance
    ):
        return True
    return (
        (first_left > tolerance) != (first_right > tolerance)
        and (second_left > tolerance) != (second_right > tolerance)
    )


def polygon_minimum_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return the exact boundary gap for two simple polygons in centimetres."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (
        cv2.pointPolygonTest(first.astype(np.float32), tuple(second[0]), False)
        >= 0
        or cv2.pointPolygonTest(second.astype(np.float32), tuple(first[0]), False)
        >= 0
    ):
        return 0.0

    minimum = float("inf")
    for first_index, first_start in enumerate(first):
        first_end = first[(first_index + 1) % len(first)]
        for second_index, second_start in enumerate(second):
            second_end = second[(second_index + 1) % len(second)]
            if segments_intersect(
                first_start,
                first_end,
                second_start,
                second_end,
            ):
                return 0.0
            minimum = min(
                minimum,
                point_to_segment_distance(first_start, second_start, second_end),
                point_to_segment_distance(first_end, second_start, second_end),
                point_to_segment_distance(second_start, first_start, first_end),
                point_to_segment_distance(second_end, first_start, first_end),
            )
    return minimum


def fit_targets_in_workspace(
    targets: dict[int, np.ndarray],
) -> np.ndarray | None:
    """Return one common translation that keeps the layout in the target zone."""
    all_vertices = np.vstack(list(targets.values()))
    minimum = np.min(all_vertices, axis=0)
    maximum = np.max(all_vertices, axis=0)
    lower = np.asarray(
        [
            ASSEMBLY_WORKSPACE_EDGE_MARGIN_CM,
            A4_HALF_HEIGHT_CM + TARGET_REGION_CENTERLINE_CLEARANCE_CM,
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            A4_WIDTH_CM - ASSEMBLY_WORKSPACE_EDGE_MARGIN_CM,
            A4_HEIGHT_CM - ASSEMBLY_WORKSPACE_EDGE_MARGIN_CM,
        ],
        dtype=np.float64,
    )
    if np.any(maximum - minimum > upper - lower + 1e-9):
        return None
    # Preserve the canonical position whenever it already fits, otherwise
    # shift only as much as needed to keep the whole expanded layout legal.
    return np.minimum(
        np.maximum(lower - minimum, 0.0),
        upper - maximum,
    )


def compensated_targets(
    solution: Solution,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    float,
    float,
    float,
    np.ndarray,
]:
    """Return target polygons/offsets with execution-only clearance.

    The scale applies to the vector from the completed layout centre to each
    piece centroid.  It therefore changes only translations; piece geometry
    and solved rotations remain exact.  Increase the scale in small bounded
    steps until every pair of final piece boundaries has the configured
    minimum seam gap.  When that gap cannot fit, return the largest legal,
    non-overlapping layout rather than rejecting an otherwise valid task.
    """
    placements = list(solution.placements)
    base_vertices = {
        item.piece_id: np.asarray(item.vertices, dtype=np.float64)
        for item in placements
    }
    all_vertices = np.vstack(list(base_vertices.values()))
    layout_center = 0.5 * (
        np.min(all_vertices, axis=0)
        + np.max(all_vertices, axis=0)
    )
    radial_vectors = {
        item.piece_id: (
            polygon_centroid(base_vertices[item.piece_id])
            - layout_center
        )
        for item in placements
    }
    total_area = sum(
        polygon_area(base_vertices[item.piece_id])
        for item in placements
    )

    best = None
    scale = ASSEMBLY_CLEARANCE_SCALE
    while scale <= ASSEMBLY_CLEARANCE_SEARCH_LIMIT + 1e-9:
        offsets: dict[int, np.ndarray] = {}
        targets: dict[int, np.ndarray] = {}
        for item in placements:
            vertices = base_vertices[item.piece_id]
            offset = radial_vectors[item.piece_id] * (scale - 1.0)
            offsets[item.piece_id] = offset
            targets[item.piece_id] = vertices + offset

        workspace_offset = fit_targets_in_workspace(targets)
        if workspace_offset is None:
            break
        for piece_id in targets:
            targets[piece_id] = targets[piece_id] + workspace_offset
            offsets[piece_id] = offsets[piece_id] + workspace_offset

        minimum_gap_cm = float("inf")
        for first_index, first in enumerate(placements):
            for second in placements[first_index + 1:]:
                minimum_gap_cm = min(
                    minimum_gap_cm,
                    polygon_minimum_distance(
                        targets[first.piece_id],
                        targets[second.piece_id],
                    ),
                )
        # A strictly positive exact polygon gap proves that the pair cannot
        # overlap.  The former implementation still rasterized all six pairs
        # on every 0.005 scale step, even though all rejected steps only need
        # the gap test and every accepted step is already disjoint.
        overlap_ratio = 0.0
        best = (
            targets,
            offsets,
            scale,
            overlap_ratio,
            minimum_gap_cm,
            workspace_offset,
        )
        if (
            minimum_gap_cm >= ASSEMBLY_MIN_SEAM_GAP_CM
        ):
            return best
        scale += ASSEMBLY_CLEARANCE_SCALE_STEP

    if best is None:
        raise RuntimeError("Target layout does not fit inside the A4 target zone")
    targets, offsets, scale, _, minimum_gap_cm, workspace_offset = best
    overlap_area = 0.0
    for first_index, first in enumerate(placements):
        for second in placements[first_index + 1:]:
            overlap_area += overlap_area_raster(
                targets[first.piece_id],
                targets[second.piece_id],
            )
    best = (
        targets,
        offsets,
        scale,
        overlap_area / max(total_area, 1e-9),
        minimum_gap_cm,
        workspace_offset,
    )
    return best


def build_motion_plan(
    source_pieces_cm: list[np.ndarray],
    solution: Solution,
    output_path: str | None = None,
) -> dict:
    plan_started = time.monotonic()
    placements = {
        placement.piece_id: placement
        for placement in solution.placements
    }
    (
        target_vertices,
        target_offsets,
        clearance_scale,
        compensated_overlap_ratio,
        minimum_gap_cm,
        workspace_offset,
    ) = compensated_targets(solution)
    clearance_finished = time.monotonic()
    pieces_payload = []
    commands = []

    # Place pieces from largest to smallest for a stable base.
    order = sorted(
        range(len(source_pieces_cm)),
        key=lambda piece_id: abs(
            cv2.contourArea(
                np.asarray(source_pieces_cm[piece_id], dtype=np.float32)
            )
        ),
        reverse=True,
    )
    for sequence, piece_id in enumerate(order, start=1):
        source = np.asarray(source_pieces_cm[piece_id], dtype=np.float64)
        placement = placements[piece_id]
        target_offset = target_offsets[piece_id]
        pickup_source, clearance, pickup_safe = safe_pickup_point(source)
        pickup_target = (
            placement.rotation @ pickup_source + placement.translation
            + target_offset
        )
        source_centroid = polygon_centroid(source)
        target_centroid = apply_transform(
            source_centroid.reshape(1, 2),
            placement.rotation,
            placement.translation,
        )[0] + target_offset
        angle_deg = normalize_angle_degrees(
            math.degrees(
                math.atan2(
                    placement.rotation[1, 0],
                    placement.rotation[0, 0],
                )
            )
        )
        piece_payload = {
            "id": piece_id + 1,
            "source_vertices_cm": np.round(source, 4).tolist(),
            "target_vertices_cm": np.round(
                target_vertices[piece_id], 4
            ).tolist(),
            "ideal_target_vertices_cm": np.round(
                placement.vertices, 4
            ).tolist(),
            "source_centroid_cm": np.round(
                source_centroid, 4
            ).tolist(),
            "target_centroid_cm": np.round(
                target_centroid, 4
            ).tolist(),
            "target_clearance_offset_cm": np.round(
                target_offset, 4
            ).tolist(),
            "pickup_source_cm": np.round(
                pickup_source, 4
            ).tolist(),
            "pickup_target_cm": np.round(
                pickup_target, 4
            ).tolist(),
            "pickup_clearance_cm": round(clearance, 4),
            "pickup_safe": bool(pickup_safe),
            "rotation_deg_clockwise": round(angle_deg, 4),
        }
        pieces_payload.append(piece_payload)
        commands.append(
            {
                "sequence": sequence,
                "piece_id": piece_id + 1,
                "pick_x_cm": round(float(pickup_source[0]), 4),
                "pick_y_cm": round(float(pickup_source[1]), 4),
                "place_x_cm": round(float(pickup_target[0]), 4),
                "place_y_cm": round(float(pickup_target[1]), 4),
                "rotate_deg_clockwise": round(angle_deg, 4),
                "pickup_clearance_cm": round(clearance, 4),
                "pickup_safe": bool(pickup_safe),
                "actions": [
                    "MOVE_XY_PICK",
                    "Z_DOWN",
                    "MAGNET_ON",
                    "Z_UP",
                    "ROTATE",
                    "MOVE_XY_PLACE",
                    "Z_DOWN",
                    "MAGNET_OFF",
                    "Z_UP",
                    "ROTATE_HOME",
                ],
            }
        )
    pickup_finished = time.monotonic()

    plan = {
        "protocol": PROTOCOL_NAME,
        "stm32_frame_protocol": STM32_FRAME_PROTOCOL,
        "created_unix": int(time.time()),
        "units": "cm",
        "coordinate_frame": {
            "origin": "A4_top_left",
            "x_positive": "right",
            "y_positive": "down",
            "rotation_positive": "clockwise",
        },
        "sheet": {
            "width_cm": 21.0,
            "height_cm": 29.7,
            "source_region": "upper_half",
            "target_region": "lower_half",
            "target_min_y_cm": round(
                A4_HALF_HEIGHT_CM
                + TARGET_REGION_CENTERLINE_CLEARANCE_CM,
                4,
            ),
        },
        "tool": {
            "magnet_radius_cm": MAGNET_RADIUS_CM,
            "pickup_margin_cm": PICKUP_MARGIN_CM,
        },
        "solution": {
            "piece_count": len(source_pieces_cm),
            "rectangularity": round(solution.rectangularity, 6),
            "target_width_cm": round(solution.width_cm, 4),
            "target_height_cm": round(solution.height_cm, 4),
            "search_nodes": solution.search_nodes,
            "motion_clearance": {
                "mode": "minimum_seam_gap_radial_expansion",
                "scale": round(clearance_scale, 4),
                "search_limit": ASSEMBLY_CLEARANCE_SEARCH_LIMIT,
                "workspace_offset_cm": np.round(
                    workspace_offset,
                    4,
                ).tolist(),
                "minimum_seam_gap_cm": ASSEMBLY_MIN_SEAM_GAP_CM,
                "achieved_minimum_gap_cm": round(minimum_gap_cm, 4),
                "gap_verified": bool(
                    minimum_gap_cm >= ASSEMBLY_MIN_SEAM_GAP_CM
                ),
                "overlap_ratio": round(
                    compensated_overlap_ratio, 6
                ),
                "overlap_verified": bool(
                    compensated_overlap_ratio
                    <= ASSEMBLY_MAX_OVERLAP_RATIO
                ),
                "rotations_unchanged": True,
            },
        },
        "pieces": pieces_payload,
        "commands": commands,
        "stm32_frames": [
            format_stm32_frame(command)
            for command in commands
        ],
    }
    print(
        "Motion plan detail: clearance={:.3f}s pickup_payload={:.3f}s "
        "total={:.3f}s".format(
            clearance_finished - plan_started,
            pickup_finished - clearance_finished,
            pickup_finished - plan_started,
        )
    )
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return plan
