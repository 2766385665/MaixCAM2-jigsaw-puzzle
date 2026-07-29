"""Pickup-point planning and shared motion-command JSON protocol."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from puzzle_solver import (
    Solution,
    apply_transform,
    polygon_centroid,
)


PROTOCOL_NAME = "maixcam2-puzzle-motion-v1"
PICKUP_RASTER_PX_PER_CM = 40.0
MAGNET_RADIUS_CM = 0.50
PICKUP_MARGIN_CM = 0.20


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


def build_motion_plan(
    source_pieces_cm: list[np.ndarray],
    solution: Solution,
    output_path: str | None = None,
) -> dict:
    placements = {
        placement.piece_id: placement
        for placement in solution.placements
    }
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
        pickup_source, clearance, pickup_safe = safe_pickup_point(source)
        pickup_target = (
            placement.rotation @ pickup_source + placement.translation
        )
        source_centroid = polygon_centroid(source)
        target_centroid = apply_transform(
            source_centroid.reshape(1, 2),
            placement.rotation,
            placement.translation,
        )[0]
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
                placement.vertices, 4
            ).tolist(),
            "source_centroid_cm": np.round(
                source_centroid, 4
            ).tolist(),
            "target_centroid_cm": np.round(
                target_centroid, 4
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

    plan = {
        "protocol": PROTOCOL_NAME,
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
        },
        "pieces": pieces_payload,
        "commands": commands,
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return plan
