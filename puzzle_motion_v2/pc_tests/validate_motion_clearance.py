"""Regression for execution-only rail-clearance compensation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIX_DIR = PROJECT_DIR / "maix_project"
SIMULATOR_DIR = PROJECT_DIR / "pc_simulator"
if str(MAIX_DIR) not in sys.path:
    sys.path.insert(0, str(MAIX_DIR))
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

import motion_protocol
import pc_motion_simulator
import puzzle_solver


def rectangle(x0, y0, x1, y1):
    return np.asarray(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float64,
    )


def main() -> int:
    # A rail boundary can exclude every point with the preferred 2 mm margin
    # while still leaving enough room for the complete magnet footprint.
    edge_piece = rectangle(0.0, 0.0, 4.0, 2.0)
    edge_pickup, edge_clearance, edge_safe = (
        motion_protocol.safe_pickup_point(
            edge_piece,
            candidate_validator=lambda point: point[0] <= 0.65,
        )
    )
    if not edge_safe:
        raise AssertionError("Rail-limit pickup lost full magnet support")
    if not (
        motion_protocol.MAGNET_RADIUS_CM
        <= edge_clearance
        < motion_protocol.MAGNET_RADIUS_CM
        + motion_protocol.PICKUP_MARGIN_CM
    ):
        raise AssertionError(
            "Rail-limit pickup did not use the reduced-margin fallback"
        )
    if edge_pickup[0] > 0.65:
        raise AssertionError("Rail-limit pickup exceeded its reachable bound")

    # When the rail excludes every full-support point, use the reachable point
    # with maximum clearance and explicitly report it as partial support.
    partial_pickup, partial_clearance, partial_safe = (
        motion_protocol.safe_pickup_point(
            edge_piece,
            candidate_validator=lambda point: point[0] <= 0.45,
        )
    )
    if partial_safe:
        raise AssertionError("Partial rail-limit pickup reported full support")
    if not (
        motion_protocol.PARTIAL_PICKUP_MIN_CLEARANCE_CM
        <= partial_clearance
        < motion_protocol.MAGNET_RADIUS_CM
    ):
        raise AssertionError("Partial pickup clearance is outside its policy")
    if partial_pickup[0] > 0.45:
        raise AssertionError("Partial pickup exceeded its reachable bound")

    # Simulate 1 mm contour/rail overlap on both internal axes.  The solver
    # result itself remains untouched; only the generated motion targets may
    # move outward.
    target = [
        rectangle(0.00, 0.00, 5.05, 3.05),
        rectangle(4.95, 0.00, 10.00, 3.05),
        rectangle(0.00, 2.95, 5.05, 6.00),
        rectangle(4.95, 2.95, 10.00, 6.00),
    ]
    source_shift = np.asarray([2.0, 2.0])
    source = [piece + source_shift for piece in target]
    identity = np.eye(2, dtype=np.float64)
    placements = [
        puzzle_solver.Placement(
            piece_id=index,
            vertices=piece.copy(),
            rotation=identity.copy(),
            translation=-source_shift.copy(),
            used_edges=frozenset(),
        )
        for index, piece in enumerate(target)
    ]
    solution = puzzle_solver.Solution(
        placements=placements,
        rectangularity=0.98,
        width_cm=10.0,
        height_cm=6.0,
    )

    plan = motion_protocol.build_motion_plan(source, solution)
    clearance = plan["solution"]["motion_clearance"]
    if not clearance["overlap_verified"]:
        raise AssertionError(clearance)
    if not clearance["gap_verified"]:
        raise AssertionError(clearance)
    if (
        clearance["achieved_minimum_gap_cm"]
        < motion_protocol.ASSEMBLY_MIN_SEAM_GAP_CM
    ):
        raise AssertionError(clearance)
    if clearance["scale"] <= motion_protocol.ASSEMBLY_CLEARANCE_SCALE:
        raise AssertionError(
            "Synthetic overlap should require adaptive clearance"
        )

    for item, ideal in zip(
        sorted(plan["pieces"], key=lambda value: value["id"]),
        target,
    ):
        reported_ideal = np.asarray(
            item["ideal_target_vertices_cm"],
            dtype=np.float64,
        )
        if not np.allclose(reported_ideal, ideal, atol=1e-4):
            raise AssertionError("The ideal solver layout was modified")
        offset = np.asarray(
            item["target_clearance_offset_cm"],
            dtype=np.float64,
        )
        compensated = np.asarray(
            item["target_vertices_cm"],
            dtype=np.float64,
        )
        if not np.allclose(compensated, ideal + offset, atol=1e-4):
            raise AssertionError("Clearance is not translation-only")

    simulator_report = pc_motion_simulator.validate_plan(plan)
    if not simulator_report["valid"]:
        raise AssertionError(simulator_report)

    print(
        "PASS motion-clearance scale={:.3f} gap={:.3f}cm overlap={:.6f}".format(
            clearance["scale"],
            clearance["achieved_minimum_gap_cm"],
            clearance["overlap_ratio"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
