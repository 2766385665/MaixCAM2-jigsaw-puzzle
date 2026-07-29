"""Create a deterministic four-piece motion plan without a camera."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import motion_protocol
import puzzle_solver


def rotate_and_center(
    vertices: np.ndarray,
    center: tuple[float, float],
    angle_deg: float,
) -> np.ndarray:
    centroid = puzzle_solver.polygon_centroid(vertices)
    angle = math.radians(angle_deg)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    return (
        (vertices - centroid) @ rotation.T
        + np.asarray(center, dtype=np.float64)
    )


def demo_pieces() -> list[np.ndarray]:
    center = np.asarray([5.0, 3.0])
    corners = [
        np.asarray([0.0, 0.0]),
        np.asarray([10.0, 0.0]),
        np.asarray([10.0, 6.0]),
        np.asarray([0.0, 6.0]),
    ]
    bend_1 = np.asarray([7.5, 1.7])
    bend_2 = np.asarray([7.6, 4.4])
    assembled = [
        np.asarray([corners[0], corners[1], bend_1, center]),
        np.asarray(
            [
                corners[1],
                corners[2],
                bend_2,
                center,
                bend_1,
            ]
        ),
        np.asarray([corners[2], corners[3], center, bend_2]),
        np.asarray([corners[3], corners[0], center]),
    ]
    return [
        rotate_and_center(assembled[0], (5.5, 3.0), 24.0),
        rotate_and_center(assembled[1], (15.8, 3.5), -37.0),
        rotate_and_center(assembled[2], (5.4, 9.8), 68.0),
        rotate_and_center(assembled[3], (16.2, 10.2), -112.0),
    ]


def main() -> int:
    pieces = demo_pieces()
    solution, nodes = puzzle_solver.solve_geometry(pieces)
    if solution is None:
        raise RuntimeError("Demo puzzle has no solution")
    solution.search_nodes = nodes
    final = puzzle_solver.canonical_target_solution(solution)
    output = Path(__file__).resolve().parents[1] / "pc_simulator" / (
        "demo_motion_plan.json"
    )
    motion_protocol.build_motion_plan(
        pieces,
        final,
        str(output),
    )
    print("Demo motion plan:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
