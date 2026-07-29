"""PC regression test for full-edge and one-to-many T-junction solving."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIX_PROJECT_DIR = PROJECT_DIR / "maix_project"
if str(MAIX_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(MAIX_PROJECT_DIR))

import puzzle_solver


def rectangle(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.asarray(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float64,
    )


def t_junction_case() -> list[np.ndarray]:
    """A mandatory T seam: one 10 cm edge meets 4 cm + 6 cm edges."""
    return [
        rectangle(0.0, 0.0, 4.0, 3.0),
        rectangle(4.0, 0.0, 10.0, 3.0),
        rectangle(0.0, 3.0, 10.0, 7.0),
    ]


def full_edge_case() -> list[np.ndarray]:
    return [
        rectangle(0.0, 0.0, 5.0, 3.0),
        rectangle(5.0, 0.0, 10.0, 3.0),
        rectangle(0.0, 3.0, 5.0, 7.0),
        rectangle(5.0, 3.0, 10.0, 7.0),
    ]


def validate(name: str, pieces: list[np.ndarray]) -> None:
    started = time.perf_counter()
    solution, nodes = puzzle_solver.solve_geometry(pieces)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if solution is None:
        raise AssertionError("{}: no solution, nodes={}".format(name, nodes))
    short_side, long_side = sorted(
        (solution.width_cm, solution.height_cm)
    )
    if abs(short_side - 7.0) > 0.02 or abs(long_side - 10.0) > 0.02:
        raise AssertionError(
            "{}: wrong target {:.3f}x{:.3f}".format(
                name,
                solution.width_cm,
                solution.height_cm,
            )
        )
    if solution.rectangularity < 0.999:
        raise AssertionError(
            "{}: rectangularity={:.6f}".format(
                name,
                solution.rectangularity,
            )
        )
    print(
        "PASS {:<16} rectangle={:.3f}x{:.3f} "
        "rectangularity={:.6f} nodes={} time={:.2f}ms".format(
            name,
            solution.width_cm,
            solution.height_cm,
            solution.rectangularity,
            nodes,
            elapsed_ms,
        )
    )


def main() -> int:
    validate("full-edge", full_edge_case())
    validate("T-junction", t_junction_case())
    print("All solver regression tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
