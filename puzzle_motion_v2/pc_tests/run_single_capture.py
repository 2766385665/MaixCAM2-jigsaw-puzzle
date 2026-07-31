"""Run one saved capture through the current solver for regression checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIX_DIR = PROJECT_DIR / "maix_project"
if str(MAIX_DIR) not in sys.path:
    sys.path.insert(0, str(MAIX_DIR))

import puzzle_solver_v3 as solver
import texture_matcher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("rectified_image", type=Path)
    parser.add_argument("--px-per-cm", type=float, default=40.0)
    parser.add_argument("--seconds", type=float, default=14.0)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    payload = json.loads(
        args.result_json.read_text(encoding="utf-8")
    )
    pieces = [
        np.asarray(piece["vertices_cm"], dtype=np.float64)
        for piece in payload["pieces"]
    ]
    image = cv2.imread(str(args.rectified_image))
    if image is None:
        raise FileNotFoundError(args.rectified_image)
    context = texture_matcher.build_context(
        image,
        args.px_per_cm,
    )
    solution, nodes = solver.solve_geometry(
        pieces,
        max_seconds=args.seconds,
        texture_context=context,
    )
    diagnostics = solver.last_diagnostics()
    output = {
        "solution": solution is not None,
        "nodes": nodes,
        "diagnostics": diagnostics,
    }
    if args.summary:
        output = {
            "solution": solution is not None,
            "nodes": nodes,
            "version": diagnostics.get("version"),
            "card_texture_mode": diagnostics.get("card_texture_mode"),
            "texture_richness": diagnostics.get("texture_richness"),
            "topology_selection": diagnostics.get("topology_selection"),
            "enumerated_topology_count": diagnostics.get(
                "enumerated_topology_count"
            ),
            "elapsed_seconds": diagnostics.get("elapsed_seconds"),
            "stage_seconds": diagnostics.get("stage_seconds"),
            "timed_out": diagnostics.get("timed_out"),
            "best_iou": diagnostics.get("best_iou"),
            "best_overlap_ratio": diagnostics.get("best_overlap_ratio"),
            "fallback_reason": diagnostics.get("fallback_reason"),
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if solution is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
