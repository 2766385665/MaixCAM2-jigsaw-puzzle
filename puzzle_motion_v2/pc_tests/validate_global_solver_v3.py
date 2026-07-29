"""PC validation for the v3 global seam-topology solver."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIX_DIR = PROJECT_DIR / "maix_project"
if str(MAIX_DIR) not in sys.path:
    sys.path.insert(0, str(MAIX_DIR))

import motion_protocol
import puzzle_solver_v3


def locate_dataset(filename: str) -> Path:
    """Find saved captures without depending on a localized folder name."""
    homes = [Path.home()]
    wsl_windows_home = Path("/mnt/c/Users/zhao")
    if wsl_windows_home.exists():
        homes.append(wsl_windows_home)
    direct_candidates = [
        folder / location / filename
        for folder in homes
        for location in ("Downloads", "Desktop")
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    for folder in homes:
        desktop = folder / "Desktop"
        if desktop.exists():
            for candidate in desktop.glob("*/" + filename):
                if candidate.exists():
                    return candidate
    return direct_candidates[0]


DATASETS = [
    locate_dataset("pieces_20260729_153339_result.json"),
    locate_dataset("pieces_20260729_150642_result.json"),
    locate_dataset("pieces_20260729_143400_result.json"),
    locate_dataset("pieces_20260729_122919_result.json"),
    locate_dataset("pieces_20260729_121937_result.json"),
    locate_dataset("pieces_20260729_113525_result.json"),
]


def load_pieces(path: Path) -> list[np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        np.asarray(piece["vertices_cm"], dtype=np.float64)
        for piece in payload["pieces"]
    ]


def main() -> int:
    failures = 0
    output_dir = PROJECT_DIR / "pc_simulator" / "v3_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in DATASETS:
        if not path.exists():
            print("SKIP", path)
            continue
        solution, layouts = puzzle_solver_v3.solve_geometry(
            load_pieces(path),
            max_seconds=30.0,
        )
        diagnostics = puzzle_solver_v3.last_diagnostics()
        if solution is None:
            failures += 1
            print("FAIL", path.name, diagnostics)
            continue
        source_pieces = load_pieces(path)
        final_solution = puzzle_solver_v3.canonical_target_solution(
            solution
        )
        plan = motion_protocol.build_motion_plan(
            source_pieces,
            final_solution,
        )
        plan_path = output_dir / (
            path.stem.replace("_result", "") + "_motion.json"
        )
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        unsafe = [
            item["id"]
            for item in plan["pieces"]
            if not item["pickup_safe"]
        ]
        print(
            "PASS {} rectangle={:.3f}x{:.3f} "
            "iou={:.4f} layouts={} unsafe={} plan={} "
            "diagnostics={}".format(
                path.name,
                solution.width_cm,
                solution.height_cm,
                solution.rectangularity,
                layouts,
                unsafe,
                plan_path,
                diagnostics,
            )
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
