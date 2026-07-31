"""Regression checks for printed-card contour detection."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIX_DIR = PROJECT_DIR / "maix_project"
if str(MAIX_DIR) not in sys.path:
    sys.path.insert(0, str(MAIX_DIR))

import piece_vision


def make_dark_background_case() -> np.ndarray:
    """Four white printed-card fragments on a matte black board."""
    image = np.full(
        (
            piece_vision.RECTIFIED_HEIGHT,
            piece_vision.RECTIFIED_WIDTH,
            3,
        ),
        18,
        dtype=np.uint8,
    )
    polygons = [
        np.asarray([[90, 70], [310, 70], [300, 215], [125, 230]]),
        np.asarray([[430, 75], [625, 90], [610, 230], [445, 220]]),
        np.asarray([[105, 315], [300, 270], [335, 450], [165, 475]]),
        np.asarray([[455, 300], [640, 320], [620, 485], [430, 460]]),
    ]
    for index, polygon in enumerate(polygons):
        cv2.fillPoly(image, [polygon], (238, 238, 238))
        center = np.mean(polygon, axis=0).astype(int)
        cv2.circle(image, tuple(center), 25, (20, 20, 20), -1)
        cv2.line(
            image,
            tuple(center - [45, 25]),
            tuple(center + [45, 25]),
            (20 + index * 35, 40, 190 - index * 25),
            12,
        )
    return image


def check_short_edge_model_selection() -> list[str]:
    """Reject a shallow bevel without deleting a real one-centimetre edge."""
    cases = (
        (
            "synthetic-false-short-edge",
            np.asarray(
                [[50, 50], [300, 50], [300, 220], [260, 224], [50, 220]],
                dtype=np.int32,
            ),
            4,
        ),
        (
            "synthetic-real-short-edge",
            np.asarray(
                [[50, 50], [300, 50], [300, 220], [272, 248], [50, 220]],
                dtype=np.int32,
            ),
            5,
        ),
        (
            "synthetic-near-limit-short-edge",
            np.asarray(
                [[50, 50], [300, 50], [300, 220], [275, 245], [50, 220]],
                dtype=np.int32,
            ),
            5,
        ),
    )
    failures = []
    for name, vertices, expected_edges in cases:
        polygon = piece_vision.fit_polygon(
            vertices.reshape(-1, 1, 2)
        )
        actual_edges = 0 if polygon is None else len(polygon)
        passed = actual_edges == expected_edges
        print(
            "PASS" if passed else "FAIL",
            name,
            f"edges={actual_edges}/{expected_edges}",
        )
        if not passed:
            failures.append(name)
    return failures


def check_collinear_vertex_cleanup() -> list[str]:
    """Remove a straight-edge split while preserving a real fourth corner."""
    cases = (
        (
            "synthetic-collinear-false-corner",
            np.asarray(
                [[50, 50], [300, 50], [300, 220], [170, 224], [50, 220]],
                dtype=np.int32,
            ),
            4,
        ),
        (
            "synthetic-obtuse-real-corner",
            np.asarray(
                [[50, 50], [300, 50], [300, 220], [170, 245], [50, 220]],
                dtype=np.int32,
            ),
            5,
        ),
        (
            "synthetic-long-edge-perspective-split",
            np.asarray(
                [[50, 50], [460, 86], [300, 270], [190, 285], [50, 220]],
                dtype=np.int32,
            ),
            4,
        ),
    )
    failures = []
    for name, vertices, expected_edges in cases:
        cleaned = piece_vision.remove_nearly_collinear_vertices(vertices)
        actual_edges = len(cleaned)
        passed = actual_edges == expected_edges
        print(
            "PASS" if passed else "FAIL",
            name,
            f"edges={actual_edges}/{expected_edges}",
        )
        if not passed:
            failures.append(name)
    return failures


def locate_rectified(stamp: str) -> Path | None:
    roots = [
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        Path("/mnt/c/Users/zhao/Downloads"),
        Path("/mnt/c/Users/zhao/Desktop"),
    ]
    filename = f"pieces_20260729_{stamp}_rectified.jpg"
    for root in roots:
        direct = root / filename
        if direct.exists():
            return direct
        if root.exists():
            for candidate in root.glob(f"*/{filename}"):
                return candidate
    return None


def main() -> int:
    expected_counts = {
        "105427": 4,
        "110517": 4,
        "111559": 4,
        "112159": 4,
        "123753": 4,
        "130435": 4,
        "131923": 4,
    }
    failures = check_short_edge_model_selection()
    failures.extend(check_collinear_vertex_cleanup())
    tested = 0
    for stamp, expected in expected_counts.items():
        path = locate_rectified(stamp)
        if path is None:
            print("SKIP", stamp)
            continue
        image = cv2.imread(str(path))
        pieces, _, threshold = piece_vision.detect_pieces(image)
        valid = all(piece.valid for piece in pieces)
        print(
            "PASS" if len(pieces) == expected and valid else "FAIL",
            stamp,
            "count={}/{} edges={} threshold={}".format(
                len(pieces),
                expected,
                [len(piece.polygon) for piece in pieces],
                threshold,
            ),
        )
        tested += 1
        if len(pieces) != expected or not valid:
            failures.append(stamp)
    dark_image = make_dark_background_case()
    dark_pieces, _, dark_threshold = piece_vision.detect_pieces(
        dark_image
    )
    dark_valid = all(piece.valid for piece in dark_pieces)
    print(
        "PASS"
        if len(dark_pieces) == 4 and dark_valid
        else "FAIL",
        "synthetic-dark-board",
        "count={}/4 edges={} threshold={}".format(
            len(dark_pieces),
            [len(piece.polygon) for piece in dark_pieces],
            dark_threshold,
        ),
    )
    tested += 1
    if len(dark_pieces) != 4 or not dark_valid:
        failures.append("synthetic-dark-board")
    if tested == 0:
        print("SKIP no local contour regression captures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
