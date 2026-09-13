"""Compare historical contour detection before and after 640x360 sampling."""

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


DOWNLOADS = Path("/mnt/c/Users/zhao/Downloads")
SOURCE_A4_WIDTH = int(round(640 * (0.855 - 0.150)))
SOURCE_A4_HEIGHT = int(round(360 * (0.862 - 0.112)))


def detect_full_sheet(image: np.ndarray):
    piece_vision.WORK_X0 = 0.02
    piece_vision.WORK_X1 = 0.98
    piece_vision.WORK_Y0 = 0.02
    piece_vision.WORK_Y1 = 0.98
    piece_vision.SOURCE_CENTER_X_MIN = 0.0
    return piece_vision.detect_pieces(image)[0]


def sampled_at_640(image: np.ndarray) -> np.ndarray:
    low = cv2.resize(
        image,
        (SOURCE_A4_WIDTH, SOURCE_A4_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(
        low,
        (piece_vision.RECTIFIED_WIDTH, piece_vision.RECTIFIED_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )


def signature(pieces) -> tuple[int, tuple[int, ...]]:
    return len(pieces), tuple(sorted(len(piece.polygon) for piece in pieces))


def length_signature(pieces) -> np.ndarray:
    values = [
        length
        for piece in pieces
        for length in piece.edge_lengths_cm
    ]
    return np.asarray(sorted(values), dtype=np.float64)


def main() -> int:
    paths = sorted(DOWNLOADS.glob("pieces_*_rectified.jpg"))
    comparable = 0
    unchanged = 0
    regressions = []
    length_errors = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        baseline = detect_full_sheet(image)
        # Only evaluate captures for which the current detector establishes a
        # valid four-piece baseline; other files cannot isolate resolution.
        if len(baseline) != 4 or not all(piece.valid for piece in baseline):
            continue
        reduced = detect_full_sheet(sampled_at_640(image))
        comparable += 1
        before = signature(baseline)
        after = signature(reduced)
        if before == after and all(piece.valid for piece in reduced):
            unchanged += 1
            base_lengths = length_signature(baseline)
            reduced_lengths = length_signature(reduced)
            if len(base_lengths) == len(reduced_lengths):
                length_errors.extend(np.abs(base_lengths - reduced_lengths))
            print("PASS", path.stem, before, "->", after)
        else:
            regressions.append((path.name, before, after))
            print("FAIL", path.stem, before, "->", after)

    print(
        "SUMMARY captures={} comparable={} unchanged={} regressions={}".format(
            len(paths), comparable, unchanged, len(regressions)
        )
    )
    if length_errors:
        errors = np.asarray(length_errors)
        print(
            "EDGE_ERROR mean={:.3f}cm p95={:.3f}cm max={:.3f}cm".format(
                float(np.mean(errors)),
                float(np.percentile(errors, 95)),
                float(np.max(errors)),
            )
        )
    for name, before, after in regressions:
        print("REGRESSION", name, before, after)
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
