"""Regression checks for printed-card contour detection."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIX_DIR = PROJECT_DIR / "maix_project"
if str(MAIX_DIR) not in sys.path:
    sys.path.insert(0, str(MAIX_DIR))

import piece_vision


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
        "123753": 4,
        "130435": 4,
    }
    failures = []
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
    if tested == 0:
        print("SKIP no local contour regression captures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
