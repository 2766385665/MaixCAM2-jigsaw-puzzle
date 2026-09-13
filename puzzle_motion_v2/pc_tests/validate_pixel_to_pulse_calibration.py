"""Regression for held-out screen-pixel to rail-pulse measurements."""

import math
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1] / "maix_project"
sys.path.insert(0, str(PROJECT_DIR))

import pixel_to_pulse_calibration as calibration


VALIDATION_POINTS = (
    ((351, 136), (5000, 6000)),
    ((269, 191), (3000, 3000)),
    ((375, 81), (7000, 7000)),
)
MAX_ERROR_PULSE = 60.0


def main():
    errors = []
    for screen, expected in VALIDATION_POINTS:
        predicted = calibration.screen_pixel_to_pulse(*screen)
        error = math.hypot(
            predicted[0] - expected[0],
            predicted[1] - expected[1],
        )
        errors.append(error)
        print(
            "screen={} expected={} predicted={} error={:.1f}pulse".format(
                screen,
                expected,
                predicted,
                error,
            )
        )
    maximum = max(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    assert maximum <= MAX_ERROR_PULSE, (rmse, maximum)
    print("PASS pulse calibration RMSE={:.1f} MAX={:.1f}".format(rmse, maximum))


if __name__ == "__main__":
    main()
