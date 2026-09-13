"""Lens model used by sparse contour-point undistortion.

The strength/zoom pair mirrors MaixPy ``Image.lens_corr`` without allocating
a corrected full-resolution image.  If an OpenCV calibration is supplied
later, CAMERA_MATRIX and DISTORTION_COEFFICIENTS take precedence.
"""

# Fitted from repeated observations of the same pieces at different image
# positions. This effective center also absorbs small camera/A4 alignment error.
LENS_CORR_STRENGTH = 1.61872752
LENS_CORR_ZOOM = 1.0
LENS_CORR_CENTER_NORMALIZED = (0.42831543, 0.20000080)

# Resolution used when CAMERA_MATRIX was calibrated, as (width, height).
CALIBRATION_IMAGE_SIZE = (1920, 1080)

# OpenCV camera matrix:
# CAMERA_MATRIX = (
#     (fx, 0.0, cx),
#     (0.0, fy, cy),
#     (0.0, 0.0, 1.0),
# )
CAMERA_MATRIX = None

# OpenCV pinhole distortion coefficients. Four, five, eight, twelve or
# fourteen values are supported, for example (k1, k2, p1, p2, k3).
DISTORTION_COEFFICIENTS = None
