"""Fixed 640x360 live-screen pixel to C8T6 pulse calibration."""

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
SCREEN_WIDTH = 640
SCREEN_PREVIEW_HEIGHT = 360
PULSE_X_LIMITS = (0, 8000)
PULSE_Y_LIMITS = (0, 10500)
# The measured grid ends near x=454 at 10000 pulses. The rail extension to
# 10500 uses the fitted edge trend up to x=467; pulse limits remain the final
# safety gate for the mildly y-dependent cubic boundary.
CALIBRATED_SCREEN_BOUNDS = (190.0, 57.0, 467.0, 272.0)

# Terms: 1, x, y, x2, xy, y2, x3, x2y, xy2, y3.
PULSE_X_COEFFICIENTS = (
    3382.016535833,
    -217.623208688,
    -6590.574615444,
    234.102373910,
    88.247747704,
    28.439880469,
    -482.104757871,
    -1191.783644873,
    -63.415431923,
    -371.221525399,
)
PULSE_Y_COEFFICIENTS = (
    4823.841308162,
    11647.700085979,
    -228.279383508,
    267.941614165,
    82.450851172,
    88.055944358,
    2996.762593080,
    383.156858962,
    399.270962081,
    296.410694542,
)


def _terms(screen_x, screen_y):
    raw_x = float(screen_x) * (IMAGE_WIDTH - 1) / (SCREEN_WIDTH - 1)
    raw_y = (
        float(screen_y)
        * (IMAGE_HEIGHT - 1)
        / (SCREEN_PREVIEW_HEIGHT - 1)
    )
    x = (raw_x - 960.0) / 960.0
    y = (raw_y - 540.0) / 540.0
    x2 = x * x
    y2 = y * y
    return (
        1.0, x, y, x2, x * y, y2,
        x2 * x, x2 * y, x * y2, y2 * y,
    )


def screen_pixel_to_pulse_float(screen_x, screen_y):
    terms = _terms(screen_x, screen_y)
    pulse_x = sum(
        value * coefficient
        for value, coefficient in zip(terms, PULSE_X_COEFFICIENTS)
    )
    pulse_y = sum(
        value * coefficient
        for value, coefficient in zip(terms, PULSE_Y_COEFFICIENTS)
    )
    return pulse_x, pulse_y


def screen_pixel_to_pulse(screen_x, screen_y):
    pulse_x, pulse_y = screen_pixel_to_pulse_float(screen_x, screen_y)
    pulse_x = max(PULSE_X_LIMITS[0], min(PULSE_X_LIMITS[1], pulse_x))
    pulse_y = max(PULSE_Y_LIMITS[0], min(PULSE_Y_LIMITS[1], pulse_y))
    return int(round(pulse_x)), int(round(pulse_y))


def in_calibrated_screen_area(screen_x, screen_y):
    x0, y0, x1, y1 = CALIBRATED_SCREEN_BOUNDS
    return x0 <= screen_x <= x1 and y0 <= screen_y <= y1
