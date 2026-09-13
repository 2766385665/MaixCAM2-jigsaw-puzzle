"""Standalone MaixCAM2 manual-exposure test.

Run this file directly in MaixVision. The test is intentionally isolated from
the puzzle application. Use the touch buttons to compare automatic exposure
with explicit exposure times and note the value that best suppresses glare.
"""

import cv2
import numpy as np

from maix import app, camera, display, image, touchscreen, time


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FPS = 20
# MaixCAM2's physical panel is portrait, but applications use a 640x480
# landscape framebuffer and the touch driver reports coordinates in it.
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
PREVIEW_Y = 78
PREVIEW_HEIGHT = 270
INITIAL_MANUAL_EXPOSURE_US = 1000
MIN_EXPOSURE_US = 100
MAX_EXPOSURE_US = 30000
EXPOSURE_FACTOR = 1.25

BUTTONS = (
    ("AUTO", (8, 382, 153, 468)),
    ("-25%", (164, 382, 309, 468)),
    ("+25%", (320, 382, 465, 468)),
    ("RESET", (476, 382, 632, 468)),
)


def point_in_rect(x, y, rect):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def read_camera_value(getter, fallback=-1):
    try:
        return int(getter())
    except Exception:
        return fallback


def set_auto_exposure(cam):
    cam.exp_mode(camera.AeMode.Auto)
    print("Exposure mode=AUTO")


def set_manual_exposure(cam, exposure_us):
    exposure_us = max(MIN_EXPOSURE_US, min(MAX_EXPOSURE_US, int(exposure_us)))
    applied = int(cam.exposure(exposure_us))
    print(
        "Exposure mode=MANUAL requested={}us applied={}us gain={}".format(
            exposure_us,
            applied,
            read_camera_value(cam.gain),
        )
    )
    return applied


def draw_screen(frame, mode, exposure_us, gain, fps):
    canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    preview = cv2.resize(
        frame,
        (SCREEN_WIDTH, PREVIEW_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    canvas[PREVIEW_Y : PREVIEW_Y + PREVIEW_HEIGHT] = preview

    cv2.putText(
        canvas,
        "EXPOSURE TEST",
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "mode={} exposure={}us gain={}".format(mode, exposure_us, gain),
        (14, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Tap +/- for manual exposure. FPS={:.1f}".format(fps),
        (14, 370),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    for label, rect in BUTTONS:
        active = label == "AUTO" and mode == "AUTO"
        colour = (0, 170, 0) if active else (75, 75, 75)
        cv2.rectangle(canvas, rect[:2], rect[2:], colour, -1)
        cv2.rectangle(canvas, rect[:2], rect[2:], (230, 230, 230), 2)
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )[0]
        text_x = rect[0] + (rect[2] - rect[0] - text_size[0]) // 2
        text_y = rect[1] + (rect[3] - rect[1] + text_size[1]) // 2
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def main():
    cam = camera.Camera(
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        image.Format.FMT_BGR888,
        fps=CAMERA_FPS,
    )
    cam.skip_frames(20)
    set_auto_exposure(cam)

    disp = display.Display()
    touch = touchscreen.TouchScreen()
    touch.clear()
    pressed_before = False
    last_touch = (0, 0)
    mode = "AUTO"
    manual_exposure = INITIAL_MANUAL_EXPOSURE_US

    print(
        "Exposure test ready: {}x{}, initial manual={}us".format(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            INITIAL_MANUAL_EXPOSURE_US,
        )
    )

    while not app.need_exit():
        raw = cam.read()
        frame = image.image2cv(raw, ensure_bgr=False, copy=True)
        del raw
        exposure_us = read_camera_value(cam.exposure, manual_exposure)
        gain = read_camera_value(cam.gain)
        screen = draw_screen(frame, mode, exposure_us, gain, time.fps())
        disp.show(image.cv2image(screen, bgr=True, copy=True))

        if touch.available(0):
            x, y, pressed = touch.read()
            if pressed:
                last_touch = (x, y)
                pressed_before = True
            elif pressed_before:
                pressed_before = False
                x, y = last_touch
                for label, rect in BUTTONS:
                    if not point_in_rect(x, y, rect):
                        continue
                    if label == "AUTO":
                        set_auto_exposure(cam)
                        mode = "AUTO"
                    elif label == "RESET":
                        manual_exposure = set_manual_exposure(
                            cam,
                            INITIAL_MANUAL_EXPOSURE_US,
                        )
                        mode = "MANUAL"
                    else:
                        if mode == "AUTO":
                            manual_exposure = read_camera_value(
                                cam.exposure,
                                INITIAL_MANUAL_EXPOSURE_US,
                            )
                        factor = (
                            1.0 / EXPOSURE_FACTOR
                            if label == "-25%"
                            else EXPOSURE_FACTOR
                        )
                        manual_exposure = set_manual_exposure(
                            cam,
                            round(manual_exposure * factor),
                        )
                        mode = "MANUAL"
                    break


if __name__ == "__main__":
    main()
