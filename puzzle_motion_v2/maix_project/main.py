"""MaixCAM 2 puzzle recognition, solving and motion-plan application."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import motion_protocol
import piece_vision
import puzzle_solver_v3 as puzzle_solver
import texture_matcher


APP_VERSION = "5.9-card-row-swap-texture"
SOLVER_MAX_SECONDS = 23.0
OUTPUT_DIR = "/root/puzzle_motion_output"
# Use MaixCAM2 UART2 so motion frames do not share UART0 with system logs.
# Connect B0/TX to the peer RX, B1/RX to the peer TX, and share GND.
MOTION_SERIAL_DEVICE = "/dev/ttyS2"
MOTION_SERIAL_BAUDRATE = 115200
MOTION_SERIAL_PIN_FUNCTIONS = {
    "B0": "UART2_TX",
    "B1": "UART2_RX",
}
COLORS = piece_vision.COLORS


def solve_detected_pieces(pieces, rectified=None):
    source_pieces_cm = [
        np.asarray(piece.polygon, dtype=np.float64)
        / piece_vision.PX_PER_CM
        for piece in pieces
    ]
    texture_context = (
        None
        if rectified is None
        else texture_matcher.build_context(
            rectified,
            piece_vision.PX_PER_CM,
        )
    )
    solution, nodes = puzzle_solver.solve_geometry(
        source_pieces_cm,
        max_seconds=SOLVER_MAX_SECONDS,
        texture_context=texture_context,
    )
    diagnostics = puzzle_solver.last_diagnostics()
    print(
        "V3 solver diagnostics:",
        json.dumps(diagnostics, ensure_ascii=False),
    )
    if solution is None:
        if diagnostics.get("timed_out"):
            return None, "Solver timeout {:.1f}s".format(
                diagnostics.get(
                    "elapsed_seconds",
                    SOLVER_MAX_SECONDS,
                )
            )
        return None, "No solution {:.1f}s nodes={}".format(
            diagnostics.get("elapsed_seconds", 0.0),
            nodes,
        )
    final_solution = puzzle_solver.canonical_target_solution(solution)
    plan = motion_protocol.build_motion_plan(
        source_pieces_cm,
        final_solution,
    )
    plan["solver_diagnostics"] = diagnostics
    clearance = plan["solution"]["motion_clearance"]
    if not clearance["overlap_verified"]:
        return None, (
            "Unsafe target overlap {:.2f}%".format(
                100.0 * clearance["overlap_ratio"],
            )
        )
    unsafe = [
        item["id"]
        for item in plan["pieces"]
        if not item["pickup_safe"]
    ]
    if unsafe:
        return plan, "Unsafe pickup: P{}".format(
            ",P".join(str(value) for value in unsafe)
        )
    return plan, (
        "Solved {:.1f}x{:.1f} IoU={:.1f}% {:.1f}s".format(
            plan["solution"]["target_width_cm"],
            plan["solution"]["target_height_cm"],
            100.0 * plan["solution"]["rectangularity"],
            diagnostics.get("elapsed_seconds", 0.0),
        )
    )


def send_motion_plan(
    plan: dict,
    serial_port=None,
) -> tuple[bool, str]:
    """Send only the compact STM32 frames, never the debug JSON."""
    payload = "\n".join(plan["stm32_frames"]) + "\n"
    if serial_port is not None:
        try:
            serial_port.write_str(payload)
        except Exception as error:
            return False, "{}: {}".format(
                MOTION_SERIAL_DEVICE,
                error,
            )
        return True, "{} frames -> {}".format(
            len(plan["stm32_frames"]),
            MOTION_SERIAL_DEVICE,
        )
    return motion_protocol.send_stm32_frames(
        plan["stm32_frames"],
        MOTION_SERIAL_DEVICE,
    )


def draw_motion_plan(
    annotated: np.ndarray,
    plan: dict | None,
) -> np.ndarray:
    canvas = annotated.copy()
    if plan is None:
        return canvas
    scale = piece_vision.PX_PER_CM
    for item in plan["pieces"]:
        piece_id = int(item["id"])
        color = COLORS[(piece_id - 1) % len(COLORS)]
        target = np.round(
            np.asarray(item["target_vertices_cm"]) * scale
        ).astype(np.int32)
        source_pick = np.round(
            np.asarray(item["pickup_source_cm"]) * scale
        ).astype(int)
        target_pick = np.round(
            np.asarray(item["pickup_target_cm"]) * scale
        ).astype(int)
        cv2.polylines(
            canvas,
            [target],
            True,
            color,
            4,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            tuple(source_pick),
            10,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            tuple(target_pick),
            10,
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.arrowedLine(
            canvas,
            tuple(source_pick),
            tuple(target_pick),
            color,
            3,
            cv2.LINE_AA,
            tipLength=0.04,
        )
        label = "P{} {:+.1f}deg".format(
            piece_id,
            item["rotation_deg_clockwise"],
        )
        cv2.putText(
            canvas,
            label,
            tuple(target_pick + np.asarray([12, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.line(
        canvas,
        (0, int(piece_vision.RECTIFIED_HEIGHT * 0.5)),
        (
            piece_vision.RECTIFIED_WIDTH - 1,
            int(piece_vision.RECTIFIED_HEIGHT * 0.5),
        ),
        (255, 255, 0),
        2,
    )
    return canvas


def save_session(
    raw,
    rectified,
    binary,
    annotated,
    pieces,
    plan,
) -> str:
    prefix = piece_vision.save_debug(
        raw,
        rectified,
        binary,
        annotated,
        pieces,
        OUTPUT_DIR,
    )
    if plan is not None:
        Path(prefix + "_motion.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        Path(prefix + "_stm32.txt").write_text(
            "\n".join(plan["stm32_frames"]) + "\n",
            encoding="ascii",
        )
    return prefix


def run_maix() -> int:
    from maix import (
        app,
        camera,
        display,
        err,
        image,
        pinmap,
        touchscreen,
        uart,
    )

    motion_serial = None
    try:
        for pin, function in MOTION_SERIAL_PIN_FUNCTIONS.items():
            err.check_raise(
                pinmap.set_pin_function(pin, function),
                "Failed to configure {} as {}".format(
                    pin,
                    function,
                ),
            )
        motion_serial = uart.UART(
            MOTION_SERIAL_DEVICE,
            MOTION_SERIAL_BAUDRATE,
        )
        print(
            "Motion UART ready: {} {}bps".format(
                MOTION_SERIAL_DEVICE,
                MOTION_SERIAL_BAUDRATE,
            )
        )
    except Exception as error:
        print("Motion UART unavailable: {}".format(error))

    cam = camera.Camera(
        piece_vision.CAMERA_WIDTH,
        piece_vision.CAMERA_HEIGHT,
        image.Format.FMT_BGR888,
        fps=piece_vision.CAMERA_FPS,
    )
    cam.skip_frames(12)
    disp = display.Display()
    touch = touchscreen.TouchScreen()
    touch.clear()

    mode = "ready"
    pressed_before = False
    last_touch = (0, 0)
    last_raw = None
    last_rectified = None
    last_binary = None
    last_annotated = None
    last_pieces = []
    last_plan = None
    message = "v{} Align A4, tap START".format(APP_VERSION)

    while not app.need_exit():
        if mode != "frozen" or last_raw is None:
            maix_frame = cam.read()
            frame = image.image2cv(
                maix_frame,
                ensure_bgr=False,
                copy=True,
            )
            last_raw = frame
            # Keep the live path cheap. Full-resolution rectification,
            # multi-threshold segmentation and polygon fitting run once per
            # START tap.
            preview_source = piece_vision.draw_camera_guide(frame)
        else:
            preview_source = draw_motion_plan(
                last_annotated,
                last_plan,
            )

        screen = piece_vision.compose_screen(
            preview_source,
            mode,
            message,
        )
        disp.show(image.cv2image(screen, bgr=True, copy=True))

        if touch.available(0):
            x, y, pressed = touch.read()
            if pressed:
                last_touch = (x, y)
                pressed_before = True
            elif pressed_before:
                pressed_before = False
                # Every tap starts a fresh full capture-to-command cycle.
                x, y = last_touch
                if piece_vision.point_in_rect(
                    x,
                    y,
                    piece_vision.BUTTON_SAVE_RECT,
                ):
                    if last_annotated is None:
                        message = "Run START before saving"
                    else:
                        prefix = save_session(
                            last_raw,
                            last_rectified,
                            last_binary,
                            draw_motion_plan(last_annotated, last_plan),
                            last_pieces,
                            last_plan,
                        )
                        message = "Saved {}".format(Path(prefix).name)
                        print("Saved session: {}_*".format(prefix))
                    continue
                if not piece_vision.point_in_rect(
                    x,
                    y,
                    piece_vision.BUTTON_DETECT_RECT,
                ):
                    continue
                button_started = time.monotonic()
                maix_frame = cam.read()
                last_raw = image.image2cv(
                    maix_frame,
                    ensure_bgr=False,
                    copy=True,
                )
                mode = "frozen"
                message = "Detecting..."
                detecting_source = piece_vision.draw_camera_guide(last_raw)
                detecting_screen = piece_vision.compose_screen(
                    detecting_source,
                    mode,
                    message,
                )
                disp.show(image.cv2image(
                    detecting_screen,
                    bgr=True,
                    copy=True,
                ))

                last_rectified, _ = piece_vision.rectify_a4(last_raw)
                (
                    last_pieces,
                    last_binary,
                    threshold,
                ) = piece_vision.detect_pieces(last_rectified)
                last_annotated = piece_vision.draw_detection(
                    last_rectified,
                    last_pieces,
                    threshold,
                )

                message = "Solving... max {:.0f}s".format(
                    SOLVER_MAX_SECONDS
                )
                solving_screen = piece_vision.compose_screen(
                    last_annotated,
                    mode,
                    message,
                )
                disp.show(image.cv2image(
                    solving_screen,
                    bgr=True,
                    copy=True,
                ))
                last_plan, message = solve_detected_pieces(
                    last_pieces,
                    last_rectified,
                )
                print(message)
                if last_plan is not None:
                    serial_started = time.monotonic()
                    sent, serial_message = send_motion_plan(
                        last_plan,
                        motion_serial,
                    )
                    serial_finished = time.monotonic()
                    print(
                        "Motion serial {}: {}".format(
                            "sent" if sent else "not sent",
                            serial_message,
                        )
                    )
                    print(
                        "Motion timing: button_to_serial={:.3f}s "
                        "serial_write={:.1f}ms total={:.3f}s".format(
                            serial_started - button_started,
                            1000.0 * (serial_finished - serial_started),
                            serial_finished - button_started,
                        )
                    )
                    if sent:
                        message += " UART2 sent"
                # Show the completed placement immediately.  Debug JSON and
                # image saving can take seconds on the SD card, so they must
                # not delay the operator-facing result frame.
                result_screen = piece_vision.compose_screen(
                    draw_motion_plan(last_annotated, last_plan),
                    mode,
                    message,
                )
                disp.show(image.cv2image(
                    result_screen,
                    bgr=True,
                    copy=True,
                ))
                print(
                    "Motion timing: button_to_result={:.3f}s".format(
                        time.monotonic() - button_started,
                    )
                )
                if last_plan is not None:
                    print(json.dumps(
                        last_plan,
                        ensure_ascii=False,
                        indent=2,
                    ))
                prefix = save_session(
                    last_raw,
                    last_rectified,
                    last_binary,
                    draw_motion_plan(last_annotated, last_plan),
                    last_pieces,
                    last_plan,
                )
                print("Saved session: {}_*".format(prefix))
                print(
                    "Motion timing: button_to_saved={:.3f}s".format(
                        time.monotonic() - button_started,
                    )
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_maix())
