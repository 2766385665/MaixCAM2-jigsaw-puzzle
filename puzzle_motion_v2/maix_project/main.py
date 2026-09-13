"""MaixCAM 2 puzzle recognition, solving and motion-plan application."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import piece_vision
from runtime_pipeline import (
    build_point_to_pulse,
    corrected_cm_to_live_screen,
    draw_motion_plan,
    prepare_detection_frame,
    save_session as _save_session,
    send_motion_plan as _send_motion_plan,
    solve_task as _solve_task,
)
from task_pipeline import RectanglePuzzleTask


APP_VERSION = "5.16-rail-limit-pickup"
SOLVER_MAX_SECONDS = 23.0
OUTPUT_DIR = "/root/puzzle_motion_output"
# Zero keeps the ISP in automatic exposure mode. If repeatable fixed lighting
# later requires manual exposure, set an absolute value in microseconds rather
# than scaling an automatic reading; switching modes also changes ISP gain
# behaviour, so a relative exposure value is not equivalent.
CAMERA_MANUAL_EXPOSURE_US = 0
# UART0 is the MaixCAM2 system UART.  Connect U0T to the STM32 RX, U0R to the
# STM32 TX, and share GND.  The STM32 must ignore boot and system log bytes.
MOTION_SERIAL_DEVICE = "/dev/ttyS0"
MOTION_SERIAL_BAUDRATE = 115200
# UART0 uses the board's default U0T/U0R mapping; do not remap system pins.
MOTION_SERIAL_PIN_FUNCTIONS = {}
COLORS = piece_vision.COLORS
# 任务插槽：接手者可替换为 TangramTask，而无需修改设备循环。
ACTIVE_TASK = RectanglePuzzleTask()


def configure_camera_exposure(cam, camera_module) -> None:
    """入口层适配器：把设备配置转交给运行时流程模块。"""
    from runtime_pipeline import configure_camera_exposure as _configure

    return _configure(cam, camera_module, CAMERA_MANUAL_EXPOSURE_US)


def solve_detected_pieces(pieces, rectified=None, geometry_calibration=None):
    """使用当前任务插槽执行业务流程。"""
    return _solve_task(
        pieces,
        rectified,
        geometry_calibration,
        SOLVER_MAX_SECONDS,
        ACTIVE_TASK,
    )


def solve_with_task(task, pieces, rectified=None, geometry_calibration=None):
    """使用指定任务适配器求解，例如七巧板任务。"""
    return _solve_task(
        pieces,
        rectified,
        geometry_calibration,
        SOLVER_MAX_SECONDS,
        task,
    )


def send_motion_plan(plan: dict) -> tuple[bool, str]:
    """入口层适配器：使用配置的 UART 设备发送运动计划。"""
    return _send_motion_plan(plan, MOTION_SERIAL_DEVICE)


def save_session(raw, rectified, binary, annotated, pieces, plan) -> str:
    """入口层适配器：按当前输出目录保存一次完整会话。"""
    return _save_session(
        raw, rectified, binary, annotated, pieces, plan, OUTPUT_DIR
    )


def run_maix() -> int:
    # 设备层：只有本函数接触 Maix 专用 API，便于 PC 端复用业务模块。
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

    # 通信初始化：UART 失败时仍允许识别和预览运行，便于现场排查。
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

    # 摄像头初始化：先丢弃启动帧，再应用曝光和稀疏几何标定。
    cam = camera.Camera(
        piece_vision.CAMERA_WIDTH,
        piece_vision.CAMERA_HEIGHT,
        image.Format.FMT_BGR888,
        fps=piece_vision.CAMERA_FPS,
    )
    cam.skip_frames(12)
    configure_camera_exposure(cam, camera)
    cam.skip_frames(4)
    disp = display.Display()
    touch = touchscreen.TouchScreen()
    touch.clear()
    geometry_calibration = piece_vision.build_sparse_geometry_calibration(
        piece_vision.CAMERA_WIDTH,
        piece_vision.CAMERA_HEIGHT,
    )
    print(
        "Sparse contour undistortion: {}".format(
            "enabled" if geometry_calibration is not None else "disabled"
        )
    )

    # 交互状态：ready 实时预览，frozen 保留最近一次识别结果。
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

    # 主循环：预览 -> 点击 START -> 采集/识别 -> 求解 -> UART -> 保存。
    while not app.need_exit():
        if mode != "frozen" or last_raw is None:
            maix_frame = cam.read()
            frame = image.image2cv(
                maix_frame,
                ensure_bgr=False,
                copy=True,
            )
            last_raw = frame
            # 实时预览只画引导框，避免每帧执行透视矫正、分割和多边形拟合。
            preview_source = piece_vision.draw_camera_guide(frame)
        else:
            preview_source = draw_motion_plan(
                last_annotated,
                last_plan,
                geometry_calibration,
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
                # 每次抬起触摸都启动一次“采集 -> 识别 -> 求解 -> 指令”闭环。
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
                            draw_motion_plan(
                                last_annotated,
                                last_plan,
                                geometry_calibration,
                            ),
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
                last_raw = prepare_detection_frame(maix_frame, image)
                del maix_frame
                captured_at = time.monotonic()
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

                # 视觉阶段：透视矫正、分割、轮廓拟合和结果标注。
                last_rectified, _ = piece_vision.rectify_a4(last_raw)
                rectified_at = time.monotonic()
                (
                    last_pieces,
                    last_binary,
                    threshold,
                ) = piece_vision.detect_pieces(
                    last_rectified,
                    geometry_calibration,
                )
                detected_at = time.monotonic()
                last_annotated = piece_vision.draw_detection(
                    last_rectified,
                    last_pieces,
                    threshold,
                    (
                        None
                        if geometry_calibration is None
                        else geometry_calibration.uncorrect_points
                    ),
                )
                annotated_at = time.monotonic()
                print(
                    "Vision timing: capture={:.3f}s rectify={:.3f}s "
                    "detect={:.3f}s annotate={:.3f}s".format(
                        captured_at - button_started,
                        rectified_at - captured_at,
                        detected_at - rectified_at,
                        annotated_at - detected_at,
                    )
                )

                # 求解阶段：几何拼接、纹理排序、吸取点安全性检查。
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
                    geometry_calibration,
                )
                print(message)
                # 执行阶段：只有计划通过安全检查才会尝试发送 UART。
                if last_plan is not None:
                    serial_started = time.monotonic()
                    sent, serial_message = send_motion_plan(last_plan)
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
                        message += " UART0 sent"
                # 先显示结果，再写 SD 卡；调试文件较大，不能阻塞操作员反馈。
                result_screen = piece_vision.compose_screen(
                    draw_motion_plan(
                        last_annotated,
                        last_plan,
                        geometry_calibration,
                    ),
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
                    draw_motion_plan(
                        last_annotated,
                        last_plan,
                        geometry_calibration,
                    ),
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
