"""运行时流程模块：把视觉结果转换为可执行的拼图搬运计划。

本模块只负责设备无关的业务流程，便于在 PC 上单独测试。Maix 摄像头、
显示屏和触摸屏的生命周期仍由 ``main.py`` 管理。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import motion_protocol
import pixel_to_pulse_calibration
import piece_vision
import puzzle_solver_v3 as puzzle_solver
import texture_matcher
from task_pipeline import PuzzlePipeline, TaskAdapter, create_rectangle_pipeline


def solve_task(
    pieces,
    rectified=None,
    geometry_calibration=None,
    solver_max_seconds: float = 23.0,
    task: TaskAdapter | None = None,
):
    """运行可替换任务；默认使用当前矩形拼图适配器。"""
    pipeline = (
        create_rectangle_pipeline(solver_max_seconds)
        if task is None
        else PuzzlePipeline(task, solver_max_seconds)
    )
    return pipeline.solve(pieces, rectified, geometry_calibration)


def corrected_cm_to_live_screen(point_cm, geometry_calibration):
    """将矫正后的 A4 厘米坐标映射到 640x360 实时预览坐标。"""
    if geometry_calibration is None:
        raise ValueError("Sparse geometry calibration is required for motion")
    corrected_pixels = (
        np.asarray(point_cm, dtype=np.float32) * piece_vision.PX_PER_CM
    )
    camera_pixel = geometry_calibration.corrected_to_camera_points(
        corrected_pixels
    )
    camera_pixel = np.asarray(camera_pixel, dtype=np.float64).reshape(2)
    return np.asarray(
        [
            camera_pixel[0]
            * (pixel_to_pulse_calibration.SCREEN_WIDTH - 1)
            / (piece_vision.CAMERA_WIDTH - 1),
            camera_pixel[1]
            * (pixel_to_pulse_calibration.SCREEN_PREVIEW_HEIGHT - 1)
            / (piece_vision.CAMERA_HEIGHT - 1),
        ],
        dtype=np.float64,
    )


def build_point_to_pulse(geometry_calibration) -> Callable:
    """创建“矫正 A4 坐标 -> 滑轨脉冲”的转换回调，并检查安全边界。"""

    def convert(point_cm):
        screen = corrected_cm_to_live_screen(point_cm, geometry_calibration)
        if not pixel_to_pulse_calibration.in_calibrated_screen_area(
            screen[0], screen[1]
        ):
            raise ValueError(
                "screen point ({:.1f},{:.1f}) outside pulse calibration".format(
                    screen[0], screen[1]
                )
            )
        pulse_float = pixel_to_pulse_calibration.screen_pixel_to_pulse_float(
            screen[0], screen[1]
        )
        x_limits = pixel_to_pulse_calibration.PULSE_X_LIMITS
        y_limits = pixel_to_pulse_calibration.PULSE_Y_LIMITS
        if not (
            x_limits[0] <= pulse_float[0] <= x_limits[1]
            and y_limits[0] <= pulse_float[1] <= y_limits[1]
        ):
            raise ValueError(
                "predicted pulse ({:.1f},{:.1f}) outside rail limits".format(
                    pulse_float[0], pulse_float[1]
                )
            )
        return int(round(pulse_float[0])), int(round(pulse_float[1]))

    return convert


def configure_camera_exposure(cam, camera_module, manual_exposure_us: int) -> None:
    """按配置设置曝光；设为 0 时保持自动曝光，兼容旧固件。"""
    try:
        if manual_exposure_us > 0:
            applied_exposure = int(cam.exposure(manual_exposure_us))
            print("Camera exposure: manual={}us".format(applied_exposure))
        else:
            cam.exp_mode(camera_module.AeMode.Auto)
            print("Camera exposure: auto")
    except Exception as error:
        print("Camera exposure unchanged: {}".format(error))


def solve_detected_pieces(
    pieces,
    rectified=None,
    geometry_calibration=None,
    solver_max_seconds: float = 23.0,
):
    """执行识别结果 -> 几何求解 -> 运动计划的完整业务流程。"""
    # 统一转换为厘米坐标，后续几何算法与相机分辨率解耦。
    solve_started = time.monotonic()
    source_pieces_cm = [
        np.asarray(piece.polygon, dtype=np.float64) / piece_vision.PX_PER_CM
        for piece in pieces
    ]
    # 纹理上下文只建立一次，供求解器对多个几何候选重复评分。
    texture_context = (
        None
        if rectified is None
        else texture_matcher.build_context(
            rectified,
            piece_vision.PX_PER_CM,
            sample_point_transform=(
                None
                if geometry_calibration is None
                else geometry_calibration.uncorrect_points
            ),
        )
    )
    texture_ready = time.monotonic()
    # V3 先做离散拓扑搜索，再做连续姿态优化；返回节点数用于诊断性能。
    solution, nodes = puzzle_solver.solve_geometry(
        source_pieces_cm,
        max_seconds=solver_max_seconds,
        texture_context=texture_context,
    )
    solver_finished = time.monotonic()
    diagnostics = puzzle_solver.last_diagnostics()
    print("V3 solver diagnostics:", json.dumps(diagnostics, ensure_ascii=False))
    if solution is None:
        print(
            "Solve timing: texture_context={:.3f}s solver={:.3f}s".format(
                texture_ready - solve_started, solver_finished - texture_ready
            )
        )
        if diagnostics.get("timed_out"):
            return None, "Solver timeout {:.1f}s".format(
                diagnostics.get("elapsed_seconds", solver_max_seconds)
            )
        return None, "No solution {:.1f}s nodes={}".format(
            diagnostics.get("elapsed_seconds", 0.0), nodes
        )

    final_solution = puzzle_solver.canonical_target_solution(solution)
    canonical_finished = time.monotonic()
    # 几何解转运动计划时执行标定范围和滑轨限位检查。
    try:
        plan = motion_protocol.build_motion_plan(
            source_pieces_cm,
            final_solution,
            point_to_pulse=build_point_to_pulse(geometry_calibration),
        )
    except ValueError as error:
        return None, "Motion calibration error: {}".format(error)
    plan_finished = time.monotonic()
    print(
        "Solve timing: texture_context={:.3f}s solver={:.3f}s "
        "canonical={:.3f}s motion_plan={:.3f}s total={:.3f}s".format(
            texture_ready - solve_started,
            solver_finished - texture_ready,
            canonical_finished - solver_finished,
            plan_finished - canonical_finished,
            plan_finished - solve_started,
        )
    )
    plan["solver_diagnostics"] = diagnostics
    clearance = plan["solution"]["motion_clearance"]
    if not clearance["overlap_verified"]:
        return None, "Unsafe target overlap {:.2f}%".format(
            100.0 * clearance["overlap_ratio"]
        )
    unsafe = [item["id"] for item in plan["pieces"] if not item["pickup_safe"]]
    if unsafe:
        return plan, "Partial pickup: P{}".format(",P".join(map(str, unsafe)))
    return plan, "Solved {:.1f}x{:.1f} IoU={:.1f}% {:.1f}s".format(
        plan["solution"]["target_width_cm"],
        plan["solution"]["target_height_cm"],
        100.0 * plan["solution"]["rectangularity"],
        diagnostics.get("elapsed_seconds", 0.0),
    )


def send_motion_plan(plan: dict, serial_device: str) -> tuple[bool, str]:
    """向 STM32 发送原始帧；JSON 中保留十六进制帧用于调试。"""
    return motion_protocol.send_stm32_frames(plan["commands"], serial_device)


def prepare_detection_frame(maix_frame, image_module) -> np.ndarray:
    """把一帧 Maix 图像转换为 OpenCV 数组，不在实时预览阶段矫正镜头。"""
    return image_module.image2cv(maix_frame, ensure_bgr=False, copy=True)


def draw_motion_plan(annotated: np.ndarray, plan: dict | None, geometry_calibration=None):
    """在识别图上绘制目标轮廓、吸取点、箭头和旋转角。"""
    canvas = annotated.copy()
    if plan is None:
        return canvas
    scale = piece_vision.PX_PER_CM

    def display_points(points):
        pixels = np.asarray(points, dtype=np.float32) * scale
        if geometry_calibration is not None:
            pixels = geometry_calibration.uncorrect_points(pixels)
        return np.asarray(pixels)

    for item in plan["pieces"]:
        piece_id = int(item["id"])
        color = piece_vision.COLORS[(piece_id - 1) % len(piece_vision.COLORS)]
        target = np.round(display_points(item["target_vertices_cm"])).astype(np.int32)
        source_pick = np.round(display_points(item["pickup_source_cm"])).astype(int)
        target_pick = np.round(display_points(item["pickup_target_cm"])).astype(int)
        cv2.polylines(canvas, [target], True, color, 4, cv2.LINE_AA)
        cv2.circle(canvas, tuple(source_pick), 10, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(target_pick), 10, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, tuple(source_pick), tuple(target_pick), color, 3, cv2.LINE_AA, tipLength=0.04)
        cv2.putText(
            canvas,
            "P{} {:+.1f}deg".format(piece_id, item["rotation_deg_clockwise"]),
            tuple(target_pick + np.asarray([12, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.line(
        canvas,
        (int(piece_vision.RECTIFIED_WIDTH * 0.5), 0),
        (int(piece_vision.RECTIFIED_WIDTH * 0.5), piece_vision.RECTIFIED_HEIGHT - 1),
        (255, 255, 0),
        2,
    )
    return canvas


def save_session(raw, rectified, binary, annotated, pieces, plan, output_dir: str) -> str:
    """保存图像、运动 JSON、STM32 帧和求解诊断，形成可复现的会话包。"""
    prefix = piece_vision.save_debug(raw, rectified, binary, annotated, pieces, output_dir)
    if plan is not None:
        Path(prefix + "_motion.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        Path(prefix + "_stm32.txt").write_text(
            "\n".join(plan["stm32_frames"]) + "\n", encoding="ascii"
        )
    Path(prefix + "_solver_diagnostics.json").write_text(
        json.dumps(puzzle_solver.last_diagnostics(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return prefix
