"""可替换的任务适配器和通用求解流程。

设备层（摄像头、触摸屏）不应该知道“矩形拼图”或“七巧板”的规则。
本文件定义稳定的任务接口：新任务只需实现 ``TaskAdapter``，即可复用同一套
采集、计时、诊断、显示和保存流程。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import motion_protocol
import numpy as np
import piece_vision
import puzzle_solver_v3
import texture_matcher


class TaskAdapter(Protocol):
    """任务规则与算法的最小接口。

    ``source_data`` 可以是任意领域对象，不要求新任务沿用当前的厘米多边形
    格式；只有适配器内部需要与自己的 solver/planner 对接。
    """

    name: str

    def prepare_source(self, pieces: list[Any]) -> Any:
        """将视觉输出转换为本任务求解器需要的输入。"""

    def build_texture_context(self, rectified, geometry_calibration) -> Any:
        """可选的图像特征上下文；不需要纹理时返回 ``None``。"""

    def solve(self, source_data: Any, max_seconds: float, texture_context: Any):
        """返回 ``(solution, explored_nodes)``。"""

    def diagnostics(self) -> dict:
        """返回最近一次求解诊断。"""

    def canonicalize(self, solution: Any) -> Any:
        """消除平移/旋转等等价解，得到稳定的目标坐标。"""

    def build_motion_plan(self, source_data, solution, geometry_calibration):
        """将任务解转换为统一运动计划字典。"""

    def validate_plan(self, plan: dict, diagnostics: dict) -> str | None:
        """返回错误/警告文本；返回 ``None`` 表示可以执行。"""

    def success_message(self, plan: dict, diagnostics: dict) -> str:
        """生成显示给操作员的成功状态文本。"""


@dataclass
class RectanglePuzzleTask:
    """当前四块矩形拼图任务的默认适配器。"""

    name: str = "rectangle-puzzle-v3"

    def prepare_source(self, pieces):
        return [
            np.asarray(piece.polygon, dtype=np.float64)
            / piece_vision.PX_PER_CM
            for piece in pieces
        ]

    def build_texture_context(self, rectified, geometry_calibration):
        if rectified is None:
            return None
        return texture_matcher.build_context(
            rectified,
            piece_vision.PX_PER_CM,
            sample_point_transform=(
                None
                if geometry_calibration is None
                else geometry_calibration.uncorrect_points
            ),
        )

    def solve(self, source_data, max_seconds, texture_context):
        return puzzle_solver_v3.solve_geometry(
            source_data,
            max_seconds=max_seconds,
            texture_context=texture_context,
        )

    def diagnostics(self):
        return puzzle_solver_v3.last_diagnostics()

    def canonicalize(self, solution):
        return puzzle_solver_v3.canonical_target_solution(solution)

    def build_motion_plan(self, source_data, solution, geometry_calibration):
        return motion_protocol.build_motion_plan(
            source_data,
            solution,
            point_to_pulse=_build_point_to_pulse(geometry_calibration),
        )

    def validate_plan(self, plan, diagnostics):
        clearance = plan["solution"]["motion_clearance"]
        if not clearance["overlap_verified"]:
            return "Unsafe target overlap {:.2f}%".format(
                100.0 * clearance["overlap_ratio"]
            )
        unsafe = [item["id"] for item in plan["pieces"] if not item["pickup_safe"]]
        if unsafe:
            return "Partial pickup: P{}".format(",P".join(map(str, unsafe)))
        return None

    def success_message(self, plan, diagnostics):
        return "Solved {:.1f}x{:.1f} IoU={:.1f}% {:.1f}s".format(
            plan["solution"]["target_width_cm"],
            plan["solution"]["target_height_cm"],
            100.0 * plan["solution"]["rectangularity"],
            diagnostics.get("elapsed_seconds", 0.0),
        )


def _build_point_to_pulse(geometry_calibration):
    """延迟导入以避免 task_pipeline 与 runtime_pipeline 互相依赖。"""
    from runtime_pipeline import build_point_to_pulse

    return build_point_to_pulse(geometry_calibration)


class PuzzlePipeline:
    """任务无关的求解编排器，集中处理计时、诊断和统一状态文本。"""

    def __init__(self, task: TaskAdapter, solver_max_seconds: float = 23.0):
        self.task = task
        self.solver_max_seconds = float(solver_max_seconds)

    def solve(self, pieces, rectified=None, geometry_calibration=None):
        started = time.monotonic()
        source_data = self.task.prepare_source(pieces)
        texture_context = self.task.build_texture_context(
            rectified, geometry_calibration
        )
        texture_ready = time.monotonic()
        solution, nodes = self.task.solve(
            source_data, self.solver_max_seconds, texture_context
        )
        solver_finished = time.monotonic()
        diagnostics = dict(self.task.diagnostics())
        print("{} diagnostics:".format(self.task.name), json.dumps(diagnostics, ensure_ascii=False))
        if solution is None:
            if diagnostics.get("timed_out"):
                return None, "Solver timeout {:.1f}s".format(
                    diagnostics.get("elapsed_seconds", self.solver_max_seconds)
                )
            return None, "No solution {:.1f}s nodes={}".format(
                diagnostics.get("elapsed_seconds", 0.0), nodes
            )
        canonical = self.task.canonicalize(solution)
        try:
            plan = self.task.build_motion_plan(
                source_data, canonical, geometry_calibration
            )
        except ValueError as error:
            return None, "Motion calibration error: {}".format(error)
        plan["solver_diagnostics"] = diagnostics
        validation_message = self.task.validate_plan(plan, diagnostics)
        elapsed = time.monotonic() - started
        print(
            "Task timing: texture_context={:.3f}s solver={:.3f}s total={:.3f}s".format(
                texture_ready - started,
                solver_finished - texture_ready,
                elapsed,
            )
        )
        if validation_message is not None:
            return plan, validation_message
        return plan, self.task.success_message(plan, diagnostics)


def create_rectangle_pipeline(solver_max_seconds: float = 23.0) -> PuzzlePipeline:
    """创建当前默认矩形拼图流程；七巧板应注册自己的 TaskAdapter。"""
    return PuzzlePipeline(RectanglePuzzleTask(), solver_max_seconds)
