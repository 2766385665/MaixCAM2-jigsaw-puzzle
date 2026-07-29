"""PC simulation of XY/Z/theta puzzle-piece transport commands.

The simulator consumes the JSON emitted by the MaixCAM 2 project. It uses
PyQt5 for the window because the installed OpenCV package is headless.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SCALE = 25.0
MARGIN = 22
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
SHEET_WIDTH = int(A4_WIDTH_CM * SCALE)
SHEET_HEIGHT = int(A4_HEIGHT_CM * SCALE)
PANEL_WIDTH = 390
CANVAS_WIDTH = MARGIN * 3 + SHEET_WIDTH + PANEL_WIDTH
CANVAS_HEIGHT = MARGIN * 2 + SHEET_HEIGHT
FPS = 25

COLORS = [
    (0, 200, 255),
    (80, 210, 80),
    (255, 150, 40),
    (210, 80, 210),
]


@dataclass
class FrameState:
    polygons: dict[int, np.ndarray]
    head: np.ndarray
    trace: list[np.ndarray]
    phase: str
    piece_id: int | None
    angle_deg: float
    magnet_on: bool
    z_down: bool
    progress: float


def rotation_matrix_clockwise(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )


def transport_polygon(
    source: np.ndarray,
    pickup_source: np.ndarray,
    pickup_current: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    rotation = rotation_matrix_clockwise(angle_deg)
    return (
        (source - pickup_source) @ rotation.T
        + pickup_current
    )


def append_transition(
    frames: list[FrameState],
    polygons: dict[int, np.ndarray],
    trace: list[np.ndarray],
    head_start: np.ndarray,
    head_end: np.ndarray,
    phase: str,
    piece_id: int | None,
    angle_start: float,
    angle_end: float,
    frame_count: int,
    source: np.ndarray | None = None,
    pickup_source: np.ndarray | None = None,
    carry_piece: bool = False,
    magnet_on: bool = False,
    z_down: bool = False,
) -> np.ndarray:
    for index in range(1, frame_count + 1):
        fraction = index / frame_count
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        head = head_start * (1.0 - smooth) + head_end * smooth
        angle = angle_start * (1.0 - smooth) + angle_end * smooth
        current = {
            key: value.copy() for key, value in polygons.items()
        }
        if (
            carry_piece
            and piece_id is not None
            and source is not None
            and pickup_source is not None
        ):
            current[piece_id] = transport_polygon(
                source,
                pickup_source,
                head,
                angle,
            )
        trace.append(head.copy())
        frames.append(
            FrameState(
                polygons=current,
                head=head.copy(),
                trace=[point.copy() for point in trace],
                phase=phase,
                piece_id=piece_id,
                angle_deg=angle,
                magnet_on=magnet_on,
                z_down=z_down,
                progress=fraction,
            )
        )
    return head_end.copy()


def max_set_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return max(
        float(np.min(np.linalg.norm(second - point, axis=1)))
        for point in first
    )


def validate_plan(plan: dict) -> dict:
    piece_by_id = {
        int(piece["id"]): piece for piece in plan["pieces"]
    }
    maximum_error = 0.0
    unsafe = []
    for command in plan["commands"]:
        piece_id = int(command["piece_id"])
        piece = piece_by_id[piece_id]
        source = np.asarray(piece["source_vertices_cm"], dtype=np.float64)
        target = np.asarray(piece["target_vertices_cm"], dtype=np.float64)
        pickup_source = np.asarray(
            piece["pickup_source_cm"],
            dtype=np.float64,
        )
        pickup_target = np.asarray(
            piece["pickup_target_cm"],
            dtype=np.float64,
        )
        predicted = transport_polygon(
            source,
            pickup_source,
            pickup_target,
            float(piece["rotation_deg_clockwise"]),
        )
        error = max(
            max_set_distance(predicted, target),
            max_set_distance(target, predicted),
        )
        maximum_error = max(maximum_error, error)
        if not piece["pickup_safe"]:
            unsafe.append(piece_id)
    return {
        "maximum_final_vertex_error_cm": maximum_error,
        "unsafe_pickup_piece_ids": unsafe,
        "valid": maximum_error <= 0.03 and not unsafe,
    }


def build_animation(plan: dict) -> list[FrameState]:
    piece_by_id = {
        int(piece["id"]): piece for piece in plan["pieces"]
    }
    polygons = {
        piece_id: np.asarray(
            piece["source_vertices_cm"],
            dtype=np.float64,
        )
        for piece_id, piece in piece_by_id.items()
    }
    head = np.asarray([1.0, 1.0], dtype=np.float64)
    trace = [head.copy()]
    frames = [
        FrameState(
            polygons={
                key: value.copy() for key, value in polygons.items()
            },
            head=head.copy(),
            trace=[head.copy()],
            phase="READY",
            piece_id=None,
            angle_deg=0.0,
            magnet_on=False,
            z_down=False,
            progress=0.0,
        )
    ] * 20

    for command in plan["commands"]:
        piece_id = int(command["piece_id"])
        piece = piece_by_id[piece_id]
        source = polygons[piece_id].copy()
        pickup_source = np.asarray(
            piece["pickup_source_cm"],
            dtype=np.float64,
        )
        pickup_target = np.asarray(
            piece["pickup_target_cm"],
            dtype=np.float64,
        )
        angle = float(piece["rotation_deg_clockwise"])

        head = append_transition(
            frames, polygons, trace, head, pickup_source,
            "MOVE XY TO PICK", piece_id, 0.0, 0.0, 24,
        )
        head = append_transition(
            frames, polygons, trace, head, head,
            "Z DOWN / MAGNET ON", piece_id, 0.0, 0.0, 10,
            magnet_on=True, z_down=True,
        )
        head = append_transition(
            frames, polygons, trace, head, head,
            "Z UP", piece_id, 0.0, 0.0, 8,
            source=source, pickup_source=pickup_source,
            carry_piece=True, magnet_on=True,
        )
        head = append_transition(
            frames, polygons, trace, head, head,
            "ROTATE", piece_id, 0.0, angle, 24,
            source=source, pickup_source=pickup_source,
            carry_piece=True, magnet_on=True,
        )
        head = append_transition(
            frames, polygons, trace, head, pickup_target,
            "MOVE XY TO PLACE", piece_id, angle, angle, 32,
            source=source, pickup_source=pickup_source,
            carry_piece=True, magnet_on=True,
        )
        head = append_transition(
            frames, polygons, trace, head, head,
            "Z DOWN / MAGNET OFF", piece_id, angle, angle, 10,
            source=source, pickup_source=pickup_source,
            carry_piece=True, z_down=True,
        )
        polygons[piece_id] = transport_polygon(
            source,
            pickup_source,
            pickup_target,
            angle,
        )
        head = append_transition(
            frames, polygons, trace, head, head,
            "Z UP / PLACED", piece_id, angle, angle, 10,
        )
        head = append_transition(
            frames, polygons, trace, head, head,
            "THETA HOME", piece_id, angle, 0.0, 12,
        )

    head = append_transition(
        frames, polygons, trace, head, np.asarray([1.0, 1.0]),
        "RETURN HOME", None, 0.0, 0.0, 28,
    )
    for _ in range(50):
        frames.append(
            FrameState(
                polygons={
                    key: value.copy() for key, value in polygons.items()
                },
                head=head.copy(),
                trace=[point.copy() for point in trace],
                phase="COMPLETE",
                piece_id=None,
                angle_deg=0.0,
                magnet_on=False,
                z_down=False,
                progress=1.0,
            )
        )
    return frames


def sheet_point(point_cm: np.ndarray) -> tuple[int, int]:
    return (
        MARGIN + int(round(float(point_cm[0]) * SCALE)),
        MARGIN + int(round(float(point_cm[1]) * SCALE)),
    )


def draw_text(
    canvas: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float = 0.58,
    color: tuple[int, int, int] = (230, 230, 230),
) -> None:
    cv2.putText(
        canvas,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def render_frame(
    plan: dict,
    validation: dict,
    state: FrameState,
    frame_id: int,
    frame_count: int,
) -> np.ndarray:
    canvas = np.full(
        (CANVAS_HEIGHT, CANVAS_WIDTH, 3),
        (26, 28, 26),
        dtype=np.uint8,
    )
    sheet_tl = (MARGIN, MARGIN)
    sheet_br = (MARGIN + SHEET_WIDTH, MARGIN + SHEET_HEIGHT)
    cv2.rectangle(canvas, sheet_tl, sheet_br, (242, 242, 242), -1)
    cv2.rectangle(canvas, sheet_tl, sheet_br, (120, 120, 120), 2)
    half_y = MARGIN + int(A4_HEIGHT_CM * 0.5 * SCALE)
    cv2.line(
        canvas,
        (MARGIN, half_y),
        (MARGIN + SHEET_WIDTH, half_y),
        (120, 120, 120),
        2,
    )

    for point_a, point_b in zip(state.trace, state.trace[1:]):
        cv2.line(
            canvas,
            sheet_point(point_a),
            sheet_point(point_b),
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

    for piece_id, polygon in state.polygons.items():
        color = COLORS[(piece_id - 1) % len(COLORS)]
        points = np.asarray(
            [sheet_point(point) for point in polygon],
            dtype=np.int32,
        )
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
        cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)
        center = np.mean(points, axis=0).astype(int)
        cv2.putText(
            canvas,
            "P{}".format(piece_id),
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    hx, hy = sheet_point(state.head)
    cv2.line(
        canvas,
        (MARGIN, hy),
        (MARGIN + SHEET_WIDTH, hy),
        (80, 80, 80),
        1,
    )
    cv2.line(
        canvas,
        (hx, MARGIN),
        (hx, MARGIN + SHEET_HEIGHT),
        (80, 80, 80),
        1,
    )
    head_color = (0, 0, 255) if state.magnet_on else (30, 30, 30)
    radius = 10 if not state.z_down else 14
    cv2.circle(canvas, (hx, hy), radius, head_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (hx, hy), radius, (255, 255, 255), 2)

    panel_x = MARGIN * 2 + SHEET_WIDTH
    draw_text(canvas, "XY-Z-THETA MOTION SIM", (panel_x, 45), 0.72)
    draw_text(
        canvas,
        "Phase: {}".format(state.phase),
        (panel_x, 85),
        0.62,
        (0, 230, 255),
    )
    draw_text(
        canvas,
        "Piece: {}".format(
            "-" if state.piece_id is None else state.piece_id
        ),
        (panel_x, 120),
    )
    draw_text(
        canvas,
        "Head: ({:.2f}, {:.2f}) cm".format(
            state.head[0], state.head[1]
        ),
        (panel_x, 150),
    )
    draw_text(
        canvas,
        "Theta: {:+.1f} deg CW".format(state.angle_deg),
        (panel_x, 180),
    )
    draw_text(
        canvas,
        "Magnet: {}".format("ON" if state.magnet_on else "OFF"),
        (panel_x, 210),
    )
    solution = plan["solution"]
    draw_text(
        canvas,
        "Target: {:.2f} x {:.2f} cm".format(
            solution["target_width_cm"],
            solution["target_height_cm"],
        ),
        (panel_x, 270),
    )
    draw_text(
        canvas,
        "Rectangularity: {:.4f}".format(
            solution["rectangularity"]
        ),
        (panel_x, 300),
    )
    draw_text(
        canvas,
        "Command error: {:.3f} mm".format(
            validation["maximum_final_vertex_error_cm"] * 10.0
        ),
        (panel_x, 330),
    )
    status_color = (
        (80, 230, 80) if validation["valid"] else (0, 0, 255)
    )
    draw_text(
        canvas,
        "PLAN {}".format("VALID" if validation["valid"] else "INVALID"),
        (panel_x, 370),
        0.8,
        status_color,
    )
    draw_text(
        canvas,
        "Space: pause/resume",
        (panel_x, CANVAS_HEIGHT - 75),
        0.5,
    )
    draw_text(
        canvas,
        "R: restart    Esc: quit",
        (panel_x, CANVAS_HEIGHT - 45),
        0.5,
    )
    progress = frame_id / max(1, frame_count - 1)
    cv2.rectangle(
        canvas,
        (panel_x, CANVAS_HEIGHT - 20),
        (CANVAS_WIDTH - MARGIN, CANVAS_HEIGHT - 10),
        (70, 70, 70),
        -1,
    )
    cv2.rectangle(
        canvas,
        (panel_x, CANVAS_HEIGHT - 20),
        (
            panel_x
            + int((CANVAS_WIDTH - MARGIN - panel_x) * progress),
            CANVAS_HEIGHT - 10,
        ),
        (0, 210, 255),
        -1,
    )
    return canvas


def write_video(
    plan: dict,
    validation: dict,
    frames: list[FrameState],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot create video: {}".format(path))
    for frame_id, state in enumerate(frames):
        writer.write(
            render_frame(
                plan,
                validation,
                state,
                frame_id,
                len(frames),
            )
        )
    writer.release()
    print("Video saved:", path.resolve())


def run_window(
    plan: dict,
    validation: dict,
    frames: list[FrameState],
) -> int:
    try:
        from PyQt5.QtCore import QTimer, Qt
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import QApplication, QLabel
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyQt5 is required; run with D:\\label\\python\\python.exe"
        ) from error

    class Window(QLabel):
        def __init__(self):
            super().__init__()
            self.frame_id = 0
            self.paused = False
            self.setWindowTitle("MaixCAM 2 Puzzle Motion Simulator")
            self.setFocusPolicy(Qt.StrongFocus)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.advance)
            self.timer.start(round(1000 / FPS))
            self.show_frame()

        def show_frame(self):
            bgr = render_frame(
                plan,
                validation,
                frames[self.frame_id],
                self.frame_id,
                len(frames),
            )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format_RGB888,
            ).copy()
            self.setPixmap(QPixmap.fromImage(image))
            self.adjustSize()

        def advance(self):
            if not self.paused and self.frame_id < len(frames) - 1:
                self.frame_id += 1
                self.show_frame()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Space:
                self.paused = not self.paused
            elif event.key() == Qt.Key_R:
                self.frame_id = 0
                self.paused = False
                self.show_frame()
            elif event.key() == Qt.Key_Escape:
                self.close()
            else:
                super().keyPressEvent(event)

    application = QApplication([])
    window = Window()
    window.show()
    return application.exec_()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("motion_simulation.mp4"),
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("protocol") != "maixcam2-puzzle-motion-v1":
        raise ValueError("Unsupported motion protocol")
    validation = validate_plan(plan)
    frames = build_animation(plan)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if args.headless:
        write_video(plan, validation, frames, args.video)
        return 0 if validation["valid"] else 1
    return run_window(plan, validation, frames)


if __name__ == "__main__":
    raise SystemExit(main())
