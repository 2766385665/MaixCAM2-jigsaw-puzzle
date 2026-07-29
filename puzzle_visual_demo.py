"""Visual, animated demonstration of the E-problem puzzle solver.

The animation shows four randomly rotated pieces in the upper half of an A4
sheet, runs contour extraction and geometry solving, then moves every piece
to the solved rectangle in the lower half.

Controls:
  SPACE  pause/resume
  R      generate a new random example
  T      switch between T-junction and legacy random examples
  Q/ESC  quit
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path


def ensure_project_python() -> None:
    """Restart with the tested interpreter if an editor selected another one."""
    try:
        __import__("cv2")
        __import__("numpy")
        __import__("PyQt5")
        __import__("shapely")
        return
    except ModuleNotFoundError as error:
        runtime = Path(r"D:\label\python\python.exe")
        current = Path(sys.executable).resolve()
        if runtime.exists() and current != runtime.resolve():
            print(
                "Missing dependency in {}. Switching to {}.".format(
                    current, runtime
                )
            )
            os.execv(
                str(runtime),
                [
                    str(runtime),
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                ],
            )
        raise ModuleNotFoundError(
            "{}; run this project with {}".format(error, runtime)
        ) from error


ensure_project_python()

import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

sys.path.insert(0, str(Path(__file__).resolve().parent))
import puzzle_solver_validation as solver

V3_SOLVER_DIR = (
    Path(__file__).resolve().parent
    / "puzzle_motion_v2"
    / "maix_project"
)
sys.path.insert(0, str(V3_SOLVER_DIR))
import puzzle_solver_v3 as global_solver


DISPLAY_SCALE = 25
A4_HEIGHT_CM = 29.7
FPS = 30
MOVE_FRAMES = 32
HOLD_FRAMES = 16
START_HOLD_FRAMES = 45
FINAL_HOLD_FRAMES = 90
HEADER_HEIGHT = 72

PIECE_COLORS = [
    (95, 180, 255),
    (120, 220, 130),
    (230, 160, 90),
    (190, 120, 230),
]


def rotation_matrix(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def normalize_targets(solution: solver.Solution) -> dict[int, np.ndarray]:
    """Rotate and center the solution in the lower half of the A4 sheet."""
    all_points = np.vstack([item.vertices for item in solution.placements])
    rectangle = solver.Polygon(all_points).convex_hull.minimum_rotated_rectangle
    corners = np.asarray(rectangle.exterior.coords[:-1], dtype=np.float64)
    vectors = [
        corners[(index + 1) % 4] - corners[index]
        for index in range(4)
    ]
    longest = max(vectors, key=lambda vector: np.linalg.norm(vector))
    angle = math.atan2(longest[1], longest[0])
    rotate = rotation_matrix(-angle)

    rotated = {
        item.piece_id: item.vertices @ rotate.T
        for item in solution.placements
    }
    combined = np.vstack(list(rotated.values()))
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    size = maximum - minimum

    # If the long side is vertical after normalization, turn it horizontal.
    if size[1] > size[0]:
        quarter_turn = rotation_matrix(-math.pi / 2)
        rotated = {
            piece_id: vertices @ quarter_turn.T
            for piece_id, vertices in rotated.items()
        }
        combined = np.vstack(list(rotated.values()))
        minimum = combined.min(axis=0)
        maximum = combined.max(axis=0)

    center = (minimum + maximum) * 0.5
    destination_center = np.array(
        [solver.A4_WIDTH_CM * 0.5, solver.A4_HALF_HEIGHT_CM + 7.1]
    )
    translation = destination_center - center
    return {
        piece_id: vertices + translation
        for piece_id, vertices in rotated.items()
    }


def rigid_interpolation(
    source: np.ndarray,
    target: np.ndarray,
    progress: float,
) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_edge = source[1] - source[0]
    target_edge = target[1] - target[0]
    angle = math.atan2(target_edge[1], target_edge[0]) - math.atan2(
        source_edge[1], source_edge[0]
    )
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    rotate = rotation_matrix(angle * progress)
    center = source_center * (1.0 - progress) + target_center * progress
    return (source - source_center) @ rotate.T + center


def cm_to_pixels(vertices: np.ndarray) -> np.ndarray:
    points = np.rint(vertices * DISPLAY_SCALE).astype(np.int32)
    points[:, 1] += HEADER_HEIGHT
    return points


def draw_piece(
    canvas: np.ndarray,
    vertices: np.ndarray,
    piece_id: int,
    moving: bool = False,
) -> None:
    points = cm_to_pixels(vertices)
    color = PIECE_COLORS[piece_id % len(PIECE_COLORS)]
    cv2.fillPoly(canvas, [points], color, lineType=cv2.LINE_AA)
    border = (30, 30, 30) if not moving else (0, 255, 255)
    thickness = 2 if not moving else 4
    cv2.polylines(canvas, [points], True, border, thickness, cv2.LINE_AA)
    center = np.rint(points.mean(axis=0)).astype(int)
    cv2.putText(
        canvas,
        str(piece_id + 1),
        tuple(center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )


def find_t_junction(
    targets: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Locate a collinear long edge covered by two complete short edges."""
    edges = []
    for piece_id, vertices in targets.items():
        for edge_id in range(len(vertices)):
            points = np.asarray(
                [
                    vertices[edge_id],
                    vertices[(edge_id + 1) % len(vertices)],
                ],
                dtype=np.float64,
            )
            length = float(np.linalg.norm(points[1] - points[0]))
            edges.append((piece_id, edge_id, points, length))

    best = None
    for long_piece, _, long_edge, long_length in edges:
        if long_length < 7.5:
            continue
        direction = long_edge[1] - long_edge[0]
        unit = direction / long_length
        normal = np.asarray([-unit[1], unit[0]])
        short_candidates = [
            item
            for item in edges
            if item[0] != long_piece
            and 1.8 <= item[3] < long_length - 0.5
        ]
        for first_index, first in enumerate(short_candidates):
            for second in short_candidates[first_index + 1 :]:
                if first[0] == second[0]:
                    continue
                if abs(first[3] + second[3] - long_length) > 0.55:
                    continue
                short_points = np.vstack([first[2], second[2]])
                relative = short_points - long_edge[0]
                line_error = float(np.sum(np.abs(relative @ normal)))
                projections = relative @ unit
                coverage_error = abs(float(np.min(projections)))
                coverage_error += abs(
                    float(np.max(projections)) - long_length
                )
                joins = np.linalg.norm(
                    first[2][:, None, :]
                    - second[2][None, :, :],
                    axis=2,
                )
                join_index = np.unravel_index(
                    int(np.argmin(joins)),
                    joins.shape,
                )
                join_error = float(joins[join_index])
                score = line_error + coverage_error + join_error
                if line_error > 0.8 or join_error > 0.35:
                    continue
                candidate = (
                    score,
                    long_edge,
                    first[2],
                    second[2],
                    np.mean(
                        np.vstack(
                            [
                                first[2][join_index[0]],
                                second[2][join_index[1]],
                            ]
                        ),
                        axis=0,
                    ),
                )
                if best is None or score < best[0]:
                    best = candidate
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def draw_t_junction(
    canvas: np.ndarray,
    targets: dict[int, np.ndarray],
) -> None:
    contact = find_t_junction(targets)
    if contact is None:
        return
    long_edge, short_a, short_b, junction = contact
    long_px = cm_to_pixels(long_edge)
    short_a_px = cm_to_pixels(short_a)
    short_b_px = cm_to_pixels(short_b)
    junction_px = cm_to_pixels(junction.reshape(1, 2))[0]
    cv2.line(
        canvas,
        tuple(long_px[0]),
        tuple(long_px[1]),
        (20, 20, 240),
        9,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        tuple(short_a_px[0]),
        tuple(short_a_px[1]),
        (240, 120, 20),
        5,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        tuple(short_b_px[0]),
        tuple(short_b_px[1]),
        (20, 190, 30),
        5,
        cv2.LINE_AA,
    )
    cv2.circle(
        canvas,
        tuple(junction_px),
        9,
        (0, 225, 255),
        -1,
        cv2.LINE_AA,
    )


def make_canvas(
    initial: dict[int, np.ndarray],
    targets: dict[int, np.ndarray],
    placed_count: int,
    moving_piece: int | None = None,
    progress: float = 0.0,
    status: str = "",
    show_t_junction: bool = False,
) -> np.ndarray:
    width = int(round(solver.A4_WIDTH_CM * DISPLAY_SCALE))
    paper_height = int(round(A4_HEIGHT_CM * DISPLAY_SCALE))
    height = paper_height + HEADER_HEIGHT
    canvas = np.full((height, width, 3), (42, 47, 48), dtype=np.uint8)
    divider_y = HEADER_HEIGHT + int(
        round(solver.A4_HALF_HEIGHT_CM * DISPLAY_SCALE)
    )
    cv2.line(canvas, (0, divider_y), (width - 1, divider_y), (220, 220, 220), 3)

    for piece_id in sorted(initial):
        if piece_id < placed_count:
            vertices = targets[piece_id]
        elif piece_id == moving_piece:
            vertices = rigid_interpolation(
                initial[piece_id],
                targets[piece_id],
                progress,
            )
        else:
            vertices = initial[piece_id]
        draw_piece(canvas, vertices, piece_id, moving=(piece_id == moving_piece))

    if show_t_junction:
        draw_t_junction(canvas, targets)

    cv2.rectangle(
        canvas,
        (0, 0),
        (width - 1, HEADER_HEIGHT - 1),
        (25, 25, 25),
        -1,
    )
    status_scale = 0.72
    status_width = cv2.getTextSize(
        status,
        cv2.FONT_HERSHEY_SIMPLEX,
        status_scale,
        2,
    )[0][0]
    if status_width > width - 28:
        status_scale *= (width - 28) / status_width
    cv2.putText(
        canvas,
        status,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        status_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "SPACE: pause   R: new   T: switch case   Q: quit",
        (14, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_t_junction_target() -> tuple[list[np.ndarray], float, float]:
    """One triangle and three quadrilaterals with a 10 = 4 + 6 seam."""
    pieces = [
        np.asarray(
            [[0.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        ),
        np.asarray(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [4.0, 4.0]]
        ),
        np.asarray(
            [[0.0, 4.0], [10.0, 4.0], [10.0, 6.0], [0.0, 7.0]]
        ),
        np.asarray(
            [[0.0, 7.0], [10.0, 6.0], [10.0, 9.0], [0.0, 9.0]]
        ),
    ]
    return [
        solver.counter_clockwise(piece) for piece in pieces
    ], 10.0, 9.0


def build_example(seed: int, case: str = "t_junction"):
    rng = random.Random(seed)
    if case == "t_junction":
        target_pieces, target_width, target_height = (
            make_t_junction_target()
        )
    elif case == "legacy_random":
        # Each piece has two partial rectangle-boundary edges. The bent spoke
        # also makes the example contain quadrilaterals and pentagons.
        target_pieces, target_width, target_height = (
            solver.make_target_pieces(
                rng,
                bend_mask=0b0010,
                topology="partial_sides",
            )
        )
    else:
        raise ValueError("Unsupported demo case: {}".format(case))
    scattered = solver.scatter_pieces(target_pieces, rng)
    scene = solver.render_scene(scattered, rng)
    observed = solver.extract_piece_polygons(scene)
    if len(observed) != 4:
        raise RuntimeError("Expected four contours, got {}".format(len(observed)))

    if case == "t_junction":
        solution, nodes = global_solver.solve_geometry(
            observed,
            max_seconds=10.0,
        )
    else:
        solution, nodes = solver.solve_geometry(observed)
    if solution is None:
        raise RuntimeError("No valid puzzle solution found")

    initial = {
        piece_id: solver.counter_clockwise(vertices)
        for piece_id, vertices in enumerate(observed)
    }
    targets = normalize_targets(solution)
    metrics = {
        "target_width": target_width,
        "target_height": target_height,
        "rectangularity": solution.rectangularity,
        "nodes": nodes,
        "case": case,
        "seed": seed,
    }
    return initial, targets, metrics


def timeline_frames(initial, targets, metrics):
    piece_count = len(initial)
    case_label = (
        "T junction: 10cm = 4cm + 6cm"
        if metrics["case"] == "t_junction"
        else "Legacy random partial-side case"
    )
    for _ in range(START_HOLD_FRAMES):
        yield make_canvas(
            initial,
            targets,
            0,
            status="{} - detected {} pieces".format(
                case_label,
                piece_count,
            )
            + "  seed={}".format(metrics["seed"]),
        )

    for piece_id in range(piece_count):
        for frame_index in range(MOVE_FRAMES):
            progress = (frame_index + 1) / MOVE_FRAMES
            # Smooth acceleration/deceleration for easier visual inspection.
            progress = progress * progress * (3.0 - 2.0 * progress)
            yield make_canvas(
                initial,
                targets,
                piece_id,
                moving_piece=piece_id,
                progress=progress,
                status="Move piece {} / {}".format(
                    piece_id + 1,
                    piece_count,
                ),
            )
        for _ in range(HOLD_FRAMES):
            yield make_canvas(
                initial,
                targets,
                piece_id + 1,
                status="Piece {} placed".format(piece_id + 1),
            )

    final_status = (
        "Complete {}  IoU={:.3f}  seed={}"
    ).format(
        case_label,
        metrics["rectangularity"],
        metrics["seed"],
    )
    for _ in range(FINAL_HOLD_FRAMES):
        yield make_canvas(
            initial,
            targets,
            piece_count,
            status=final_status,
            show_t_junction=metrics["case"] == "t_junction",
        )


def open_video_writer(path: Path, frame: np.ndarray) -> cv2.VideoWriter:
    frame_height, frame_width = frame.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create video: {}".format(path))
    return writer


def run_headless(seed: int, video_path: Path, case: str) -> int:
    initial, targets, metrics = build_example(seed, case)
    first_frame = make_canvas(initial, targets, 0, status="Preparing...")
    writer = open_video_writer(video_path, first_frame)
    last_frame = first_frame
    try:
        for frame in timeline_frames(initial, targets, metrics):
            writer.write(frame)
            last_frame = frame
    finally:
        writer.release()
    final_image_path = video_path.with_name(
        video_path.stem + "_final.png"
    )
    if not cv2.imwrite(str(final_image_path), last_frame):
        raise RuntimeError(
            "Could not create final image: {}".format(final_image_path)
        )
    print("Video saved:", video_path.resolve())
    print("Final image saved:", final_image_path.resolve())
    print("Metrics:", metrics)
    return 0


class PuzzleDemoWindow(QMainWindow):
    """Qt window used because the installed OpenCV build is headless."""

    def __init__(self, seed: int, video_path: Path, case: str):
        super().__init__()
        self.seed = seed
        self.video_path = video_path
        self.case = case
        self.paused = False
        self.frames = None
        self.writer = None
        self.last_frame = None
        self.loading_future: Future | None = None
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="puzzle-demo",
        )

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.image_label)
        self.setWindowTitle("MaixCAM 2 Puzzle Visual Demo")

        self.timer = QTimer(self)
        self.timer.setInterval(max(1, int(1000 / FPS)))
        self.timer.timeout.connect(self.advance_frame)
        self.loading_timer = QTimer(self)
        self.loading_timer.setInterval(40)
        self.loading_timer.timeout.connect(self.check_generation)
        self.start_example()

    def start_example(self) -> None:
        if (
            self.loading_future is not None
            and not self.loading_future.done()
        ):
            return
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.timer.stop()
        self.frames = None
        self.paused = False
        loading_frame = make_canvas(
            {},
            {},
            0,
            status="Generating random puzzle... seed={}".format(
                self.seed
            ),
        )
        self.last_frame = loading_frame
        self.show_frame(loading_frame)
        self.loading_future = self.executor.submit(
            build_example,
            self.seed,
            self.case,
        )
        self.loading_timer.start()

    def check_generation(self) -> None:
        if self.loading_future is None:
            self.loading_timer.stop()
            return
        if not self.loading_future.done():
            return
        self.loading_timer.stop()
        future = self.loading_future
        self.loading_future = None
        try:
            initial, targets, metrics = future.result()
        except Exception as error:
            message = "Generation failed: {}".format(error)
            print(message)
            error_frame = make_canvas(
                {},
                {},
                0,
                status=message,
            )
            self.last_frame = error_frame
            self.show_frame(error_frame)
            return

        self.frames = timeline_frames(initial, targets, metrics)
        first_frame = make_canvas(
            initial,
            targets,
            0,
            status="Ready - seed={}".format(metrics["seed"]),
        )
        self.writer = open_video_writer(self.video_path, first_frame)
        self.last_frame = first_frame
        self.show_frame(first_frame)
        self.paused = False
        self.timer.start()

    def show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        qt_image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        if self.width() < width or self.height() < height:
            self.resize(width, height)

    def advance_frame(self) -> None:
        if self.paused or self.frames is None:
            return
        try:
            frame = next(self.frames)
        except StopIteration:
            self.timer.stop()
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            return

        self.last_frame = frame
        if self.writer is not None:
            self.writer.write(frame)
        self.show_frame(frame)

    def keyPressEvent(self, event) -> None:
        if event.isAutoRepeat():
            event.accept()
            return
        if event.key() == Qt.Key_Space:
            if self.loading_future is not None:
                return
            self.paused = not self.paused
            if self.paused:
                self.timer.stop()
            else:
                self.timer.start()
            return
        if event.key() == Qt.Key_R:
            if self.loading_future is not None:
                return
            self.seed += 1
            self.start_example()
            return
        if event.key() == Qt.Key_T:
            if self.loading_future is not None:
                return
            self.case = (
                "legacy_random"
                if self.case == "t_junction"
                else "t_junction"
            )
            self.start_example()
            return
        if event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.loading_timer.stop()
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.loading_future is not None:
            self.loading_future.cancel()
            self.loading_future = None
        self.executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("puzzle_validation_output/puzzle_demo.mp4"),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Only create the video; do not open a window",
    )
    parser.add_argument(
        "--case",
        choices=("t_junction", "legacy_random"),
        default="t_junction",
        help="Visual example topology (default: t_junction)",
    )
    args = parser.parse_args()

    if args.headless:
        return run_headless(args.seed, args.video, args.case)

    qt_app = QApplication(sys.argv)
    window = PuzzleDemoWindow(args.seed, args.video, args.case)
    window.show()
    return qt_app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
