"""End-to-end geometry validation for the 2026 E-problem puzzle device.

The test pipeline is:
1. Generate four legal polygon pieces from a random target rectangle.
2. Randomly rotate and scatter the pieces in the upper half of an A4 sheet.
3. Render a synthetic overhead-camera image.
4. Segment the pieces and extract polygon contours with OpenCV.
5. Reassemble them by matching edges and searching non-overlapping placements.
6. Verify that the result fills a rectangle of a legal size.

This validates the geometry branch used for plain white pieces. Printed-card
pieces should use the same geometry candidates plus a texture continuity score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def ensure_project_python() -> None:
    """Restart direct runs with the interpreter that owns all dependencies."""
    try:
        __import__("cv2")
        __import__("numpy")
        __import__("shapely")
        return
    except ModuleNotFoundError as error:
        runtime = Path(r"D:\label\python\python.exe")
        current = Path(sys.executable).resolve()
        if (
            __name__ == "__main__"
            and runtime.exists()
            and current != runtime.resolve()
        ):
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
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


A4_WIDTH_CM = 21.0
A4_HALF_HEIGHT_CM = 14.85
# A 1280 px-wide camera view over a 21 cm A4 sheet gives about 61 px/cm.
PIXELS_PER_CM = 60
EDGE_ABS_TOLERANCE_CM = 0.18
EDGE_REL_TOLERANCE = 0.045
MAX_OVERLAP_AREA_CM2 = 0.16
MIN_RECTANGULARITY = 0.980
CONTOUR_EPSILON_RATIO = 0.015


@dataclass
class Placement:
    piece_id: int
    vertices: np.ndarray
    polygon: Polygon
    used_edges: frozenset[int]


@dataclass
class Solution:
    placements: list[Placement]
    rectangularity: float
    width_cm: float
    height_cm: float


def polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def counter_clockwise(vertices: np.ndarray) -> np.ndarray:
    if polygon_area(vertices) < 0:
        return vertices[::-1].copy()
    return vertices.copy()


def polygon_edges(vertices: np.ndarray) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    for index in range(len(vertices)):
        yield index, vertices[index], vertices[(index + 1) % len(vertices)]


def edge_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))


def lengths_match(first: float, second: float) -> bool:
    tolerance = max(
        EDGE_ABS_TOLERANCE_CM,
        EDGE_REL_TOLERANCE * max(first, second),
    )
    return abs(first - second) <= tolerance


def transform_edge_to_edge(
    vertices: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> np.ndarray:
    """Rotate/translate source A->B onto reversed target B->A."""
    source_vector = source_b - source_a
    target_vector = target_a - target_b
    angle = math.atan2(target_vector[1], target_vector[0]) - math.atan2(
        source_vector[1], source_vector[0]
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)

    source_midpoint = (source_a + source_b) * 0.5
    target_midpoint = (target_a + target_b) * 0.5
    translation = target_midpoint - source_midpoint @ rotation.T
    return vertices @ rotation.T + translation


def legal_target_dimensions(width: float, height: float) -> bool:
    short_side, long_side = sorted((width, height))
    return 4.75 <= short_side <= 9.25 and 8.75 <= long_side <= 12.25


def minimum_rectangle_dimensions(polygon: Polygon) -> tuple[float, float]:
    rectangle = polygon.minimum_rotated_rectangle
    coordinates = np.asarray(rectangle.exterior.coords[:-1], dtype=np.float64)
    lengths = [
        float(np.linalg.norm(coordinates[(i + 1) % 4] - coordinates[i]))
        for i in range(4)
    ]
    return min(lengths), max(lengths)


def evaluate_complete(placements: list[Placement]) -> Solution | None:
    polygons = [placement.polygon for placement in placements]
    total_piece_area = sum(polygon.area for polygon in polygons)
    merged = unary_union(polygons)
    rectangle = merged.minimum_rotated_rectangle
    if rectangle.area <= 0:
        return None

    rectangularity = total_piece_area / rectangle.area
    width, height = minimum_rectangle_dimensions(merged)
    if (
        rectangularity >= MIN_RECTANGULARITY
        and legal_target_dimensions(width, height)
    ):
        return Solution(
            placements=list(placements),
            rectangularity=rectangularity,
            width_cm=width,
            height_cm=height,
        )
    return None


def solve_geometry(pieces: list[np.ndarray], max_nodes: int = 120_000) -> tuple[Solution | None, int]:
    """Search all edge-compatible rigid placements for at most four pieces."""
    normalized = [counter_clockwise(np.asarray(piece, dtype=np.float64)) for piece in pieces]
    order = sorted(
        range(len(normalized)),
        key=lambda idx: Polygon(normalized[idx]).area,
        reverse=True,
    )
    anchor_id = order[0]
    anchor_vertices = normalized[anchor_id]
    anchor = Placement(
        anchor_id,
        anchor_vertices,
        Polygon(anchor_vertices),
        frozenset(),
    )
    remaining = set(range(len(normalized))) - {anchor_id}
    best_solution: Solution | None = None
    visited: set[tuple] = set()
    node_count = 0

    def state_key(placements: list[Placement]) -> tuple:
        items = []
        for placement in sorted(placements, key=lambda item: item.piece_id):
            centroid = placement.polygon.centroid
            first_edge = placement.vertices[1] - placement.vertices[0]
            angle = math.atan2(first_edge[1], first_edge[0])
            items.append(
                (
                    placement.piece_id,
                    round(centroid.x, 2),
                    round(centroid.y, 2),
                    round(angle, 2),
                )
            )
        return tuple(items)

    def recurse(placements: list[Placement], unplaced: set[int]) -> bool:
        nonlocal best_solution, node_count
        node_count += 1
        if node_count > max_nodes:
            return False

        key = state_key(placements)
        if key in visited:
            return False
        visited.add(key)

        if not unplaced:
            solution = evaluate_complete(placements)
            if solution and (
                best_solution is None
                or solution.rectangularity > best_solution.rectangularity
            ):
                best_solution = solution
                if solution.rectangularity >= 0.995:
                    return True
            return False

        # Try the most constrained pieces first: longest edge tends to eliminate
        # false candidates quickly.
        candidate_ids = sorted(
            unplaced,
            key=lambda idx: max(
                edge_length(a, b)
                for _, a, b in polygon_edges(normalized[idx])
            ),
            reverse=True,
        )

        for candidate_id in candidate_ids:
            source_vertices = normalized[candidate_id]
            for source_edge_id, source_a, source_b in polygon_edges(source_vertices):
                source_length = edge_length(source_a, source_b)
                for target_index, target in enumerate(placements):
                    for target_edge_id, target_a, target_b in polygon_edges(target.vertices):
                        if target_edge_id in target.used_edges:
                            continue
                        target_length = edge_length(target_a, target_b)
                        if not lengths_match(source_length, target_length):
                            continue

                        transformed = transform_edge_to_edge(
                            source_vertices,
                            source_a,
                            source_b,
                            target_a,
                            target_b,
                        )
                        candidate_polygon = Polygon(transformed)
                        if not candidate_polygon.is_valid or candidate_polygon.area <= 0.1:
                            continue

                        overlaps = False
                        for placed in placements:
                            if candidate_polygon.intersection(placed.polygon).area > MAX_OVERLAP_AREA_CM2:
                                overlaps = True
                                break
                        if overlaps:
                            continue

                        updated_target = Placement(
                            target.piece_id,
                            target.vertices,
                            target.polygon,
                            target.used_edges | {target_edge_id},
                        )
                        next_placements = list(placements)
                        next_placements[target_index] = updated_target
                        next_placements.append(
                            Placement(
                                candidate_id,
                                transformed,
                                candidate_polygon,
                                frozenset({source_edge_id}),
                            )
                        )
                        if recurse(next_placements, unplaced - {candidate_id}):
                            return True
        return False

    recurse([anchor], remaining)
    return best_solution, node_count


def make_target_pieces(
    rng: random.Random,
    bend_mask: int | None = None,
    topology: str = "random",
) -> tuple[list[np.ndarray], float, float]:
    """Partition a rectangle into legal 3-, 4-, and 5-edge pieces.

    Two topology families are mixed:

    ``full_sides`` connects the four rectangle corners to an interior junction.
    Each piece owns one complete rectangle side.

    ``partial_sides`` starts the four internal spokes from points inside the
    rectangle sides. Each rectangle side is therefore split between two pieces,
    and every piece has two partial exterior boundary edges. This specifically
    avoids assuming that a piece owns a complete target-rectangle side.

    A spoke may contain one bend, producing 3-, 4-, or 5-edge pieces.
    """
    if topology not in ("random", "full_sides", "partial_sides"):
        raise ValueError("Unsupported puzzle topology: {}".format(topology))
    if topology == "random":
        topology = rng.choices(
            ("full_sides", "partial_sides"),
            weights=(0.4, 0.6),
            k=1,
        )[0]

    width = rng.uniform(9.2, 11.8)
    height = rng.uniform(5.2, 8.8)
    corners = [
        np.array([0.0, 0.0]),
        np.array([width, 0.0]),
        np.array([width, height]),
        np.array([0.0, height]),
    ]
    target_rectangle = Polygon(corners)

    if topology == "full_sides":
        # 0011 produces one triangle, two quadrilaterals and one pentagon.
        masks = (0b0000, 0b0001, 0b0011, 0b0101, 0b0111, 0b1111)
    else:
        # Only bend the longer left/right spokes. Bending the shorter
        # top/bottom spokes can conflict with the 2 cm minimum-edge rule when
        # the legal rectangle height is close to its 5 cm lower limit.
        masks = (0b0000, 0b0010, 0b1000, 0b1010)

    selected_mask = rng.choice(masks) if bend_mask is None else bend_mask & 0xF
    if topology == "partial_sides" and selected_mask not in masks:
        raise ValueError(
            "partial_sides bend_mask must use only the left/right spokes"
        )

    for _ in range(400):
        center = np.array(
            [
                rng.uniform(width * 0.36, width * 0.64),
                rng.uniform(height * 0.36, height * 0.64),
            ],
            dtype=np.float64,
        )
        if topology == "partial_sides":
            boundary_points = [
                np.array([rng.uniform(2.05, width - 2.05), 0.0]),
                np.array([width, rng.uniform(2.05, height - 2.05)]),
                np.array([rng.uniform(2.05, width - 2.05), height]),
                np.array([0.0, rng.uniform(2.05, height - 2.05)]),
            ]
        else:
            boundary_points = corners

        spokes: list[list[np.ndarray]] = []

        for point_id, boundary_point in enumerate(boundary_points):
            spoke = [boundary_point]
            if selected_mask & (1 << point_id):
                vector = center - boundary_point
                distance = float(np.linalg.norm(vector))
                perpendicular = np.array([-vector[1], vector[0]]) / distance
                fraction = rng.uniform(0.43, 0.57)
                offset = rng.uniform(0.45, 0.90) * rng.choice((-1.0, 1.0))
                bend = (
                    boundary_point
                    + vector * fraction
                    + perpendicular * offset
                )
                spoke.append(bend)
            spoke.append(center)
            spokes.append(spoke)

        pieces: list[np.ndarray] = []
        for side_id in range(4):
            next_id = (side_id + 1) % 4
            if topology == "partial_sides":
                # The path between adjacent side-interior points passes through
                # one rectangle corner. Both resulting outer edges are only
                # portions of their respective rectangle sides.
                vertices = [
                    boundary_points[side_id],
                    corners[next_id],
                    boundary_points[next_id],
                ]
            else:
                vertices = [corners[side_id], corners[next_id]]
            # Walk from the next corner to the center.
            vertices.extend(spokes[next_id][1:])
            # Walk from the center back toward this side's first corner.
            vertices.extend(reversed(spokes[side_id][1:-1]))
            pieces.append(counter_clockwise(np.asarray(vertices)))

        polygons = [Polygon(piece) for piece in pieces]
        all_edges_legal = all(
            edge_length(a, b) >= 2.0
            for piece in pieces
            for _, a, b in polygon_edges(piece)
        )
        shapes_legal = all(
            polygon.is_valid
            and polygon.area > 1.0
            and 3 <= len(piece) <= 5
            for piece, polygon in zip(pieces, polygons)
        )
        if not all_edges_legal or not shapes_legal:
            continue

        merged = unary_union(polygons)
        overlap_area = sum(polygon.area for polygon in polygons) - merged.area
        coverage_error = merged.symmetric_difference(target_rectangle).area
        if overlap_area <= 1e-6 and coverage_error <= 1e-6:
            return pieces, width, height

    raise RuntimeError("Could not generate a legal mixed-edge puzzle")


def count_rectangle_boundary_edges(
    piece: np.ndarray,
    width: float,
    height: float,
    tolerance: float = 1e-6,
) -> int:
    count = 0
    for _, first, second in polygon_edges(piece):
        on_left = abs(first[0]) <= tolerance and abs(second[0]) <= tolerance
        on_right = (
            abs(first[0] - width) <= tolerance
            and abs(second[0] - width) <= tolerance
        )
        on_top = abs(first[1]) <= tolerance and abs(second[1]) <= tolerance
        on_bottom = (
            abs(first[1] - height) <= tolerance
            and abs(second[1] - height) <= tolerance
        )
        count += on_left or on_right or on_top or on_bottom
    return int(count)


def rotate_about_centroid(vertices: np.ndarray, angle: float) -> np.ndarray:
    centroid = np.mean(vertices, axis=0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return (vertices - centroid) @ rotation.T


def scatter_pieces(pieces: list[np.ndarray], rng: random.Random) -> list[np.ndarray]:
    ordered = sorted(pieces, key=lambda item: Polygon(item).area, reverse=True)

    # A greedy packing can occasionally trap the final piece even though a
    # valid random arrangement exists. Restart the synthetic placement instead
    # of incorrectly counting that generator artifact as a vision failure.
    for _restart in range(30):
        placed: list[np.ndarray] = []
        placed_polygons: list[Polygon] = []

        for piece in ordered:
            accepted = False
            for _ in range(1500):
                rotated = rotate_about_centroid(piece, rng.uniform(-math.pi, math.pi))
                minimum = rotated.min(axis=0)
                maximum = rotated.max(axis=0)
                size = maximum - minimum
                if size[0] >= A4_WIDTH_CM - 0.4 or size[1] >= A4_HALF_HEIGHT_CM - 0.4:
                    continue
                shift = np.array(
                    [
                        rng.uniform(0.2 - minimum[0], A4_WIDTH_CM - 0.2 - maximum[0]),
                        rng.uniform(0.2 - minimum[1], A4_HALF_HEIGHT_CM - 0.2 - maximum[1]),
                    ]
                )
                candidate = rotated + shift
                candidate_polygon = Polygon(candidate)
                if all(
                    candidate_polygon.distance(existing) >= 0.22
                    for existing in placed_polygons
                ):
                    placed.append(candidate)
                    placed_polygons.append(candidate_polygon)
                    accepted = True
                    break
            if not accepted:
                break
        if len(placed) == len(ordered):
            return placed

    raise RuntimeError("Could not scatter pieces without overlap")


def render_scene(scattered: list[np.ndarray], rng: random.Random) -> np.ndarray:
    width_px = int(round(A4_WIDTH_CM * PIXELS_PER_CM))
    height_px = int(round(A4_HALF_HEIGHT_CM * PIXELS_PER_CM))
    scene = np.full((height_px, width_px, 3), (35, 45, 48), dtype=np.uint8)

    for vertices in scattered:
        points = np.rint(vertices * PIXELS_PER_CM).astype(np.int32)
        cv2.fillPoly(scene, [points], (238, 238, 238), lineType=cv2.LINE_AA)

    noise = np.random.default_rng(rng.randrange(2**32)).normal(
        0.0, 2.0, scene.shape
    )
    scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(scene, (3, 3), 0.45)


def extract_piece_polygons(scene: np.ndarray) -> list[np.ndarray]:
    grayscale = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grayscale, 130, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces: list[np.ndarray] = []

    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(
            contour,
            CONTOUR_EPSILON_RATIO * perimeter,
            True,
        )
        vertices = approximation[:, 0, :].astype(np.float64) / PIXELS_PER_CM
        if 3 <= len(vertices) <= 5:
            pieces.append(counter_clockwise(vertices))
    return pieces


def draw_solution(solution: Solution, path: Path) -> None:
    all_points = np.vstack([item.vertices for item in solution.placements])
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    scale = 45
    margin = 35
    canvas_size = np.ceil((maximum - minimum) * scale).astype(int) + margin * 2
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 245, dtype=np.uint8)
    colors = [(95, 180, 255), (120, 220, 130), (230, 160, 90), (190, 120, 230)]

    for index, placement in enumerate(solution.placements):
        points = np.rint((placement.vertices - minimum) * scale + margin).astype(np.int32)
        cv2.fillPoly(canvas, [points], colors[index % len(colors)], lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [points], True, (30, 30, 30), 2, cv2.LINE_AA)
        center = np.rint(points.mean(axis=0)).astype(int)
        cv2.putText(
            canvas,
            str(placement.piece_id),
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), canvas)


def validate_texture_metric(rng: random.Random, trials: int = 200) -> float:
    """Validate that reversed seam signatures select their true partner."""
    successes = 0
    for _ in range(trials):
        signature = np.asarray([rng.random() for _ in range(96)], dtype=np.float64)
        true_partner = signature[::-1] + np.random.default_rng(
            rng.randrange(2**32)
        ).normal(0, 0.02, signature.shape)
        distractors = [
            np.asarray([rng.random() for _ in range(96)], dtype=np.float64)
            for _ in range(7)
        ]
        candidates = [true_partner] + distractors
        rng.shuffle(candidates)
        costs = [
            float(np.mean((signature - candidate[::-1]) ** 2))
            for candidate in candidates
        ]
        if candidates[int(np.argmin(costs))] is true_partner:
            successes += 1
    return successes / trials


def make_random_non_puzzle(rng: random.Random) -> list[np.ndarray]:
    """Create four unrelated legal-sized triangles for false-positive testing."""
    pieces: list[np.ndarray] = []
    while len(pieces) < 4:
        width = rng.uniform(2.5, 7.0)
        apex = np.array(
            [rng.uniform(0.2, width - 0.2), rng.uniform(2.2, 6.5)]
        )
        triangle = np.array([[0.0, 0.0], [width, 0.0], apex])
        lengths = [
            edge_length(a, b)
            for _, a, b in polygon_edges(triangle)
        ]
        if min(lengths) >= 2.0 and Polygon(triangle).area >= 3.0:
            angle = rng.uniform(-math.pi, math.pi)
            pieces.append(rotate_about_centroid(triangle, angle))
    return pieces


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--negative-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("puzzle_validation_output"),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    successes = 0
    extraction_failures = 0
    search_failures = 0
    elapsed_times: list[float] = []
    node_counts: list[int] = []
    piece_edge_histogram = {3: 0, 4: 0, 5: 0}
    target_boundary_edge_histogram: dict[int, int] = {}
    first_scene = None
    first_solution = None

    for trial in range(args.trials):
        target_pieces, target_width, target_height = make_target_pieces(rng)
        for piece in target_pieces:
            boundary_edge_count = count_rectangle_boundary_edges(
                piece,
                target_width,
                target_height,
            )
            target_boundary_edge_histogram[boundary_edge_count] = (
                target_boundary_edge_histogram.get(boundary_edge_count, 0) + 1
            )
        try:
            scattered = scatter_pieces(target_pieces, rng)
        except RuntimeError:
            extraction_failures += 1
            continue
        scene = render_scene(scattered, rng)
        observed = extract_piece_polygons(scene)
        if len(observed) != 4:
            extraction_failures += 1
            continue
        for piece in observed:
            if len(piece) in piece_edge_histogram:
                piece_edge_histogram[len(piece)] += 1

        start = time.perf_counter()
        solution, nodes = solve_geometry(observed)
        elapsed_times.append(time.perf_counter() - start)
        node_counts.append(nodes)
        if solution is None:
            search_failures += 1
            continue

        successes += 1
        if first_scene is None:
            first_scene = scene
            first_solution = solution

    texture_accuracy = validate_texture_metric(rng)
    false_acceptances = 0
    for _ in range(args.negative_trials):
        unrelated_pieces = make_random_non_puzzle(rng)
        false_solution, _ = solve_geometry(unrelated_pieces)
        false_acceptances += false_solution is not None

    if first_scene is not None and first_solution is not None:
        cv2.imwrite(str(args.output / "synthetic_camera_input.png"), first_scene)
        draw_solution(first_solution, args.output / "reconstructed_puzzle.png")

    summary = {
        "seed": args.seed,
        "trials": args.trials,
        "successes": successes,
        "success_rate": successes / args.trials if args.trials else 0.0,
        "extraction_failures": extraction_failures,
        "search_failures": search_failures,
        "mean_solve_ms": (
            1000.0 * sum(elapsed_times) / len(elapsed_times)
            if elapsed_times
            else None
        ),
        "max_solve_ms": (
            1000.0 * max(elapsed_times) if elapsed_times else None
        ),
        "mean_search_nodes": (
            sum(node_counts) / len(node_counts) if node_counts else None
        ),
        "detected_piece_edge_histogram": piece_edge_histogram,
        "target_boundary_edge_histogram": target_boundary_edge_histogram,
        "texture_pairing_accuracy": texture_accuracy,
        "negative_trials": args.negative_trials,
        "false_acceptances": false_acceptances,
        "false_acceptance_rate": (
            false_acceptances / args.negative_trials
            if args.negative_trials
            else 0.0
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if successes == args.trials and false_acceptances == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
