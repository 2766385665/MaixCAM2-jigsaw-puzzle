"""Synthetic random-angle regression for the portable v3 solver."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIX_DIR = PROJECT_DIR / "maix_project"
if str(MAIX_DIR) not in sys.path:
    sys.path.insert(0, str(MAIX_DIR))

import puzzle_solver_v3


VISUAL_DIR = PROJECT_DIR / "pc_simulator" / "v3_validation"


def rectangle(x0, y0, x1, y1):
    return np.asarray(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float64,
    )


def scatter(pieces):
    angles = [37.0, -81.0, 123.0, -46.0]
    targets = [
        np.asarray([3.0, 3.0]),
        np.asarray([14.0, 3.5]),
        np.asarray([4.0, 11.0]),
        np.asarray([15.0, 10.5]),
    ]
    result = []
    for index, piece in enumerate(pieces):
        center = np.mean(piece, axis=0)
        angle = math.radians(angles[index])
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        result.append(
            (piece - center) @ rotation.T + targets[index]
        )
    return result


def _panel_mapper(polygons, origin, size):
    all_points = np.vstack(polygons)
    low = np.min(all_points, axis=0)
    high = np.max(all_points, axis=0)
    span = np.maximum(high - low, 1e-6)
    scale = min(
        (size[0] - 80.0) / span[0],
        (size[1] - 110.0) / span[1],
    )
    offset = np.asarray(origin, dtype=np.float64) + np.asarray(
        [
            40.0 + 0.5 * (size[0] - 80.0 - span[0] * scale),
            70.0 + 0.5 * (size[1] - 110.0 - span[1] * scale),
        ]
    )

    def convert(points):
        return np.round(
            (np.asarray(points) - low) * scale + offset
        ).astype(np.int32)

    return convert


def save_mandatory_t_visual(scattered, solution):
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    output = VISUAL_DIR / "mandatory_t_junction_visual.png"
    canvas = np.full((680, 1500, 3), 248, dtype=np.uint8)
    colors = [
        (255, 220, 120),
        (140, 225, 150),
        (240, 175, 235),
    ]

    solved = [
        np.asarray(placement.vertices, dtype=np.float64)
        for placement in solution.placements
    ]
    panels = [
        (
            scattered,
            (20, 35),
            (700, 600),
            "Random-angle input",
        ),
        (
            solved,
            (780, 35),
            (700, 600),
            "Solved geometry: mandatory T junction",
        ),
    ]
    mappers = []
    for polygons, origin, size, title in panels:
        cv2.rectangle(
            canvas,
            origin,
            (origin[0] + size[0], origin[1] + size[1]),
            (80, 80, 80),
            2,
        )
        cv2.putText(
            canvas,
            title,
            (origin[0] + 22, origin[1] + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        mapper = _panel_mapper(polygons, origin, size)
        mappers.append(mapper)
        for index, polygon in enumerate(polygons):
            points = mapper(polygon)
            cv2.fillPoly(canvas, [points], colors[index])
            cv2.polylines(
                canvas,
                [points],
                True,
                (45, 45, 45),
                3,
                cv2.LINE_AA,
            )
            center = np.mean(points, axis=0).astype(int)
            cv2.putText(
                canvas,
                "P{}".format(index + 1),
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

    mapper = mappers[1]

    # The solver may rotate the finished rectangle by any angle and may use
    # either of a rectangle's parallel edges. Locate the actual 10=(4+6)
    # contact in the returned placement instead of assuming fixed edge IDs.
    best_contact = None
    for long_id in range(len(solved[2])):
        long_points = np.asarray(
            [
                solved[2][long_id],
                solved[2][(long_id + 1) % len(solved[2])],
            ]
        )
        direction = long_points[1] - long_points[0]
        long_length = float(np.linalg.norm(direction))
        if long_length < 9.5:
            continue
        unit = direction / long_length
        normal = np.asarray([-unit[1], unit[0]])
        for edge_a_id in range(len(solved[0])):
            edge_a = np.asarray(
                [
                    solved[0][edge_a_id],
                    solved[0][
                        (edge_a_id + 1) % len(solved[0])
                    ],
                ]
            )
            if abs(np.linalg.norm(edge_a[1] - edge_a[0]) - 4.0) > 0.2:
                continue
            for edge_b_id in range(len(solved[1])):
                edge_b = np.asarray(
                    [
                        solved[1][edge_b_id],
                        solved[1][
                            (edge_b_id + 1) % len(solved[1])
                        ],
                    ]
                )
                if (
                    abs(np.linalg.norm(edge_b[1] - edge_b[0]) - 6.0)
                    > 0.2
                ):
                    continue
                all_short = np.vstack([edge_a, edge_b])
                relative = all_short - long_points[0]
                line_error = float(
                    np.sum(np.abs(relative @ normal))
                )
                projections = relative @ unit
                coverage_error = abs(float(np.min(projections)))
                coverage_error += abs(
                    float(np.max(projections)) - long_length
                )
                join_distances = np.linalg.norm(
                    edge_a[:, None, :] - edge_b[None, :, :],
                    axis=2,
                )
                join_index = np.unravel_index(
                    int(np.argmin(join_distances)),
                    join_distances.shape,
                )
                join_error = float(join_distances[join_index])
                score = line_error + coverage_error + join_error
                candidate = (
                    score,
                    long_points,
                    edge_a,
                    edge_b,
                    join_index,
                )
                if best_contact is None or score < best_contact[0]:
                    best_contact = candidate
    if best_contact is None:
        raise AssertionError("Could not locate mandatory T contact")
    _, long_world, short_a_world, short_b_world, join_index = (
        best_contact
    )
    long_edge = mapper(long_world)
    short_a = mapper(short_a_world)
    short_b = mapper(short_b_world)
    cv2.line(
        canvas,
        tuple(long_edge[0]),
        tuple(long_edge[1]),
        (20, 20, 235),
        10,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        tuple(short_a[0]),
        tuple(short_a[1]),
        (235, 120, 20),
        5,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        tuple(short_b[0]),
        tuple(short_b[1]),
        (20, 175, 30),
        5,
        cv2.LINE_AA,
    )
    t_point = np.mean(
        np.vstack(
            [
                short_a[join_index[0]],
                short_b[join_index[1]],
            ]
        ),
        axis=0,
    ).astype(int)
    cv2.circle(
        canvas,
        tuple(t_point),
        13,
        (0, 220, 255),
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "LONG P3 = 10.0 cm",
        (815, 570),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (20, 20, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "SHORT P1 = 4.0 cm  +  SHORT P2 = 6.0 cm",
        (815, 602),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.61,
        (35, 100, 35),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "4.0 + 6.0 = 10.0 cm; yellow dot = virtual T point",
        (815, 632),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output), canvas):
        raise OSError("Failed to save {}".format(output))
    return output


def validate(name, pieces, expected_short=7.0, expected_long=10.0):
    scattered_pieces = scatter(pieces)
    solution, layouts = puzzle_solver_v3.solve_geometry(
        scattered_pieces,
        max_seconds=20.0,
    )
    if solution is None:
        raise AssertionError(
            "{} failed: {}".format(
                name,
                puzzle_solver_v3.last_diagnostics(),
            )
        )
    short_side, long_side = sorted(
        (solution.width_cm, solution.height_cm)
    )
    if (
        abs(short_side - expected_short) > 0.08
        or abs(long_side - expected_long) > 0.08
        or solution.rectangularity < 0.98
    ):
        raise AssertionError(
            "{} wrong result {:.3f}x{:.3f}, IoU={:.4f}".format(
                name,
                solution.width_cm,
                solution.height_cm,
                solution.rectangularity,
            )
        )
    print(
        "PASS {} {:.3f}x{:.3f} IoU={:.4f} layouts={}".format(
            name,
            solution.width_cm,
            solution.height_cm,
            solution.rectangularity,
            layouts,
        )
    )
    if name == "mandatory-T":
        output = save_mandatory_t_visual(
            scattered_pieces,
            solution,
        )
        print("VISUAL {}".format(output))


def validate_reject(name, pieces):
    solution, layouts = puzzle_solver_v3.solve_geometry(
        scatter(pieces),
        max_seconds=4.0,
    )
    if solution is not None:
        raise AssertionError(
            "{} should be rejected, got {:.3f}x{:.3f}".format(
                name,
                solution.width_cm,
                solution.height_cm,
            )
        )
    print(
        "PASS {} rejected layouts={} diagnostics={}".format(
            name,
            layouts,
            puzzle_solver_v3.last_diagnostics(),
        )
    )


def validate_three_family_reserve():
    pieces = [
        np.asarray(points, dtype=np.float64)
        for points in [
            [
                [2.676, 4.325],
                [3.647, 6.173],
                [5.546, 7.473],
                [7.124, 5.375],
            ],
            [
                [8.350, 3.450],
                [8.124, 6.450],
                [14.301, 6.950],
            ],
            [
                [13.801, 1.853],
                [14.650, 5.925],
                [18.851, 7.200],
                [19.450, 4.972],
                [15.995, 1.697],
            ],
            [
                [9.175, 7.750],
                [7.025, 8.925],
                [7.425, 11.850],
                [11.675, 11.300],
            ],
        ]
    ]
    centered = [piece - np.mean(piece, axis=0) for piece in pieces]
    families, _ = puzzle_solver_v3.generate_seam_families(centered)
    topologies = puzzle_solver_v3.enumerate_topologies(
        centered,
        families,
    )

    edge = puzzle_solver_v3.EdgeRef
    target_signature = tuple(
        sorted(
            [
                (edge(1, 2), (edge(0, 0), edge(2, 3))),
                (edge(2, 0), (edge(3, 3),)),
                (edge(0, 3), (edge(2, 4), edge(3, 0))),
            ]
        )
    )
    signatures = [
        puzzle_solver_v3.topology_signature(topology)
        for topology in topologies
    ]
    if target_signature not in signatures:
        raise AssertionError(
            "three-family reserve dropped the capture regression topology"
        )
    rank = signatures.index(target_signature)
    if rank < puzzle_solver_v3.MAX_ENUMERATED_TOPOLOGIES:
        raise AssertionError(
            "capture regression no longer exercises the reserve"
        )
    print(
        "PASS three-family-reserve rank={} topologies={}".format(
            rank,
            len(topologies),
        )
    )


def validate_three_family_reserve_jitter():
    pieces = [
        np.asarray(points, dtype=np.float64)
        for points in [
            [
                [3.900, 4.325],
                [7.701, 6.926],
                [7.449, 4.798],
                [6.150, 2.901],
            ],
            [
                [10.348, 2.474],
                [11.352, 5.450],
                [17.224, 3.700],
            ],
            [
                [16.950, 5.575],
                [14.675, 6.300],
                [14.050, 11.150],
                [15.600, 12.951],
                [18.550, 9.749],
            ],
            [
                [9.600, 9.725],
                [5.250, 10.650],
                [5.499, 13.125],
                [8.426, 13.900],
            ],
        ]
    ]
    centered = [piece - np.mean(piece, axis=0) for piece in pieces]
    families, _ = puzzle_solver_v3.generate_seam_families(centered)
    topologies = puzzle_solver_v3.enumerate_topologies(
        centered,
        families,
    )

    edge = puzzle_solver_v3.EdgeRef
    target_signature = tuple(
        sorted(
            [
                (edge(0, 0), (edge(2, 2), edge(3, 1))),
                (edge(1, 2), (edge(0, 1), edge(2, 1))),
                (edge(2, 3), (edge(3, 0),)),
            ]
        )
    )
    signatures = [
        puzzle_solver_v3.topology_signature(topology)
        for topology in topologies
    ]
    if target_signature not in signatures:
        raise AssertionError(
            "jittered capture fell outside the three-family reserve"
        )
    rank = signatures.index(target_signature)
    if rank < puzzle_solver_v3.MAX_ENUMERATED_TOPOLOGIES:
        raise AssertionError(
            "jittered capture no longer exercises the reserve tail"
        )
    print(
        "PASS three-family-reserve-jitter rank={} topologies={}".format(
            rank,
            len(topologies),
        )
    )


def validate_two_by_two_chain_model():
    """Regression for the measured 024248 two-chain-on-two-chain case."""
    pieces = [
        np.asarray(points, dtype=np.float64)
        for points in [
            [[7.951, 2.550], [4.973, 6.951], [10.074, 6.525]],
            [[8.372, 1.860], [11.476, 6.726], [16.527, 6.299]],
            [[4.852, 7.701], [12.101, 12.375], [9.774, 7.348]],
            [[16.577, 7.222], [11.576, 7.278], [13.650, 11.700]],
        ]
    ]
    centers = [
        puzzle_solver_v3.legacy.polygon_centroid(piece)
        for piece in pieces
    ]
    centered = [
        piece - center for piece, center in zip(pieces, centers)
    ]
    families, _ = puzzle_solver_v3.generate_seam_families(centered)
    edge = puzzle_solver_v3.EdgeRef

    def chain_signature(first, second):
        first = tuple(sorted(first))
        second = tuple(sorted(second))
        if second < first:
            first, second = second, first
        edges = tuple(sorted(first + second))
        return (edges[0], edges[1:], first, second)

    target_signature = tuple(
        sorted(
            [
                chain_signature(
                    (edge(0, 1), edge(1, 1)),
                    (edge(2, 2), edge(3, 0)),
                ),
                chain_signature(
                    (edge(0, 2), edge(2, 1)),
                    (edge(1, 0), edge(3, 1)),
                ),
            ]
        )
    )
    topologies = puzzle_solver_v3.enumerate_topologies(
        centered,
        families,
        source_pieces=pieces,
    )
    signatures = [
        puzzle_solver_v3.topology_signature(topology)
        for topology in topologies
    ]
    if target_signature not in signatures:
        raise AssertionError("2x2 target topology was not reserved")
    topology = topologies[signatures.index(target_signature)]
    best_iou = 0.0
    best_seam_rms = float("inf")
    anchor_id = 2
    for orders in puzzle_solver_v3.topology_orders(topology):
        pairs = puzzle_solver_v3.all_point_pairs(
            centered,
            topology,
            orders,
        )
        initial = puzzle_solver_v3.initial_poses(
            centered,
            pairs,
            anchor_id,
        )
        if initial is None:
            continue
        poses, seam_rms = puzzle_solver_v3.optimize_poses(
            initial,
            pairs,
            anchor_id,
        )
        placements = puzzle_solver_v3.build_placements(
            pieces,
            centered,
            centers,
            poses,
        )
        iou, _, _, _ = puzzle_solver_v3.layout_metrics(placements)
        if seam_rms <= 0.30 and iou > best_iou:
            best_iou = iou
            best_seam_rms = seam_rms
    if best_iou < 0.90 or best_seam_rms > 0.30:
        raise AssertionError(
            "2x2 cumulative-chain constraints did not reconstruct "
            "the measured layout"
        )
    print(
        "PASS two-by-two-chain rank={} IoU={:.4f} seam={:.4f}".format(
            signatures.index(target_signature),
            best_iou,
            best_seam_rms,
        )
    )


def main():
    validate_two_by_two_chain_model()
    validate_three_family_reserve()
    validate_three_family_reserve_jitter()
    validate(
        "three-part-long-edge",
        [
            rectangle(0, 0, 2, 3),
            rectangle(2, 0, 5, 3),
            rectangle(5, 0, 10, 3),
            rectangle(0, 3, 10, 6),
        ],
        expected_short=6.0,
    )
    validate(
        "mandatory-T",
        [
            rectangle(0, 0, 4, 3),
            rectangle(4, 0, 10, 3),
            rectangle(0, 3, 10, 7),
        ],
    )
    validate(
        "four-piece-grid",
        [
            rectangle(0, 0, 4, 3),
            rectangle(4, 0, 10, 3),
            rectangle(0, 3, 6, 7),
            rectangle(6, 3, 10, 7),
        ],
    )
    validate_reject(
        "impossible-small-pieces",
        [
            rectangle(0, 0, 1, 1),
            rectangle(1, 0, 2, 1),
            rectangle(0, 1, 1, 2),
            rectangle(1, 1, 2, 2),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
