"""Validate texture disambiguation for two geometrically identical pieces."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIX_DIR = PROJECT_DIR / "maix_project"
if str(MAIX_DIR) not in sys.path:
    sys.path.insert(0, str(MAIX_DIR))

import puzzle_solver_v3 as solver
import texture_matcher


PX_PER_CM = 40.0


def locate_real_captures() -> list[tuple[str, Path, Path]]:
    """Locate patterned-card regression captures when available."""
    roots = [Path.home(), Path("/mnt/c/Users/zhao")]
    captures = []
    seen = set()
    for root in roots:
        for capture_date, capture_id in (
            ("20260729", "102952"),
            ("20260729", "112159"),
            ("20260729", "123753"),
            ("20260729", "131419"),
            ("20260729", "135433"),
            ("20260729", "141104"),
            ("20260729", "142730"),
            # Regression for the card ambiguity gate: its first candidate
            # is a verified 0.962-IoU rectangle, while the 0.935-IoU runner
            # up is below the texture-comparison quality floor.
            ("20260731", "012106"),
        ):
            result = (
                root
                / "Downloads"
                / (
                    "pieces_{}_{}_result.json".format(
                        capture_date,
                        capture_id
                    )
                )
            )
            rectified = (
                root
                / "Downloads"
                / (
                    "pieces_{}_{}_rectified.jpg".format(
                        capture_date,
                        capture_id
                    )
                )
            )
            key = (capture_id, str(result))
            if (
                key not in seen
                and result.exists()
                and rectified.exists()
            ):
                seen.add(key)
                captures.append(
                    (capture_id, result, rectified)
                )
    return captures


def placement(piece_id, vertices, rotation, translation):
    vertices = np.asarray(vertices, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    return SimpleNamespace(
        piece_id=piece_id,
        vertices=vertices @ rotation.T + translation,
        rotation=rotation,
        translation=translation,
    )


def make_source_image() -> np.ndarray:
    image = np.full((160, 320, 3), (210, 145, 65), dtype=np.uint8)
    pattern = np.full((80, 160, 3), (238, 241, 244), dtype=np.uint8)
    cv2.line(pattern, (5, 70), (155, 9), (35, 35, 190), 9, cv2.LINE_AA)
    cv2.circle(pattern, (82, 42), 25, (25, 25, 25), 7, cv2.LINE_AA)
    cv2.line(pattern, (24, 12), (137, 68), (40, 165, 235), 6, cv2.LINE_AA)
    cv2.putText(
        pattern,
        "K",
        (68, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (25, 25, 25),
        3,
        cv2.LINE_AA,
    )
    image[40:120, 40:120] = pattern[:, :80]
    image[40:120, 200:280] = pattern[:, 80:]
    return image


def validate_blank_white_geometry_mode() -> int:
    """Pure white Q1/Q2 pieces must not activate card-only constraints."""
    source = np.full((160, 320, 3), (210, 145, 65), dtype=np.uint8)
    pieces = [
        np.asarray(
            [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]],
            dtype=np.float64,
        ),
        np.asarray(
            [[5.0, 1.0], [7.0, 1.0], [7.0, 3.0], [5.0, 3.0]],
            dtype=np.float64,
        ),
    ]
    for piece in pieces:
        cv2.fillPoly(
            source,
            [np.round(piece * PX_PER_CM).astype(np.int32)],
            (242, 242, 242),
        )
    context = texture_matcher.build_context(source, PX_PER_CM)
    texture_matcher.prepare_context_for_pieces(context, pieces)
    if context.white_card_confidence >= solver.WHITE_CARD_MODE_CONFIDENCE:
        print(
            "FAIL blank white geometry activated card mode",
            context.white_card_confidence,
        )
        return 1
    print("PASS blank white geometry stays in geometry mode")
    return 0


def main() -> int:
    if validate_blank_white_geometry_mode() != 0:
        return 1
    source = make_source_image()
    context = texture_matcher.build_context(source, PX_PER_CM)
    pieces = [
        np.asarray(
            [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]],
            dtype=np.float64,
        ),
        np.asarray(
            [[5.0, 1.0], [7.0, 1.0], [7.0, 3.0], [5.0, 3.0]],
            dtype=np.float64,
        ),
    ]
    identity = np.eye(2, dtype=np.float64)
    correct_placements = [
        placement(0, pieces[0], identity, [0.0, 0.0]),
        placement(1, pieces[1], identity, [-2.0, 0.0]),
    ]
    correct_family = solver.SeamFamily(
        long_edge=solver.EdgeRef(0, 1),
        short_edges=(solver.EdgeRef(1, 3),),
        length_error_cm=0.0,
        normalized_error=0.0,
        used_mask=0,
        internal_length_cm=2.0,
    )
    correct = texture_matcher.score_layout(
        context,
        pieces,
        correct_placements,
        (correct_family,),
        ((solver.EdgeRef(1, 3),),),
    )

    half_turn = -np.eye(2, dtype=np.float64)
    wrong_placements = [
        placement(0, pieces[0], identity, [0.0, 0.0]),
        placement(1, pieces[1], half_turn, [10.0, 4.0]),
    ]
    wrong_family = solver.SeamFamily(
        long_edge=solver.EdgeRef(0, 1),
        short_edges=(solver.EdgeRef(1, 1),),
        length_error_cm=0.0,
        normalized_error=0.0,
        used_mask=0,
        internal_length_cm=2.0,
    )
    wrong = texture_matcher.score_layout(
        context,
        pieces,
        wrong_placements,
        (wrong_family,),
        ((solver.EdgeRef(1, 1),),),
    )

    output = (
        PROJECT_DIR
        / "pc_simulator"
        / "v3_validation"
        / "texture_identical_shapes_source.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), source)
    print("correct", correct)
    print("wrong  ", wrong)
    print("visual", output)
    if correct["confidence"] < 0.25:
        print("FAIL texture confidence too low")
        return 1
    if correct["score"] + 0.10 >= wrong["score"]:
        print("FAIL identical-shape layouts were not separated")
        return 1
    print("PASS texture selects the correct identical-shape layout")

    captures = locate_real_captures()
    if not captures:
        print("SKIP real patterned-card flow regression")
        return 0
    for capture_id, result_path, rectified_path in captures:
        payload = json.loads(
            result_path.read_text(encoding="utf-8")
        )
        pieces = [
            np.asarray(piece["vertices_cm"], dtype=np.float64)
            for piece in payload["pieces"]
        ]
        rectified = cv2.imread(str(rectified_path))
        real_context = texture_matcher.build_context(
            rectified,
            PX_PER_CM,
        )
        real_solution, real_nodes = solver.solve_geometry(
            pieces,
            max_seconds=(20.0 if capture_id == "012106" else 14.0),
            texture_context=real_context,
        )
        diagnostics = solver.last_diagnostics()
        print(
            "real patterned-card {}".format(capture_id),
            {
                "solution": real_solution is not None,
                "nodes": real_nodes,
                "diagnostics": diagnostics,
            },
        )
        if real_solution is None:
            print(
                "FAIL real patterned-card {} has no solution".format(
                    capture_id
                )
            )
            return 1
        if diagnostics.get("best_texture_confidence", 0.0) < 0.75:
            print(
                "FAIL real capture {} has weak texture evidence".format(
                    capture_id
                )
            )
            return 1
        if diagnostics.get("fallback") == "legacy_edge_dfs":
            print(
                "FAIL real capture {} used geometry fallback".format(
                    capture_id
                )
            )
            return 1
        short_side, long_side = sorted(
            (real_solution.width_cm, real_solution.height_cm)
        )
        aspect_ratio = long_side / max(short_side, 1e-9)
        if not (
            solver.WHITE_CARD_MIN_ASPECT_RATIO
            <= aspect_ratio
            <= solver.WHITE_CARD_MAX_ASPECT_RATIO
        ):
            print(
                "FAIL real capture {} produced aspect {:.3f}".format(
                    capture_id,
                    aspect_ratio,
                )
            )
            return 1
        if diagnostics.get("size_pass") != "card_aspect":
            print(
                "FAIL real capture {} skipped card aspect mode".format(
                    capture_id
                )
            )
            return 1
        if diagnostics.get(
            "white_card_confidence",
            0.0,
        ) < 0.9:
            print("FAIL white-card mode was not activated")
            return 1
        canonical = solver.canonical_target_solution(real_solution)
        target_centroids = {
            int(item.piece_id): np.mean(
                np.asarray(item.vertices, dtype=np.float64),
                axis=0,
            )
            for item in canonical.placements
        }
        # Only 142730 currently has user-confirmed artwork ground truth:
        # P1/P4 form one card-width column and P2/P3 form the other.  The old
        # table for all five captures was inferred from solver previews and
        # was therefore circular and wrong; do not silently restore it.
        if capture_id == "142730":
            if (
                not diagnostics.get("early_accepted", False)
                or diagnostics.get("early_accept_reason")
                != "card_quality_gate"
            ):
                print(
                    "FAIL real capture 142730 did not use the "
                    "protected card-quality early stop"
                )
                return 1
            first_column = 0.5 * (
                target_centroids[0] + target_centroids[3]
            )
            second_column = 0.5 * (
                target_centroids[1] + target_centroids[2]
            )
            column_gap = second_column[0] - first_column[0]
            if column_gap <= 1.0:
                print(
                    "FAIL real capture 142730 did not separate "
                    "P1/P4 from P2/P3",
                    {
                        piece_id + 1: np.round(center, 4).tolist()
                        for piece_id, center in target_centroids.items()
                    },
                )
                return 1
        if capture_id == "102952":
            layout_variant = str(
                diagnostics.get("best_layout_variant", "")
            )
            if "half_turn_P1_P2" not in layout_variant:
                print(
                    "FAIL real capture 102952 did not turn the "
                    "user-confirmed P1/P2 card column",
                    layout_variant,
                )
                return 1
            left_column = 0.5 * (
                target_centroids[2] + target_centroids[3]
            )
            right_column = 0.5 * (
                target_centroids[0] + target_centroids[1]
            )
            if right_column[0] - left_column[0] <= 1.0:
                print(
                    "FAIL real capture 102952 put the patterned "
                    "P1/P2 edge on the outside instead of the right",
                    {
                        piece_id + 1: np.round(center, 4).tolist()
                        for piece_id, center in target_centroids.items()
                    },
                )
                return 1
        if capture_id == "112159":
            # User-confirmed complete card: P4/P3 and P2/P1 are the two
            # columns.  Their top-to-bottom directions must agree.  The old
            # v4.6 result had P3/P4 reversed, making these vectors point in
            # opposite directions even though the rectangle and white frame
            # were both valid.
            left_direction = (
                target_centroids[3] - target_centroids[2]
            )
            right_direction = (
                target_centroids[1] - target_centroids[0]
            )
            direction_cosine = float(
                np.dot(left_direction, right_direction)
                / max(
                    np.linalg.norm(left_direction)
                    * np.linalg.norm(right_direction),
                    1e-9,
                )
            )
            if direction_cosine <= 0.5:
                print(
                    "FAIL real capture 112159 left card column "
                    "is upside down",
                    {
                        "direction_cosine": direction_cosine,
                        "centroids": {
                            piece_id + 1: np.round(
                                center,
                                4,
                            ).tolist()
                            for piece_id, center
                            in target_centroids.items()
                        },
                    },
                )
                return 1
        if capture_id == "012106":
            if diagnostics.get("early_accept_reason") != (
                "two_by_two_grid_quality_gate"
            ):
                print(
                    "FAIL real capture 012106 did not preserve its "
                    "2x2 quality decision"
                )
                return 1
            if diagnostics.get("fallback_reason") == (
                "ambiguous_global_candidate"
            ):
                print(
                    "FAIL real capture 012106 let a below-floor "
                    "runner-up trigger ambiguity"
                )
                return 1
            if diagnostics.get("best_iou", 0.0) < solver.TEXTURE_ACCEPT_IOU:
                print(
                    "FAIL real capture 012106 lost the verified "
                    "card candidate"
                )
                return 1
        if diagnostics.get("best_layout_perimeter_confidence", 0.0) < 0.60:
            print(
                "FAIL real capture {} has no reliable outer border".format(
                    capture_id
                )
            )
            return 1
    print("PASS real patterned-card flow avoids legacy fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
