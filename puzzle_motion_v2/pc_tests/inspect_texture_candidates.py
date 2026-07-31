"""Dump every geometrically legal textured-card candidate as a contact sheet."""

from __future__ import annotations

import argparse
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
import motion_protocol


PX_PER_CM = 40.0


def clone_placements(placements):
    return [
        SimpleNamespace(
            piece_id=item.piece_id,
            vertices=np.asarray(item.vertices).copy(),
            rotation=np.asarray(item.rotation).copy(),
            translation=np.asarray(item.translation).copy(),
            used_edges=item.used_edges,
            edge_coverage=item.edge_coverage,
        )
        for item in placements
    ]


def render_candidate(image, pieces, placements, label):
    points = np.concatenate(
        [item.vertices for item in placements],
        axis=0,
    )
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    margin = 0.35
    size = maximum - minimum + 2.0 * margin
    output_scale = 45.0
    width = int(round(size[0] * output_scale))
    height = int(round(size[1] * output_scale))
    canvas = np.full((height, width, 3), 230, dtype=np.uint8)
    covered = np.zeros((height, width), dtype=np.uint8)

    for placement in placements:
        piece_id = placement.piece_id
        source_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(
            source_mask,
            [
                np.round(
                    pieces[piece_id] * PX_PER_CM
                ).astype(np.int32)
            ],
            255,
        )
        transform = np.zeros((2, 3), dtype=np.float64)
        transform[:, :2] = (
            placement.rotation * (output_scale / PX_PER_CM)
        )
        transform[:, 2] = (
            (
                placement.translation
                - minimum
                + margin
            )
            * output_scale
        )
        warped = cv2.warpAffine(
            image,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(230, 230, 230),
        )
        warped_mask = cv2.warpAffine(
            source_mask,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        selected = (warped_mask > 0) & (covered == 0)
        canvas[selected] = warped[selected]
        covered[selected] = 255
        contour = np.round(
            (
                placement.vertices
                - minimum
                + margin
            )
            * output_scale
        ).astype(np.int32)
        cv2.polylines(canvas, [contour], True, (0, 220, 0), 2)
        center = np.round(
            (
                np.mean(placement.vertices, axis=0)
                - minimum
                + margin
            )
            * output_scale
        ).astype(int)
        cv2.putText(
            canvas,
            "P{}".format(piece_id + 1),
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    header = np.full((42, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        label,
        (5, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, canvas))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("rectified_jpg", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-enumerated", type=int)
    parser.add_argument("--max-coarse", type=int)
    parser.add_argument("--max-optimize", type=int)
    parser.add_argument("--max-seconds", type=float, default=14.0)
    args = parser.parse_args()

    if args.max_enumerated is not None:
        solver.MAX_ENUMERATED_TOPOLOGIES = args.max_enumerated
    if args.max_coarse is not None:
        solver.MAX_COARSE_TOPOLOGIES = args.max_coarse
        solver.FAST_STRICT_TOPOLOGIES = args.max_coarse
    if args.max_optimize is not None:
        solver.MAX_TOPOLOGIES_TO_OPTIMIZE = args.max_optimize

    payload = json.loads(args.result_json.read_text(encoding="utf-8"))
    pieces = [
        np.asarray(item["vertices_cm"], dtype=np.float64)
        for item in payload["pieces"]
    ]
    image = cv2.imread(str(args.rectified_jpg))
    context = texture_matcher.build_context(image, PX_PER_CM)

    original_score_layout = texture_matcher.score_layout
    records = []

    def capture_score(context_arg, originals, placements, topology, orders):
        result = original_score_layout(
            context_arg,
            originals,
            placements,
            topology,
            orders,
        )
        iou, overlap, width, height = solver.layout_metrics(placements)
        records.append(
            {
                "placements": clone_placements(placements),
                "topology": topology,
                "orders": orders,
                "topology_signature": (
                    None
                    if topology is None
                    else repr(solver.topology_signature(topology))
                ),
                "topology_heuristic": (
                    None
                    if topology is None
                    else solver.topology_heuristic(
                        len(pieces),
                        pieces,
                        topology,
                    )
                ),
                "iou": iou,
                "overlap": overlap,
                "width": width,
                "height": height,
                **result,
            }
        )
        return result

    texture_matcher.score_layout = capture_score
    try:
        solution, nodes = solver.solve_geometry(
            pieces,
            max_seconds=args.max_seconds,
            texture_context=context,
        )
    finally:
        texture_matcher.score_layout = original_score_layout

    for record in records:
        if record["topology"] is None:
            record["source_score"] = float(record["score"])
            record["source_seam"] = float(
                record.get("seam_score", 0.0)
            )
            record["source_perimeter"] = float(
                record.get("perimeter_score", 0.0)
            )
            record["current_texture"] = float(record["score"])
        else:
            source = texture_matcher.score_source_topology(
                context,
                pieces,
                record["topology"],
                record["orders"],
                {},
            )
            record["source_score"] = float(source["score"])
            record["source_seam"] = float(
                source.get("seam_score", 0.0)
            )
            record["source_perimeter"] = float(
                source.get("perimeter_score", 0.0)
            )
            record["current_texture"] = (
                solver.SOURCE_TEXTURE_BLEND
                * record["source_score"]
                + (1.0 - solver.SOURCE_TEXTURE_BLEND)
                * record["score"]
            )
        record["current_total"] = (
            1.0 - record["iou"]
            + 1.7 * record["overlap"]
            + solver.TEXTURE_SCORE_WEIGHT * record["current_texture"]
        )

    records.sort(key=lambda item: item["current_total"])
    tiles = []
    metadata = []
    for index, record in enumerate(records):
        label = (
            "#{:02d} total={:.3f} iou={:.3f} seam={:.3f} "
            "sym={:.3f} src={:.3f}"
        ).format(
            index,
            record["current_total"],
            record["iou"],
            record.get("seam_score", 0.0),
            record.get("symmetry_score", 0.0),
            record["source_score"],
        )
        tile = render_candidate(
            image,
            pieces,
            record["placements"],
            label,
        )
        individual_path = args.output.with_name(
            "{}_candidate_{:02d}.jpg".format(
                args.output.stem,
                index,
            )
        )
        cv2.imwrite(str(individual_path), tile)
        centroids = {
            str(int(item.piece_id) + 1): np.mean(
                np.asarray(item.vertices, dtype=np.float64),
                axis=0,
            ).round(6).tolist()
            for item in record["placements"]
        }
        placement_details = {
            str(int(item.piece_id) + 1): {
                "vertices": np.asarray(
                    item.vertices,
                    dtype=np.float64,
                ).round(6).tolist(),
                "used_edges": sorted(
                    int(edge_id)
                    for edge_id in item.used_edges
                ),
                "edge_coverage": [
                    [
                        [
                            round(float(start), 6),
                            round(float(end), 6),
                        ]
                        for start, end in intervals
                    ]
                    for intervals in item.edge_coverage
                ],
            }
            for item in record["placements"]
        }
        if index < 20:
            tile = cv2.resize(tile, (440, 340))
            tiles.append(tile)
        item_metadata = {
                key: value
                for key, value in record.items()
                if key not in ("placements", "topology", "orders")
        }
        item_metadata["centroids"] = centroids
        item_metadata["placements"] = placement_details
        item_metadata["individual_image"] = str(individual_path)
        metadata.append(item_metadata)

    rows = []
    for start in range(0, len(tiles), 4):
        row = tiles[start:start + 4]
        while len(row) < 4:
            row.append(np.full_like(tiles[0], 255))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "nodes": nodes,
                "solution": solution is not None,
                "diagnostics": solver.last_diagnostics(),
                "candidates": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if solution is not None:
        final_solution = solver.canonical_target_solution(solution)
        best_path = args.output.with_name(
            args.output.stem + "_best.jpg"
        )
        cv2.imwrite(
            str(best_path),
            render_candidate(
                image,
                pieces,
                final_solution.placements,
                "selected {}".format(
                    solver.last_diagnostics().get("version", "")
                ),
            ),
        )
        motion_path = args.output.with_name(
            args.output.stem + "_motion.json"
        )
        plan = motion_protocol.build_motion_plan(
            pieces,
            final_solution,
            output_path=str(motion_path),
        )
        plan["solver_diagnostics"] = solver.last_diagnostics()
        motion_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(args.output)


if __name__ == "__main__":
    main()
