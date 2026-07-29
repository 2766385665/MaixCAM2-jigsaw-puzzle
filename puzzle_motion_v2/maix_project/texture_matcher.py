"""Low-cost colour-pattern continuity scoring for puzzle seams.

Geometry remains the primary solver.  This module only re-ranks geometrically
legal layouts by sampling narrow strips just inside both sides of every
candidate seam.  It is intentionally independent from the solver classes so
it can be disabled by passing no context and can run on MaixCAM 2 with only
OpenCV and NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


SAMPLE_DEPTHS_CM = (0.10, 0.20, 0.30)
MIN_SAMPLES_PER_SEGMENT = 10
MAX_SAMPLES_PER_SEGMENT = 28
SAMPLES_PER_CM = 5.0
ENDPOINT_MARGIN = 0.08
# A low cut-line colour error alone is unsafe for playing cards: two unrelated
# blank white regions can look like an excellent seam.  A valid completed
# face card also has an uncut white outer border and a strong 180-degree
# pattern correspondence.  The five manually verified J/Q captures require
# these two independent card-level checks to outrank accidental white/white
# contacts.
CARD_SOURCE_PERIMETER_BLEND = 0.55
CARD_LAYOUT_SYMMETRY_BLEND = 0.70
CARD_LAYOUT_PERIMETER_BLEND = 0.68
CARD_SYMMETRY_SAMPLES_PER_CM = 9.0
MAX_LAYOUT_CONTACT_DISTANCE_CM = 0.28
MIN_LAYOUT_CONTACT_LENGTH_CM = 0.45
MIN_LAYOUT_CONTACT_PARALLEL_COSINE = 0.965
MAX_LAYOUT_PERIMETER_DISTANCE_CM = 0.24

# Estimate the colour exactly on the cut line from several samples inside
# each piece.  Comparing equal positive depths directly compares points that
# are 2*d apart in the original card and therefore penalizes a perfectly
# continuous high-frequency print.  The intercept of a least-squares line in
# depth is a much better approximation of the common cut-line colour.
_DEPTH_MATRIX = np.column_stack(
    (
        np.ones(len(SAMPLE_DEPTHS_CM), dtype=np.float64),
        np.asarray(SAMPLE_DEPTHS_CM, dtype=np.float64),
    )
)
BOUNDARY_EXTRAPOLATION_WEIGHTS = np.linalg.pinv(
    _DEPTH_MATRIX
)[0]


@dataclass
class TextureContext:
    lab: np.ndarray
    gray: np.ndarray
    px_per_cm: float
    white_card_confidence: float = 0.0
    piece_profile_ready: bool = False
    edge_border_costs: dict | None = None


def build_context(
    rectified_bgr: np.ndarray,
    px_per_cm: float,
) -> TextureContext:
    """Precompute colour spaces once for all candidate layouts."""
    blurred = cv2.GaussianBlur(rectified_bgr, (3, 3), 0)
    return TextureContext(
        lab=cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB),
        gray=cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY),
        px_per_cm=float(px_per_cm),
    )


def _edge_border_ink_score(
    context: TextureContext,
    polygon: np.ndarray,
    edge_id: int,
) -> tuple[float, float]:
    """Measure printed ink immediately inside one possible outer edge."""
    first = polygon[edge_id]
    second = polygon[(edge_id + 1) % len(polygon)]
    vector = second - first
    length = float(np.linalg.norm(vector))
    normal = _inward_normal(polygon, first, second)
    if normal is None or length < 1e-8:
        return 1.0, length
    sample_count = int(
        np.clip(
            round(length * SAMPLES_PER_CM),
            MIN_SAMPLES_PER_SEGMENT,
            MAX_SAMPLES_PER_SEGMENT,
        )
    )
    fractions = np.linspace(0.08, 0.92, sample_count)
    base = first[None, :] + fractions[:, None] * vector[None, :]
    samples = []
    valid = np.ones(sample_count, dtype=bool)
    for depth in (0.12, 0.25, 0.40):
        lab, layer_valid = _bilinear_samples(
            context.lab,
            (base + normal[None, :] * depth)
            * context.px_per_cm,
        )
        samples.append(lab)
        valid &= layer_valid
    if int(np.count_nonzero(valid)) < max(5, sample_count // 2):
        return 1.0, length
    lab = np.concatenate(
        [layer[valid] for layer in samples],
        axis=0,
    )
    ink = np.maximum.reduce(
        (
            np.clip((210.0 - lab[:, 0]) / 90.0, 0.0, 1.0),
            np.clip(
                np.abs(lab[:, 1] - 128.0) / 45.0,
                0.0,
                1.0,
            ),
            np.clip(
                np.abs(lab[:, 2] - 128.0) / 45.0,
                0.0,
                1.0,
            ),
        )
    )
    return float(np.mean(ink)), length


def prepare_context_for_pieces(
    context: TextureContext | None,
    original_pieces: list[np.ndarray],
) -> None:
    """Detect white playing-card material and cache its outer-edge costs."""
    if context is None or context.piece_profile_ready:
        return
    context.piece_profile_ready = True
    mask = np.zeros(context.lab.shape[:2], dtype=np.uint8)
    pieces = [
        np.asarray(piece, dtype=np.float64)
        for piece in original_pieces
    ]
    for polygon in pieces:
        cv2.fillPoly(
            mask,
            [
                np.round(
                    polygon * context.px_per_cm
                ).astype(np.int32)
            ],
            255,
        )
    pixels = context.lab[mask > 0]
    if len(pixels) == 0:
        context.edge_border_costs = {}
        return
    pixels_i16 = pixels.astype(np.int16)
    white = (
        (pixels_i16[:, 0] > 185)
        & (np.abs(pixels_i16[:, 1] - 128) < 18)
        & (np.abs(pixels_i16[:, 2] - 128) < 24)
    )
    white_fraction = float(np.mean(white))
    # Existing brown-card captures are below 1%, while the J/Q/K captures
    # are about 60%.  The wide ramp avoids a brittle binary mode switch.
    context.white_card_confidence = float(
        np.clip(
            (white_fraction - 0.20) / 0.30,
            0.0,
            1.0,
        )
    )
    context.edge_border_costs = {}
    if context.white_card_confidence <= 0.0:
        return
    for piece_id, polygon in enumerate(pieces):
        for edge_id in range(len(polygon)):
            context.edge_border_costs[(piece_id, edge_id)] = (
                _edge_border_ink_score(
                    context,
                    polygon,
                    edge_id,
                )
            )


def _bilinear_samples(
    image: np.ndarray,
    points_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an image at floating point pixel coordinates."""
    points = np.asarray(points_px, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    height, width = image.shape[:2]
    valid = (
        (x >= 0.0)
        & (y >= 0.0)
        & (x < width - 1)
        & (y < height - 1)
    )
    x = np.clip(x, 0.0, max(0.0, width - 1.001))
    y = np.clip(y, 0.0, max(0.0, height - 1.001))
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0

    first = image[y0, x0].astype(np.float64)
    second = image[y0, x1].astype(np.float64)
    third = image[y1, x0].astype(np.float64)
    fourth = image[y1, x1].astype(np.float64)
    if image.ndim == 3:
        wx = wx[:, None]
        wy = wy[:, None]
    values = (
        first * (1.0 - wx) * (1.0 - wy)
        + second * wx * (1.0 - wy)
        + third * (1.0 - wx) * wy
        + fourth * wx * wy
    )
    return values, valid


def _source_points(
    placement,
    target_points_cm: np.ndarray,
) -> np.ndarray:
    """Apply the inverse rigid placement transform."""
    target = np.asarray(target_points_cm, dtype=np.float64)
    return (
        target - np.asarray(placement.translation, dtype=np.float64)
    ) @ np.asarray(placement.rotation, dtype=np.float64)


def _inward_normal(
    polygon: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray | None:
    direction = np.asarray(second, dtype=np.float64) - np.asarray(
        first,
        dtype=np.float64,
    )
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return None
    direction /= length
    normal = np.asarray([-direction[1], direction[0]])
    midpoint = (
        np.asarray(first, dtype=np.float64)
        + np.asarray(second, dtype=np.float64)
    ) * 0.5
    contour = np.asarray(polygon, dtype=np.float32)
    positive_inside = cv2.pointPolygonTest(
        contour,
        tuple(midpoint + normal * 0.08),
        False,
    )
    negative_inside = cv2.pointPolygonTest(
        contour,
        tuple(midpoint - normal * 0.08),
        False,
    )
    if positive_inside >= negative_inside:
        return normal
    return -normal


def _score_sample_layers(
    long_lab_layers: list[np.ndarray],
    short_lab_layers: list[np.ndarray],
    long_gray_layers: list[np.ndarray],
    short_gray_layers: list[np.ndarray],
    all_valid: np.ndarray,
    length: float,
) -> tuple[float, float, float]:
    """Score two already-corresponding inward edge-strip samples."""
    sample_count = len(all_valid)
    if int(np.count_nonzero(all_valid)) < max(5, sample_count // 2):
        return 0.0, 0.0, length
    long_lab = np.stack(long_lab_layers, axis=0)[:, all_valid]
    short_lab = np.stack(short_lab_layers, axis=0)[:, all_valid]
    long_gray = np.stack(long_gray_layers, axis=0)[:, all_valid]
    short_gray = np.stack(short_gray_layers, axis=0)[:, all_valid]

    long_boundary_lab = np.tensordot(
        BOUNDARY_EXTRAPOLATION_WEIGHTS,
        long_lab,
        axes=(0, 0),
    )
    short_boundary_lab = np.tensordot(
        BOUNDARY_EXTRAPOLATION_WEIGHTS,
        short_lab,
        axes=(0, 0),
    )
    long_boundary_gray = np.tensordot(
        BOUNDARY_EXTRAPOLATION_WEIGHTS,
        long_gray,
        axes=(0, 0),
    )
    short_boundary_gray = np.tensordot(
        BOUNDARY_EXTRAPOLATION_WEIGHTS,
        short_gray,
        axes=(0, 0),
    )

    lab_scale = np.asarray([70.0, 45.0, 45.0])
    boundary_colour_delta = np.mean(
        np.minimum(
            1.0,
            np.abs(long_boundary_lab - short_boundary_lab)
            / lab_scale,
        ),
        axis=1,
    )
    near_colour_delta = np.mean(
        np.minimum(
            1.0,
            np.abs(long_lab[0] - short_lab[0]) / lab_scale,
        ),
        axis=1,
    )
    colour_error = float(
        np.mean(
            0.82 * boundary_colour_delta
            + 0.18 * near_colour_delta
        )
    )
    if len(long_boundary_gray) >= 2:
        tangent_error = float(
            np.mean(
                np.minimum(
                    1.0,
                    np.abs(
                        np.diff(long_boundary_gray)
                        - np.diff(short_boundary_gray)
                    )
                    / 55.0,
                )
            )
        )
    else:
        tangent_error = 0.0
    error = 0.78 * colour_error + 0.22 * tangent_error

    combined = np.concatenate(
        (long_boundary_gray, short_boundary_gray)
    )
    tangent_activity = (
        np.mean(np.abs(np.diff(long_boundary_gray)))
        + np.mean(np.abs(np.diff(short_boundary_gray)))
    )
    activity = float(
        np.clip(
            (
                np.std(combined)
                + 1.8 * tangent_activity
            ) / 70.0,
            0.0,
            1.0,
        )
    )
    return error, activity, length


def _segment_score(
    context: TextureContext,
    long_placement,
    short_placement,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float, float]:
    """Return error, information activity and segment length."""
    vector = np.asarray(second, dtype=np.float64) - np.asarray(
        first,
        dtype=np.float64,
    )
    length = float(np.linalg.norm(vector))
    if length < 0.35:
        return 0.0, 0.0, length
    long_normal = _inward_normal(
        long_placement.vertices,
        first,
        second,
    )
    short_normal = _inward_normal(
        short_placement.vertices,
        first,
        second,
    )
    if long_normal is None or short_normal is None:
        return 0.0, 0.0, length
    # A valid contact must place the two interiors on opposite sides.
    if float(np.dot(long_normal, short_normal)) > -0.20:
        return 1.0, 1.0, length

    sample_count = int(
        np.clip(
            round(length * SAMPLES_PER_CM),
            MIN_SAMPLES_PER_SEGMENT,
            MAX_SAMPLES_PER_SEGMENT,
        )
    )
    fractions = np.linspace(
        ENDPOINT_MARGIN,
        1.0 - ENDPOINT_MARGIN,
        sample_count,
    )
    base = (
        np.asarray(first, dtype=np.float64)[None, :]
        + fractions[:, None] * vector[None, :]
    )

    long_lab_layers = []
    short_lab_layers = []
    long_gray_layers = []
    short_gray_layers = []
    all_valid = np.ones(sample_count, dtype=bool)
    for depth in SAMPLE_DEPTHS_CM:
        long_target = base + long_normal[None, :] * depth
        short_target = base + short_normal[None, :] * depth
        long_source = _source_points(
            long_placement,
            long_target,
        ) * context.px_per_cm
        short_source = _source_points(
            short_placement,
            short_target,
        ) * context.px_per_cm
        long_lab, valid_a = _bilinear_samples(
            context.lab,
            long_source,
        )
        short_lab, valid_b = _bilinear_samples(
            context.lab,
            short_source,
        )
        long_gray, valid_c = _bilinear_samples(
            context.gray,
            long_source,
        )
        short_gray, valid_d = _bilinear_samples(
            context.gray,
            short_source,
        )
        all_valid &= valid_a & valid_b & valid_c & valid_d
        long_lab_layers.append(long_lab)
        short_lab_layers.append(short_lab)
        long_gray_layers.append(long_gray)
        short_gray_layers.append(short_gray)

    return _score_sample_layers(
        long_lab_layers,
        short_lab_layers,
        long_gray_layers,
        short_gray_layers,
        all_valid,
        length,
    )


def _layout_contact_segments(
    placements: list,
) -> list[tuple[object, object, np.ndarray, np.ndarray]]:
    """Find every collinear edge contact in the completed layout.

    Solver topology edges are only the constraints needed to propagate poses;
    they are not guaranteed to contain every physical adjacency in the final
    rectangle.  Pattern validation must therefore be derived from the final
    polygons themselves.  Pairwise overlap also handles a T-junction because
    each short edge produces its own overlap on the same long edge.
    """
    contacts = []
    for first_index, first_placement in enumerate(placements):
        first_polygon = np.asarray(
            first_placement.vertices,
            dtype=np.float64,
        )
        for second_placement in placements[first_index + 1:]:
            second_polygon = np.asarray(
                second_placement.vertices,
                dtype=np.float64,
            )
            for first_edge in range(len(first_polygon)):
                first_start = first_polygon[first_edge]
                first_end = first_polygon[
                    (first_edge + 1) % len(first_polygon)
                ]
                first_vector = first_end - first_start
                first_length = float(np.linalg.norm(first_vector))
                if first_length < MIN_LAYOUT_CONTACT_LENGTH_CM:
                    continue
                tangent = first_vector / first_length
                normal = np.asarray([-tangent[1], tangent[0]])
                for second_edge in range(len(second_polygon)):
                    second_start = second_polygon[second_edge]
                    second_end = second_polygon[
                        (second_edge + 1) % len(second_polygon)
                    ]
                    second_vector = second_end - second_start
                    second_length = float(
                        np.linalg.norm(second_vector)
                    )
                    if second_length < MIN_LAYOUT_CONTACT_LENGTH_CM:
                        continue
                    second_tangent = second_vector / second_length
                    if (
                        abs(float(np.dot(tangent, second_tangent)))
                        < MIN_LAYOUT_CONTACT_PARALLEL_COSINE
                    ):
                        continue
                    first_distance = float(
                        np.dot(second_start - first_start, normal)
                    )
                    second_distance = float(
                        np.dot(second_end - first_start, normal)
                    )
                    if (
                        max(
                            abs(first_distance),
                            abs(second_distance),
                        )
                        > MAX_LAYOUT_CONTACT_DISTANCE_CM
                    ):
                        continue
                    second_projection = sorted(
                        (
                            float(
                                np.dot(
                                    second_start - first_start,
                                    tangent,
                                )
                            ),
                            float(
                                np.dot(
                                    second_end - first_start,
                                    tangent,
                                )
                            ),
                        )
                    )
                    overlap_start = max(0.0, second_projection[0])
                    overlap_end = min(
                        first_length,
                        second_projection[1],
                    )
                    overlap_length = overlap_end - overlap_start
                    if overlap_length < MIN_LAYOUT_CONTACT_LENGTH_CM:
                        continue
                    # Sample on the midline between the two fitted polygon
                    # edges so small contour gaps do not bias either piece.
                    middle_distance = 0.25 * (
                        first_distance + second_distance
                    )
                    contact_start = (
                        first_start
                        + tangent * overlap_start
                        + normal * middle_distance
                    )
                    contact_end = (
                        first_start
                        + tangent * overlap_end
                        + normal * middle_distance
                    )
                    contacts.append(
                        (
                            first_placement,
                            second_placement,
                            contact_start,
                            contact_end,
                        )
                    )
    return contacts


def _source_edge_segment_score(
    context: TextureContext,
    long_polygon: np.ndarray,
    long_first: np.ndarray,
    long_second: np.ndarray,
    short_polygon: np.ndarray,
    short_first: np.ndarray,
    short_second: np.ndarray,
) -> tuple[float, float, float]:
    """Compare two source edges in their exact reverse correspondence."""
    length = 0.5 * (
        float(np.linalg.norm(long_second - long_first))
        + float(np.linalg.norm(short_second - short_first))
    )
    if length < 0.35:
        return 0.0, 0.0, length
    long_normal = _inward_normal(
        long_polygon,
        long_first,
        long_second,
    )
    short_normal = _inward_normal(
        short_polygon,
        short_first,
        short_second,
    )
    if long_normal is None or short_normal is None:
        return 0.0, 0.0, length
    sample_count = int(
        np.clip(
            round(length * SAMPLES_PER_CM),
            MIN_SAMPLES_PER_SEGMENT,
            MAX_SAMPLES_PER_SEGMENT,
        )
    )
    fractions = np.linspace(
        ENDPOINT_MARGIN,
        1.0 - ENDPOINT_MARGIN,
        sample_count,
    )
    long_base = (
        long_first[None, :]
        + fractions[:, None] * (long_second - long_first)[None, :]
    )
    short_base = (
        short_first[None, :]
        + fractions[:, None]
        * (short_second - short_first)[None, :]
    )
    long_lab_layers = []
    short_lab_layers = []
    long_gray_layers = []
    short_gray_layers = []
    all_valid = np.ones(sample_count, dtype=bool)
    for depth in SAMPLE_DEPTHS_CM:
        long_points = (
            long_base + long_normal[None, :] * depth
        ) * context.px_per_cm
        short_points = (
            short_base + short_normal[None, :] * depth
        ) * context.px_per_cm
        long_lab, valid_a = _bilinear_samples(
            context.lab,
            long_points,
        )
        short_lab, valid_b = _bilinear_samples(
            context.lab,
            short_points,
        )
        long_gray, valid_c = _bilinear_samples(
            context.gray,
            long_points,
        )
        short_gray, valid_d = _bilinear_samples(
            context.gray,
            short_points,
        )
        all_valid &= valid_a & valid_b & valid_c & valid_d
        long_lab_layers.append(long_lab)
        short_lab_layers.append(short_lab)
        long_gray_layers.append(long_gray)
        short_gray_layers.append(short_gray)
    return _score_sample_layers(
        long_lab_layers,
        short_lab_layers,
        long_gray_layers,
        short_gray_layers,
        all_valid,
        length,
    )


def _score_source_family(
    context: TextureContext,
    original_pieces: list[np.ndarray],
    family,
    order,
) -> tuple[float, float, float, int]:
    """Return weighted error, information, length and segment count."""
    weighted_error = 0.0
    information_weight = 0.0
    total_length = 0.0
    segment_count = 0
    long_piece = np.asarray(
        original_pieces[family.long_edge.piece_id],
        dtype=np.float64,
    )
    long_first = long_piece[family.long_edge.edge_id]
    long_second = long_piece[
        (family.long_edge.edge_id + 1) % len(long_piece)
    ]
    short_lengths = []
    for edge in order:
        piece = np.asarray(
            original_pieces[edge.piece_id],
            dtype=np.float64,
        )
        short_lengths.append(
            float(
                np.linalg.norm(
                    piece[(edge.edge_id + 1) % len(piece)]
                    - piece[edge.edge_id]
                )
            )
        )
    total_short = max(sum(short_lengths), 1e-9)
    cursor = 0.0
    for edge, short_length in zip(order, short_lengths):
        next_cursor = cursor + short_length / total_short
        segment_first = (
            long_first + cursor * (long_second - long_first)
        )
        segment_second = (
            long_first + next_cursor * (long_second - long_first)
        )
        short_piece = np.asarray(
            original_pieces[edge.piece_id],
            dtype=np.float64,
        )
        # seam_point_pairs() aligns the long direction to the reverse
        # direction of every short edge.
        short_first = short_piece[
            (edge.edge_id + 1) % len(short_piece)
        ]
        short_second = short_piece[edge.edge_id]
        error, activity, length = _source_edge_segment_score(
            context,
            long_piece,
            segment_first,
            segment_second,
            short_piece,
            short_first,
            short_second,
        )
        weight = length * activity
        weighted_error += error * weight
        information_weight += weight
        total_length += length
        segment_count += 1
        cursor = next_cursor
    return (
        weighted_error,
        information_weight,
        total_length,
        segment_count,
    )


def score_source_topology(
    context: TextureContext | None,
    original_pieces: list[np.ndarray],
    topology,
    orders,
    family_cache: dict | None = None,
) -> dict:
    """Score candidate edge pairing directly in the rectified source."""
    if context is None:
        return {
            "score": 0.0,
            "confidence": 0.0,
            "segments": 0,
        }
    weighted_error = 0.0
    information_weight = 0.0
    total_length = 0.0
    segment_count = 0
    for family, order in zip(topology, orders):
        cache_key = (family.signature, order)
        cached = (
            None
            if family_cache is None
            else family_cache.get(cache_key)
        )
        if cached is None:
            cached = _score_source_family(
                context,
                original_pieces,
                family,
                order,
            )
            if family_cache is not None:
                family_cache[cache_key] = cached
        family_error, family_info, family_length, family_segments = (
            cached
        )
        weighted_error += family_error
        information_weight += family_info
        total_length += family_length
        segment_count += family_segments
    if information_weight <= 1e-8 or total_length <= 1e-8:
        return {
            "score": 0.0,
            "confidence": 0.0,
            "segments": segment_count,
            "seam_score": 0.0,
            "perimeter_score": 0.0,
        }
    seam_score = float(weighted_error / information_weight)
    perimeter_score = seam_score
    card_confidence = float(context.white_card_confidence)
    if (
        card_confidence > 0.0
        and context.edge_border_costs
    ):
        internal_edges = set()
        for family in topology:
            internal_edges.add(
                (
                    family.long_edge.piece_id,
                    family.long_edge.edge_id,
                )
            )
            internal_edges.update(
                (edge.piece_id, edge.edge_id)
                for edge in family.short_edges
            )
        perimeter_error = 0.0
        perimeter_length = 0.0
        for edge_ref, result in context.edge_border_costs.items():
            if edge_ref in internal_edges:
                continue
            edge_error, edge_length = result
            perimeter_error += edge_error * edge_length
            perimeter_length += edge_length
        if perimeter_length > 1e-8:
            perimeter_score = float(
                perimeter_error / perimeter_length
            )
    perimeter_blend = (
        CARD_SOURCE_PERIMETER_BLEND * card_confidence
    )
    combined_score = (
        (1.0 - perimeter_blend) * seam_score
        + perimeter_blend * perimeter_score
    )
    return {
        "score": float(combined_score),
        "confidence": float(
            np.clip(
                information_weight / (0.35 * total_length),
                0.0,
                1.0,
            )
        ),
        "segments": segment_count,
        "seam_score": seam_score,
        "perimeter_score": perimeter_score,
    }


def _score_half_turn_symmetry(
    context: TextureContext,
    placements: list,
) -> tuple[float, float]:
    """Compare a low-resolution assembled card with its 180-degree turn."""
    points = np.concatenate(
        [placement.vertices for placement in placements],
        axis=0,
    ).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(points)).astype(np.float64)
    origin = box[0]
    u_axis = box[1] - origin
    v_axis = box[3] - origin
    u_length = float(np.linalg.norm(u_axis))
    v_length = float(np.linalg.norm(v_axis))
    if min(u_length, v_length) < 1e-8:
        return 1.0, 0.0
    width = max(
        24,
        int(round(u_length * CARD_SYMMETRY_SAMPLES_PER_CM)),
    )
    height = max(
        24,
        int(round(v_length * CARD_SYMMETRY_SAMPLES_PER_CM)),
    )
    x_values = (np.arange(width) + 0.5) / width
    y_values = (np.arange(height) + 0.5) / height
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    target_points = (
        origin[None, None, :]
        + grid_x[:, :, None] * u_axis[None, None, :]
        + grid_y[:, :, None] * v_axis[None, None, :]
    )
    lab_grid = np.zeros((height, width, 3), dtype=np.float64)
    covered = np.zeros((height, width), dtype=bool)
    inverse_axes = np.linalg.inv(
        np.column_stack((u_axis, v_axis))
    )
    for placement in placements:
        local_vertices = (
            np.asarray(placement.vertices, dtype=np.float64)
            - origin
        ) @ inverse_axes.T
        raster_vertices = np.round(
            local_vertices * np.asarray([width, height])
        ).astype(np.int32)
        piece_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(piece_mask, [raster_vertices], 255)
        selected = (piece_mask > 0) & (~covered)
        if not np.any(selected):
            continue
        source_points = (
            _source_points(
                placement,
                target_points[selected],
            )
            * context.px_per_cm
        )
        values, valid = _bilinear_samples(
            context.lab,
            source_points,
        )
        indices = np.argwhere(selected)[valid]
        if len(indices) == 0:
            continue
        lab_grid[indices[:, 0], indices[:, 1]] = values[valid]
        covered[indices[:, 0], indices[:, 1]] = True
    reverse_lab = lab_grid[::-1, ::-1]
    valid_pairs = covered & covered[::-1, ::-1]
    pair_count = int(np.count_nonzero(valid_pairs))
    if pair_count < max(48, (width * height) // 2):
        return 1.0, float(pair_count / max(1, width * height))
    delta = np.mean(
        np.minimum(
            1.0,
            np.abs(lab_grid - reverse_lab)
            / np.asarray([70.0, 45.0, 45.0]),
        ),
        axis=2,
    )
    return (
        float(np.mean(delta[valid_pairs])),
        float(pair_count / (width * height)),
    )


def _score_layout_perimeter(
    context: TextureContext,
    placements: list,
) -> tuple[float, float, int]:
    """Score only source edges exposed on the final card rectangle.

    Topology edges are insufficient here: a spanning-tree topology can omit
    real contacts and can therefore mistake a printed cut edge for the card
    perimeter.  Classifying transformed edges against the completed minimum
    rectangle makes the test depend on the actual final layout.
    """
    if not context.edge_border_costs:
        return 1.0, 0.0, 0
    points = np.concatenate(
        [np.asarray(item.vertices, dtype=np.float64) for item in placements],
        axis=0,
    ).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(points)).astype(np.float64)
    origin = box[0]
    u_vector = box[1] - origin
    v_vector = box[3] - origin
    u_length = float(np.linalg.norm(u_vector))
    v_length = float(np.linalg.norm(v_vector))
    if min(u_length, v_length) < 1e-8:
        return 1.0, 0.0, 0
    u_axis = u_vector / u_length
    v_axis = v_vector / v_length
    weighted_error = 0.0
    exposed_length = 0.0
    exposed_edges = 0
    for placement in placements:
        polygon = np.asarray(placement.vertices, dtype=np.float64)
        for edge_id in range(len(polygon)):
            first = polygon[edge_id]
            second = polygon[(edge_id + 1) % len(polygon)]
            first_local = np.asarray(
                [
                    np.dot(first - origin, u_axis),
                    np.dot(first - origin, v_axis),
                ]
            )
            second_local = np.asarray(
                [
                    np.dot(second - origin, u_axis),
                    np.dot(second - origin, v_axis),
                ]
            )
            on_boundary = (
                max(abs(first_local[0]), abs(second_local[0]))
                <= MAX_LAYOUT_PERIMETER_DISTANCE_CM
                or max(
                    abs(first_local[0] - u_length),
                    abs(second_local[0] - u_length),
                )
                <= MAX_LAYOUT_PERIMETER_DISTANCE_CM
                or max(abs(first_local[1]), abs(second_local[1]))
                <= MAX_LAYOUT_PERIMETER_DISTANCE_CM
                or max(
                    abs(first_local[1] - v_length),
                    abs(second_local[1] - v_length),
                )
                <= MAX_LAYOUT_PERIMETER_DISTANCE_CM
            )
            if not on_boundary:
                continue
            result = context.edge_border_costs.get(
                (int(placement.piece_id), edge_id)
            )
            if result is None:
                continue
            edge_error, edge_length = result
            weighted_error += edge_error * edge_length
            exposed_length += edge_length
            exposed_edges += 1
    expected_length = 2.0 * (u_length + v_length)
    confidence = float(
        np.clip(
            exposed_length / max(expected_length, 1e-8),
            0.0,
            1.0,
        )
    )
    if exposed_length <= 1e-8:
        return 1.0, 0.0, exposed_edges
    return (
        float(weighted_error / exposed_length),
        confidence,
        exposed_edges,
    )


def score_layout(
    context: TextureContext | None,
    original_pieces: list[np.ndarray],
    placements: list,
    topology,
    orders,
) -> dict:
    """Score pattern continuity for one already legal geometric layout."""
    if context is None:
        return {
            "score": 0.0,
            "confidence": 0.0,
            "segments": 0,
        }
    weighted_error = 0.0
    information_weight = 0.0
    total_length = 0.0
    segment_count = 0

    contacts = _layout_contact_segments(placements)
    for (
        first_placement,
        second_placement,
        first,
        second,
    ) in contacts:
        error, activity, length = _segment_score(
            context,
            first_placement,
            second_placement,
            first,
            second,
        )
        weight = length * activity
        weighted_error += error * weight
        information_weight += weight
        total_length += length
        segment_count += 1

    if information_weight <= 1e-8 or total_length <= 1e-8:
        return {
            "score": 0.0,
            "confidence": 0.0,
            "segments": segment_count,
            "seam_score": 0.0,
            "symmetry_score": 0.0,
            "perimeter_score": 1.0,
            "perimeter_confidence": 0.0,
        }
    seam_score = float(weighted_error / information_weight)
    symmetry_score = seam_score
    symmetry_confidence = 0.0
    card_confidence = float(context.white_card_confidence)
    if card_confidence > 0.0:
        symmetry_score, symmetry_confidence = (
            _score_half_turn_symmetry(
                context,
                placements,
            )
        )
    symmetry_blend = (
        CARD_LAYOUT_SYMMETRY_BLEND
        * card_confidence
        * symmetry_confidence
    )
    combined_score = (
        (1.0 - symmetry_blend) * seam_score
        + symmetry_blend * symmetry_score
    )
    perimeter_score = combined_score
    perimeter_confidence = 0.0
    perimeter_edges = 0
    if card_confidence > 0.0:
        (
            perimeter_score,
            perimeter_confidence,
            perimeter_edges,
        ) = _score_layout_perimeter(context, placements)
    perimeter_blend = (
        CARD_LAYOUT_PERIMETER_BLEND
        * card_confidence
        * perimeter_confidence
    )
    combined_score = (
        (1.0 - perimeter_blend) * combined_score
        + perimeter_blend * perimeter_score
    )
    return {
        "score": float(combined_score),
        "confidence": float(
            np.clip(
                information_weight / (0.35 * total_length),
                0.0,
                1.0,
            )
        ),
        "segments": segment_count,
        "seam_score": seam_score,
        "symmetry_score": symmetry_score,
        "symmetry_confidence": symmetry_confidence,
        "perimeter_score": perimeter_score,
        "perimeter_confidence": perimeter_confidence,
        "perimeter_edges": perimeter_edges,
        "contact_segments": len(contacts),
    }
