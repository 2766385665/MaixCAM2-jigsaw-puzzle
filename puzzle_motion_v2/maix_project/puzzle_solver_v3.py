"""Global seam-topology puzzle solver for random piece rotations.

This module is intentionally separate from puzzle_solver.py while it is
validated on the PC. It uses only OpenCV and Numpy so the same file can run
on MaixCAM 2 after validation.
"""

from __future__ import annotations

import itertools
import math
import time as pytime
from dataclasses import dataclass

import cv2
import numpy as np

import puzzle_solver as legacy
import texture_matcher


Placement = legacy.Placement
Solution = legacy.Solution
canonical_target_solution = legacy.canonical_target_solution

DIRECT_ABS_TOLERANCE_CM = 0.55
DIRECT_REL_TOLERANCE = 0.10
CHAIN_ABS_TOLERANCE_CM = 0.75
CHAIN_REL_TOLERANCE = 0.12
MIN_EDGE_CM = 1.65

MAX_CHAIN_PARTS = 2
MAX_SEAM_FAMILIES = 64
MAX_ENUMERATED_TOPOLOGIES = 700
MAX_RELAXED_TOPOLOGIES = 96
MAX_COARSE_TOPOLOGIES = 80
MAX_TOPOLOGIES_TO_OPTIMIZE = 8
FAST_STRICT_TOPOLOGIES = 120
FAST_COARSE_TOPOLOGIES = 32

HUBER_DELTA_CM = 0.28
POSE_OPTIMIZATION_ITERATIONS = 6
POSE_DAMPING = 1e-3
# Polygon fitting perturbs area and perimeter independently.  A physically
# valid topology can therefore have a slightly negative quadratic
# discriminant before pose optimization.  Keep this pre-filter tolerant and
# leave the strict decision to the final IoU/overlap/seam quality gates.
MIN_PERIMETER_DISCRIMINANT_CM2 = -4.0
STRICT_MIN_PERIMETER_DISCRIMINANT_CM2 = -2.0

MIN_RECTANGLE_IOU = 0.89
MAX_OVERLAP_RATIO = 0.055
MAX_SEAM_RMS_CM = 0.50
UNIQUE_SCORE_MARGIN = 0.018
TEXTURE_SCORE_WEIGHT = 0.28
SOURCE_TEXTURE_BLEND = 0.55
ROUGH_TEXTURE_RANK_WEIGHT = 0.22
# Geometry-only layouts need a deliberately conservative margin because
# several different seam topologies can produce similar rectangles.  Texture
# already supplies independent evidence, so reusing the geometry margin made
# patterned cards impossible to accept even when the seam pixels were active.
TEXTURE_UNIQUE_SCORE_MARGIN = 0.0025
TEXTURE_UNIQUE_RAW_MARGIN = 0.005
# Once a candidate already passes the strict rectangle/overlap/seam gates,
# the measured pattern discontinuity on the *final physical contacts* is
# stronger evidence than a tiny difference in the mixed geometry score.
# This specifically resolves equal-shape pieces whose top/bottom positions
# can be swapped while preserving almost exactly the same rectangle.
DIRECT_SEAM_UNIQUE_MARGIN = 0.040
DIRECT_SEAM_MIN_CONTACTS_PER_PIECE = 1
TEXTURE_MIN_CONFIDENCE = 0.75
TEXTURE_ACCEPT_IOU = 0.94
TEXTURE_ACCEPT_MAX_OVERLAP_RATIO = 0.02
TEXTURE_ACCEPT_MAX_SEAM_RMS_CM = 0.20
EARLY_ACCEPT_IOU = 0.94
EARLY_ACCEPT_MAX_OVERLAP_RATIO = 0.015
EARLY_ACCEPT_MAX_SEAM_RMS_CM = 0.20
EARLY_ACCEPT_SCORE_MARGIN = 0.06
# A patterned white card can be accepted earlier than a geometry-only puzzle,
# but only when several independent checks agree.  These thresholds are
# intentionally stricter than the final acceptance gate so an early exit can
# never be triggered by 180-degree symmetry alone.
CARD_EARLY_ACCEPT_IOU = 0.95
CARD_EARLY_ACCEPT_MAX_OVERLAP_RATIO = 0.02
CARD_EARLY_ACCEPT_MAX_SEAM_RMS_CM = 0.20
CARD_EARLY_ACCEPT_MIN_TEXTURE_CONFIDENCE = 0.90
CARD_EARLY_ACCEPT_MIN_PERIMETER_CONFIDENCE = 0.90
CARD_EARLY_ACCEPT_MIN_TEXTURE_MARGIN = 0.05
CARD_EARLY_ACCEPT_MIN_SCORE_MARGIN = 0.005
CARD_EARLY_ACCEPT_MIN_CONTACT_SEGMENTS = 4

# These are deliberately broad sanity limits, not a prescribed puzzle size.
# The old 8.60 cm minimum long side rejected a valid playing-card layout when
# manual A4 alignment introduced about 7% scale drift.  Shape quality is still
# guarded by IoU, overlap and seam-RMS thresholds below.
TARGET_MIN_SHORT_CM = 4.00
TARGET_MAX_SHORT_CM = 10.00
TARGET_MIN_LONG_CM = 7.50
TARGET_MAX_LONG_CM = 13.00

# A standard playing card is close to 1.53:1 after the rounded corners and
# contour fitting are taken into account.  The three real Q/J regression
# captures measure 1.527--1.531.  Without this constraint a high-IoU 2.49:1
# strip can be assembled from the same four pieces and the texture heuristic
# has no reliable way to turn that strip back into a card.
WHITE_CARD_MIN_ASPECT_RATIO = 1.45
WHITE_CARD_MAX_ASPECT_RATIO = 1.61
WHITE_CARD_MODE_CONFIDENCE = 0.50

# Keep the former, narrower range as the primary candidate band.  Layouts
# admitted only by the broader emergency range use the existing relaxed
# quota, so they cannot evict proven candidates from the bounded Maix search.
CORE_MIN_SHORT_CM = 4.60
CORE_MAX_SHORT_CM = 9.45
CORE_MIN_LONG_CM = 8.60
CORE_MAX_LONG_CM = 12.40

DEFAULT_MAX_SECONDS = 14.0
# Compatibility names used by the MaixCAM UI.  Keeping these here lets the
# old solver remain untouched and makes switching versions a one-line import.
MAX_SEARCH_SECONDS = DEFAULT_MAX_SECONDS

_LAST_DIAGNOSTICS: dict = {}


@dataclass(frozen=True, order=True)
class EdgeRef:
    piece_id: int
    edge_id: int


@dataclass(frozen=True)
class SeamFamily:
    long_edge: EdgeRef
    short_edges: tuple[EdgeRef, ...]
    length_error_cm: float
    normalized_error: float
    used_mask: int
    internal_length_cm: float

    @property
    def pieces(self) -> frozenset[int]:
        return frozenset(
            [self.long_edge.piece_id]
            + [edge.piece_id for edge in self.short_edges]
        )

    @property
    def signature(self) -> tuple:
        return (
            self.long_edge,
            tuple(sorted(self.short_edges)),
        )


@dataclass
class LayoutCandidate:
    solution: Solution
    score: float
    iou: float
    overlap_ratio: float
    seam_rms_cm: float
    texture_score: float
    texture_confidence: float
    source_texture_score: float
    layout_texture_score: float
    source_seam_score: float
    source_perimeter_score: float
    layout_seam_score: float
    layout_symmetry_score: float
    layout_perimeter_score: float
    layout_perimeter_confidence: float
    layout_contact_segments: int
    layout_variant: str
    topology_signature: tuple


def last_diagnostics() -> dict:
    return dict(_LAST_DIAGNOSTICS)


def last_search_timed_out() -> bool:
    return bool(_LAST_DIAGNOSTICS.get("timed_out", False))


def edge_vertices(
    piece: np.ndarray,
    edge: EdgeRef,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        piece[edge.edge_id],
        piece[(edge.edge_id + 1) % len(piece)],
    )


def transform_points(
    points: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    angle, tx, ty = pose
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    return points @ rotation.T + np.asarray([tx, ty])


def tolerance_match(
    first: float,
    second: float,
    absolute: float,
    relative: float,
) -> tuple[bool, float]:
    tolerance = max(absolute, relative * max(first, second))
    error = abs(first - second)
    return error <= tolerance, error / max(tolerance, 1e-9)


def build_edge_tables(
    pieces: list[np.ndarray],
) -> tuple[
    list[list[float]],
    dict[tuple[int, int], int],
]:
    lengths: list[list[float]] = []
    bit_index: dict[tuple[int, int], int] = {}
    bit = 0
    for piece_id, piece in enumerate(pieces):
        piece_lengths = []
        for edge_id, first, second in legacy.polygon_edges(piece):
            piece_lengths.append(legacy.edge_length(first, second))
            bit_index[(piece_id, edge_id)] = bit
            bit += 1
        lengths.append(piece_lengths)
    return lengths, bit_index


def edge_mask(
    refs: tuple[EdgeRef, ...],
    bit_index: dict[tuple[int, int], int],
) -> int:
    mask = 0
    for ref in refs:
        mask |= 1 << bit_index[(ref.piece_id, ref.edge_id)]
    return mask


def generate_seam_families(
    pieces: list[np.ndarray],
) -> tuple[list[SeamFamily], list[list[float]]]:
    lengths, bit_index = build_edge_tables(pieces)
    edges = [
        EdgeRef(piece_id, edge_id)
        for piece_id, piece_lengths in enumerate(lengths)
        for edge_id in range(len(piece_lengths))
        if piece_lengths[edge_id] >= MIN_EDGE_CM
    ]
    families: list[SeamFamily] = []

    # Complete edge to complete edge.
    for first_index, first in enumerate(edges):
        for second in edges[first_index + 1:]:
            if first.piece_id == second.piece_id:
                continue
            first_length = lengths[first.piece_id][first.edge_id]
            second_length = lengths[second.piece_id][second.edge_id]
            matches, normalized_error = tolerance_match(
                first_length,
                second_length,
                DIRECT_ABS_TOLERANCE_CM,
                DIRECT_REL_TOLERANCE,
            )
            if not matches:
                continue
            if (first.piece_id, first.edge_id) <= (
                second.piece_id,
                second.edge_id,
            ):
                long_edge, short_edges = first, (second,)
            else:
                long_edge, short_edges = second, (first,)
            refs = (long_edge,) + short_edges
            families.append(
                SeamFamily(
                    long_edge=long_edge,
                    short_edges=short_edges,
                    length_error_cm=abs(
                        first_length - second_length
                    ),
                    normalized_error=normalized_error,
                    used_mask=edge_mask(refs, bit_index),
                    internal_length_cm=(
                        first_length + second_length
                    ) * 0.5,
                )
            )

    # One long edge to two or three shorter edges on distinct pieces.
    for long_edge in edges:
        long_length = lengths[
            long_edge.piece_id
        ][long_edge.edge_id]
        possible_shorts = [
            edge
            for edge in edges
            if (
                edge.piece_id != long_edge.piece_id
                and lengths[edge.piece_id][edge.edge_id]
                < long_length
            )
        ]
        for part_count in range(2, MAX_CHAIN_PARTS + 1):
            for short_edges in itertools.combinations(
                possible_shorts,
                part_count,
            ):
                piece_ids = [edge.piece_id for edge in short_edges]
                if len(set(piece_ids)) != len(piece_ids):
                    continue
                short_sum = sum(
                    lengths[edge.piece_id][edge.edge_id]
                    for edge in short_edges
                )
                matches, normalized_error = tolerance_match(
                    long_length,
                    short_sum,
                    CHAIN_ABS_TOLERANCE_CM,
                    CHAIN_REL_TOLERANCE,
                )
                if not matches:
                    continue
                ordered_refs = tuple(sorted(short_edges))
                refs = (long_edge,) + ordered_refs
                families.append(
                    SeamFamily(
                        long_edge=long_edge,
                        short_edges=ordered_refs,
                        length_error_cm=abs(
                            long_length - short_sum
                        ),
                        normalized_error=normalized_error,
                        used_mask=edge_mask(refs, bit_index),
                        internal_length_cm=(
                            long_length + short_sum
                        ) * 0.5,
                    )
                )

    # Deduplicate and keep the most plausible families. A small penalty for
    # additional split parts prevents noisy three-part chains dominating.
    deduplicated: dict[tuple, SeamFamily] = {}
    for family in families:
        old = deduplicated.get(family.signature)
        if old is None or family.normalized_error < old.normalized_error:
            deduplicated[family.signature] = family
    result = sorted(
        deduplicated.values(),
        key=lambda family: (
            family.normalized_error
            + 0.08 * (len(family.short_edges) - 1),
            -family.internal_length_cm,
        ),
    )
    return result[:MAX_SEAM_FAMILIES], lengths


def connected_piece_count(
    piece_count: int,
    families: tuple[SeamFamily, ...],
) -> tuple[int, int]:
    parents = list(range(piece_count))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    used_pieces = set()
    for family in families:
        long_id = family.long_edge.piece_id
        used_pieces.add(long_id)
        for edge in family.short_edges:
            used_pieces.add(edge.piece_id)
            union(long_id, edge.piece_id)
    components = len({find(piece_id) for piece_id in used_pieces})
    return len(used_pieces), components


def plausible_outer_rectangle(
    pieces: list[np.ndarray],
    families: tuple[SeamFamily, ...],
) -> tuple[bool, float]:
    total_area = sum(legacy.polygon_area(piece) for piece in pieces)
    total_perimeter = sum(
        legacy.edge_length(first, second)
        for piece in pieces
        for _, first, second in legacy.polygon_edges(piece)
    )
    internal_length = sum(
        family.internal_length_cm for family in families
    )
    outer_perimeter = total_perimeter - 2.0 * internal_length
    side_sum = outer_perimeter * 0.5
    discriminant = side_sum * side_sum - 4.0 * total_area
    if discriminant < MIN_PERIMETER_DISCRIMINANT_CM2:
        return False, float("inf")
    discriminant = max(0.0, discriminant)
    root = math.sqrt(discriminant)
    short_side = (side_sum - root) * 0.5
    long_side = (side_sum + root) * 0.5
    # This is only a topology pre-filter. Perimeter is particularly
    # sensitive to hand-cut and polygon-fit errors, so keep a wider margin
    # here and apply the real dimension limits after pose optimization.
    topology_margin = 0.75
    legal = (
        TARGET_MIN_SHORT_CM - topology_margin
        <= short_side
        <= TARGET_MAX_SHORT_CM + topology_margin
        and TARGET_MIN_LONG_CM - topology_margin
        <= long_side
        <= TARGET_MAX_LONG_CM + topology_margin
    )
    if legal:
        return True, 0.0

    short_error = max(
        TARGET_MIN_SHORT_CM - short_side,
        short_side - TARGET_MAX_SHORT_CM,
        0.0,
    )
    long_error = max(
        TARGET_MIN_LONG_CM - long_side,
        long_side - TARGET_MAX_LONG_CM,
        0.0,
    )
    return False, short_error + long_error


def topology_heuristic(
    piece_count: int,
    pieces: list[np.ndarray],
    families: tuple[SeamFamily, ...],
) -> float:
    used_count, components = connected_piece_count(
        piece_count,
        families,
    )
    # An incomplete topology naturally has too much outer perimeter. Do not
    # penalize that intermediate state or valid noisy chains are pruned before
    # their final seam is added.
    rectangle_error = 0.0
    if used_count == piece_count and components == 1:
        _, rectangle_error = plausible_outer_rectangle(
            pieces,
            families,
        )
    return (
        0.25 * sum(
            family.normalized_error for family in families
        )
        + 0.05 * sum(
            len(family.short_edges) - 1 for family in families
        )
        + 0.55 * (piece_count - used_count)
        + 0.35 * max(0, components - 1)
        + 0.8 * rectangle_error
    )


def every_piece_keeps_outer_edge(
    pieces: list[np.ndarray],
    families: tuple[SeamFamily, ...],
) -> bool:
    used = [set() for _ in pieces]
    for family in families:
        used[family.long_edge.piece_id].add(
            family.long_edge.edge_id
        )
        for edge in family.short_edges:
            used[edge.piece_id].add(edge.edge_id)
    return all(
        len(used[piece_id]) < len(piece)
        for piece_id, piece in enumerate(pieces)
    )


def enumerate_topologies(
    pieces: list[np.ndarray],
    families: list[SeamFamily],
    deadline: float | None = None,
    return_strict_count: bool = False,
):
    """Enumerate connected seam topologies using four-piece bit masks.

    The earlier implementation rebuilt Python sets and a union-find structure
    for every 1/2/3-family combination.  That was acceptable on a PC but the
    same 63-family case spent almost 30 seconds here on MaixCAM 2 before it
    reached the first deadline check.  Precomputing masks makes the hot loop
    integer-only and the deadline keeps this stage bounded.
    """
    piece_count = len(pieces)
    all_pieces_mask = (1 << piece_count) - 1
    total_area = sum(legacy.polygon_area(piece) for piece in pieces)
    total_perimeter = sum(
        legacy.edge_length(first, second)
        for piece in pieces
        for _, first, second in legacy.polygon_edges(piece)
    )

    records = []
    for family in families:
        long_piece = family.long_edge.piece_id
        piece_mask = 1 << long_piece
        adjacency = 0
        edge_masks = [0] * piece_count
        edge_masks[long_piece] |= 1 << family.long_edge.edge_id
        for edge in family.short_edges:
            short_piece = edge.piece_id
            piece_mask |= 1 << short_piece
            adjacency |= 1 << (
                long_piece * piece_count + short_piece
            )
            adjacency |= 1 << (
                short_piece * piece_count + long_piece
            )
            edge_masks[short_piece] |= 1 << edge.edge_id
        records.append(
            (
                family,
                piece_mask,
                adjacency,
                tuple(edge_masks),
                0.25 * family.normalized_error
                + 0.05 * (len(family.short_edges) - 1),
            )
        )

    def graph_is_connected(adjacency: int) -> bool:
        visited = 1
        while True:
            expanded = visited
            pending = visited
            while pending:
                lowest = pending & -pending
                piece_id = lowest.bit_length() - 1
                expanded |= (
                    adjacency >> (piece_id * piece_count)
                ) & all_pieces_mask
                pending ^= lowest
            if expanded == visited:
                return visited == all_pieces_mask
            visited = expanded

    def dimension_plausibility_class(
        internal_length: float,
    ) -> int:
        """Return 0=reject, 1=relaxed, 2=strict.

        Relaxed candidates have their own quota so noisy perimeter estimates
        cannot displace the complete legacy candidate set.
        """
        outer_perimeter = (
            total_perimeter - 2.0 * internal_length
        )
        side_sum = outer_perimeter * 0.5
        discriminant = (
            side_sum * side_sum - 4.0 * total_area
        )
        if discriminant < MIN_PERIMETER_DISCRIMINANT_CM2:
            return 0
        strict = (
            discriminant
            >= STRICT_MIN_PERIMETER_DISCRIMINANT_CM2
        )
        root = math.sqrt(max(0.0, discriminant))
        short_side = (side_sum - root) * 0.5
        long_side = (side_sum + root) * 0.5
        margin = 0.75
        legal = (
            TARGET_MIN_SHORT_CM - margin
            <= short_side
            <= TARGET_MAX_SHORT_CM + margin
            and TARGET_MIN_LONG_CM - margin
            <= long_side
            <= TARGET_MAX_LONG_CM + margin
        )
        if not legal:
            return 0
        return 2 if strict else 1

    accepted: list[
        tuple[float, tuple, tuple[SeamFamily, ...]]
    ] = []
    relaxed_accepted: list[
        tuple[float, tuple, tuple[SeamFamily, ...]]
    ] = []
    checked = 0

    # With at most four pieces, a connected contact topology always has a
    # spanning description using at most three seam families. Exhaustively
    # checking 2- and 3-family combinations is deterministic and avoids the
    # branch-order failures of the old beam/DFS search.
    for family_count in (1, 2, 3):
        for selected_records in itertools.combinations(
            records,
            family_count,
        ):
            checked += 1
            if (
                checked & 0x3F == 0
                and deadline is not None
                and pytime.monotonic() >= deadline
            ):
                break
            used_mask = 0
            piece_mask = 0
            adjacency = 0
            edge_masks = [0] * piece_count
            internal_length = 0.0
            heuristic = 0.0
            conflict = False
            selected = []
            for record in selected_records:
                (
                    family,
                    family_piece_mask,
                    family_adjacency,
                    family_edge_masks,
                    family_heuristic,
                ) = record
                if used_mask & family.used_mask:
                    conflict = True
                    break
                used_mask |= family.used_mask
                piece_mask |= family_piece_mask
                adjacency |= family_adjacency
                internal_length += family.internal_length_cm
                heuristic += family_heuristic
                selected.append(family)
                for piece_id in range(piece_count):
                    edge_masks[piece_id] |= (
                        family_edge_masks[piece_id]
                    )
            if conflict:
                continue
            if (
                piece_mask != all_pieces_mask
                or not graph_is_connected(adjacency)
            ):
                continue
            plausibility_class = dimension_plausibility_class(
                internal_length
            )
            if not plausibility_class:
                continue
            if any(
                edge_masks[piece_id].bit_count()
                >= len(pieces[piece_id])
                for piece_id in range(piece_count)
            ):
                continue
            selected_tuple = tuple(selected)
            signature = tuple(
                sorted(family.signature for family in selected)
            )
            destination = (
                accepted
                if plausibility_class == 2
                else relaxed_accepted
            )
            destination.append(
                (heuristic, signature, selected_tuple)
            )
        if (
            deadline is not None
            and pytime.monotonic() >= deadline
        ):
            break

    accepted.sort(key=lambda item: (item[0], item[1]))
    relaxed_accepted.sort(
        key=lambda item: (item[0], item[1])
    )
    strict_topologies = [
        topology
        for _, _, topology in accepted[
            :MAX_ENUMERATED_TOPOLOGIES
        ]
    ]
    relaxed_topologies = [
        topology
        for _, _, topology in relaxed_accepted[
            :MAX_RELAXED_TOPOLOGIES
        ]
    ]
    combined = strict_topologies + relaxed_topologies
    if return_strict_count:
        return combined, len(strict_topologies)
    return combined


def seam_point_pairs(
    pieces: list[np.ndarray],
    family: SeamFamily,
    short_order: tuple[EdgeRef, ...],
) -> list[tuple[int, np.ndarray, int, np.ndarray]]:
    long_first, long_second = edge_vertices(
        pieces[family.long_edge.piece_id],
        family.long_edge,
    )
    short_lengths = []
    for edge in short_order:
        first, second = edge_vertices(pieces[edge.piece_id], edge)
        short_lengths.append(legacy.edge_length(first, second))
    total_short_length = max(sum(short_lengths), 1e-9)

    pairs = []
    cursor = 0.0
    for edge, short_length in zip(short_order, short_lengths):
        next_cursor = cursor + short_length / total_short_length
        long_start = legacy.edge_point(
            long_first,
            long_second,
            cursor,
        )
        long_end = legacy.edge_point(
            long_first,
            long_second,
            next_cursor,
        )
        short_first, short_second = edge_vertices(
            pieces[edge.piece_id],
            edge,
        )
        pairs.extend(
            [
                (
                    family.long_edge.piece_id,
                    long_start,
                    edge.piece_id,
                    short_second,
                ),
                (
                    family.long_edge.piece_id,
                    long_end,
                    edge.piece_id,
                    short_first,
                ),
            ]
        )
        cursor = next_cursor
    return pairs


def all_point_pairs(
    pieces: list[np.ndarray],
    topology: tuple[SeamFamily, ...],
    orders: tuple[tuple[EdgeRef, ...], ...],
    pair_cache: dict[tuple, tuple] | None = None,
) -> list[tuple[int, np.ndarray, int, np.ndarray]]:
    pairs = []
    for family, order in zip(topology, orders):
        cache_key = (family.signature, order)
        cached = (
            None
            if pair_cache is None
            else pair_cache.get(cache_key)
        )
        if cached is None:
            cached = tuple(
                seam_point_pairs(pieces, family, order)
            )
            if pair_cache is not None:
                pair_cache[cache_key] = cached
        pairs.extend(cached)
    return pairs


def rigid_pose_from_correspondence(
    local_first: np.ndarray,
    local_second: np.ndarray,
    global_first: np.ndarray,
    global_second: np.ndarray,
) -> np.ndarray:
    local_vector = local_second - local_first
    global_vector = global_second - global_first
    angle = math.atan2(
        global_vector[1],
        global_vector[0],
    ) - math.atan2(local_vector[1], local_vector[0])
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    local_midpoint = (local_first + local_second) * 0.5
    global_midpoint = (global_first + global_second) * 0.5
    translation = global_midpoint - rotation @ local_midpoint
    return np.asarray(
        [angle, translation[0], translation[1]],
        dtype=np.float64,
    )


def initial_poses(
    pieces: list[np.ndarray],
    point_pairs: list[
        tuple[int, np.ndarray, int, np.ndarray]
    ],
    anchor_id: int,
) -> np.ndarray | None:
    poses = np.full((len(pieces), 3), np.nan, dtype=np.float64)
    poses[anchor_id] = np.zeros(3, dtype=np.float64)
    relations: dict[
        tuple[int, int],
        list[tuple[np.ndarray, np.ndarray]],
    ] = {}
    for first_id, first_point, second_id, second_point in point_pairs:
        relations.setdefault((first_id, second_id), []).append(
            (first_point, second_point)
        )
        relations.setdefault((second_id, first_id), []).append(
            (second_point, first_point)
        )

    changed = True
    while changed:
        changed = False
        for (known_id, unknown_id), pairs in relations.items():
            if (
                np.isnan(poses[known_id, 0])
                or not np.isnan(poses[unknown_id, 0])
                or len(pairs) < 2
            ):
                continue
            known_first = pairs[0][0]
            known_second = pairs[-1][0]
            unknown_first = pairs[0][1]
            unknown_second = pairs[-1][1]
            unknown_dx = unknown_second[0] - unknown_first[0]
            unknown_dy = unknown_second[1] - unknown_first[1]
            if unknown_dx * unknown_dx + unknown_dy * unknown_dy < 1e-10:
                continue
            known_pose = poses[known_id]
            known_cosine = math.cos(float(known_pose[0]))
            known_sine = math.sin(float(known_pose[0]))
            global_first_x = (
                known_cosine * known_first[0]
                - known_sine * known_first[1]
                + known_pose[1]
            )
            global_first_y = (
                known_sine * known_first[0]
                + known_cosine * known_first[1]
                + known_pose[2]
            )
            global_second_x = (
                known_cosine * known_second[0]
                - known_sine * known_second[1]
                + known_pose[1]
            )
            global_second_y = (
                known_sine * known_second[0]
                + known_cosine * known_second[1]
                + known_pose[2]
            )
            angle = math.atan2(
                global_second_y - global_first_y,
                global_second_x - global_first_x,
            ) - math.atan2(unknown_dy, unknown_dx)
            cosine = math.cos(angle)
            sine = math.sin(angle)
            local_mid_x = (
                unknown_first[0] + unknown_second[0]
            ) * 0.5
            local_mid_y = (
                unknown_first[1] + unknown_second[1]
            ) * 0.5
            global_mid_x = (
                global_first_x + global_second_x
            ) * 0.5
            global_mid_y = (
                global_first_y + global_second_y
            ) * 0.5
            poses[unknown_id] = np.asarray(
                [
                    angle,
                    global_mid_x
                    - (cosine * local_mid_x - sine * local_mid_y),
                    global_mid_y
                    - (sine * local_mid_x + cosine * local_mid_y),
                ],
                dtype=np.float64,
            )
            changed = True
    if np.any(np.isnan(poses)):
        return None
    return poses


def pose_residuals(
    poses: np.ndarray,
    point_pairs: list[
        tuple[int, np.ndarray, int, np.ndarray]
    ],
) -> np.ndarray:
    residuals = np.empty(
        len(point_pairs) * 2,
        dtype=np.float64,
    )
    cosines = np.cos(poses[:, 0])
    sines = np.sin(poses[:, 0])
    for pair_id, (
        first_id,
        first_point,
        second_id,
        second_point,
    ) in enumerate(point_pairs):
        first_x = (
            cosines[first_id] * first_point[0]
            - sines[first_id] * first_point[1]
            + poses[first_id, 1]
        )
        first_y = (
            sines[first_id] * first_point[0]
            + cosines[first_id] * first_point[1]
            + poses[first_id, 2]
        )
        second_x = (
            cosines[second_id] * second_point[0]
            - sines[second_id] * second_point[1]
            + poses[second_id, 1]
        )
        second_y = (
            sines[second_id] * second_point[0]
            + cosines[second_id] * second_point[1]
            + poses[second_id, 2]
        )
        residuals[pair_id * 2] = first_x - second_x
        residuals[pair_id * 2 + 1] = first_y - second_y
    return residuals


def optimize_poses(
    initial: np.ndarray,
    point_pairs: list[
        tuple[int, np.ndarray, int, np.ndarray]
    ],
    anchor_id: int,
) -> tuple[np.ndarray, float]:
    variable_ids = [
        piece_id
        for piece_id in range(len(initial))
        if piece_id != anchor_id
    ]
    poses = initial.copy()

    def pack(current: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [current[piece_id] for piece_id in variable_ids]
        )

    def unpack(values: np.ndarray) -> np.ndarray:
        current = poses.copy()
        for index, piece_id in enumerate(variable_ids):
            current[piece_id] = values[index * 3:index * 3 + 3]
        current[anchor_id] = initial[anchor_id]
        return current

    values = pack(poses)
    damping = POSE_DAMPING
    for _ in range(POSE_OPTIMIZATION_ITERATIONS):
        current = unpack(values)
        residual = pose_residuals(current, point_pairs)
        pair_norms = np.linalg.norm(
            residual.reshape(-1, 2),
            axis=1,
        )
        pair_weights = np.sqrt(
            np.minimum(
                1.0,
                HUBER_DELTA_CM / np.maximum(pair_norms, 1e-9),
            )
        )
        weights = np.repeat(pair_weights, 2)
        weighted_residual = residual * weights

        jacobian = np.zeros(
            (len(residual), len(values)),
            dtype=np.float64,
        )
        for variable_index in range(len(values)):
            epsilon = (
                1e-4
                if variable_index % 3 == 0
                else 2e-4
            )
            perturbed = values.copy()
            perturbed[variable_index] += epsilon
            perturbed_residual = pose_residuals(
                unpack(perturbed),
                point_pairs,
            )
            jacobian[:, variable_index] = (
                perturbed_residual - residual
            ) / epsilon
        weighted_jacobian = jacobian * weights[:, None]
        normal = (
            weighted_jacobian.T @ weighted_jacobian
            + np.eye(len(values)) * damping
        )
        gradient = weighted_jacobian.T @ weighted_residual
        try:
            step = -np.linalg.solve(normal, gradient)
        except np.linalg.LinAlgError:
            break
        for variable_index in range(0, len(step), 3):
            step[variable_index] = np.clip(
                step[variable_index],
                -math.radians(7.0),
                math.radians(7.0),
            )
            step[variable_index + 1:variable_index + 3] = np.clip(
                step[variable_index + 1:variable_index + 3],
                -0.65,
                0.65,
            )
        candidate_values = values + step
        candidate_residual = pose_residuals(
            unpack(candidate_values),
            point_pairs,
        )
        if np.mean(candidate_residual ** 2) <= np.mean(
            residual ** 2
        ):
            values = candidate_values
            damping = max(1e-6, damping * 0.55)
        else:
            damping = min(10.0, damping * 4.0)
        if np.linalg.norm(step) < 1e-5:
            break

    result = unpack(values)
    residual = pose_residuals(result, point_pairs)
    seam_rms = math.sqrt(float(np.mean(residual ** 2)))
    return result, seam_rms


def build_placements(
    original_pieces: list[np.ndarray],
    centered_pieces: list[np.ndarray],
    centers: list[np.ndarray],
    poses: np.ndarray,
) -> list[Placement]:
    placements = []
    for piece_id, (original, centered, center, pose) in enumerate(
        zip(original_pieces, centered_pieces, centers, poses)
    ):
        angle, tx, ty = pose
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        rotation = np.asarray(
            [[cosine, -sine], [sine, cosine]],
            dtype=np.float64,
        )
        center_target = np.asarray([tx, ty], dtype=np.float64)
        translation = center_target - rotation @ center
        transformed = centered @ rotation.T + center_target
        coverage = legacy.empty_edge_coverage(original)
        placements.append(
            Placement(
                piece_id=piece_id,
                vertices=transformed,
                rotation=rotation,
                translation=translation,
                used_edges=frozenset(),
                edge_coverage=coverage,
            )
        )
    return placements


def card_layout_variants(
    placements: list[Placement],
    texture_context: texture_matcher.TextureContext | None,
) -> list[tuple[str, list[Placement]]]:
    """Also test exchanging the two card-width columns as rigid groups.

    Four-piece card cuts commonly produce two already coherent half-card
    columns.  Edge-topology enumeration can lock those columns to the wrong
    left/right side because both geometric arrangements have the same shape.
    The exchanged layout is a translation-only alternative, not a mirror:
    every piece keeps its rotation and face orientation.
    """
    variants = [("topology", placements)]
    if not white_card_mode(texture_context) or len(placements) != 4:
        return variants
    points = np.concatenate(
        [np.asarray(item.vertices, dtype=np.float64) for item in placements],
        axis=0,
    ).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(points)).astype(np.float64)
    first_axis = box[1] - box[0]
    second_axis = box[3] - box[0]
    # The two half-card columns are separated along the card's long axis
    # when the completed card is viewed in landscape orientation.
    if np.linalg.norm(first_axis) >= np.linalg.norm(second_axis):
        swap_axis = first_axis
    else:
        swap_axis = second_axis
    axis_length = float(np.linalg.norm(swap_axis))
    if axis_length < 1e-8:
        return variants
    axis = swap_axis / axis_length
    projections = []
    for item in placements:
        center = np.mean(
            np.asarray(item.vertices, dtype=np.float64),
            axis=0,
        )
        projections.append(float(np.dot(center, axis)))
    order = np.argsort(projections)
    low_ids = set(int(index) for index in order[:2])
    high_ids = set(int(index) for index in order[2:])
    low_points = np.concatenate(
        [
            np.asarray(placements[index].vertices, dtype=np.float64)
            for index in sorted(low_ids)
        ],
        axis=0,
    )
    high_points = np.concatenate(
        [
            np.asarray(placements[index].vertices, dtype=np.float64)
            for index in sorted(high_ids)
        ],
        axis=0,
    )
    low_projection = low_points @ axis
    high_projection = high_points @ axis
    low_min = float(np.min(low_projection))
    low_max = float(np.max(low_projection))
    high_min = float(np.min(high_projection))
    high_max = float(np.max(high_projection))
    # Only exchange two clearly separated half-card columns.  This keeps the
    # extra branch bounded and avoids inventing arbitrary translations.
    if high_min < low_max - 0.35:
        return variants
    low_delta = (high_max - low_max) * axis
    high_delta = (low_min - high_min) * axis
    swapped = []
    for index, item in enumerate(placements):
        delta = low_delta if index in low_ids else high_delta
        swapped.append(
            Placement(
                piece_id=item.piece_id,
                vertices=np.asarray(item.vertices) + delta,
                rotation=np.asarray(item.rotation).copy(),
                translation=np.asarray(item.translation) + delta,
                used_edges=item.used_edges,
                edge_coverage=item.edge_coverage,
            )
        )
    swapped_iou, swapped_overlap, swapped_width, swapped_height = (
        layout_metrics(swapped)
    )
    if (
        legal_dimensions(
            swapped_width,
            swapped_height,
            texture_context,
        )
        and swapped_iou >= MIN_RECTANGLE_IOU
        and swapped_overlap <= MAX_OVERLAP_RATIO
    ):
        variants.append(("column_swap", swapped))
    return variants


def layout_metrics(
    placements: list[Placement],
) -> tuple[float, float, float, float]:
    _, width, height, rectangle_area = legacy.minimum_rectangle(
        placements
    )
    if rectangle_area <= 1e-9:
        return 0.0, float("inf"), width, height
    total_area = sum(
        legacy.polygon_area(placement.vertices)
        for placement in placements
    )
    overlap_area = 0.0
    for first_index, first in enumerate(placements):
        for second in placements[first_index + 1:]:
            overlap_area += legacy.overlap_area_raster(
                first.vertices,
                second.vertices,
            )
    union_area_estimate = max(0.0, total_area - overlap_area)
    iou = min(1.0, union_area_estimate / rectangle_area)
    overlap_ratio = overlap_area / max(total_area, 1e-9)
    return iou, overlap_ratio, width, height


def white_card_mode(
    texture_context: texture_matcher.TextureContext | None,
) -> bool:
    return bool(
        texture_context is not None
        and texture_context.white_card_confidence
        >= WHITE_CARD_MODE_CONFIDENCE
    )


def legal_dimensions(
    width: float,
    height: float,
    texture_context: texture_matcher.TextureContext | None = None,
) -> bool:
    short_side, long_side = sorted((width, height))
    if not (
        TARGET_MIN_SHORT_CM <= short_side <= TARGET_MAX_SHORT_CM
        and TARGET_MIN_LONG_CM <= long_side <= TARGET_MAX_LONG_CM
    ):
        return False
    if white_card_mode(texture_context):
        aspect_ratio = long_side / max(short_side, 1e-9)
        return (
            WHITE_CARD_MIN_ASPECT_RATIO
            <= aspect_ratio
            <= WHITE_CARD_MAX_ASPECT_RATIO
        )
    return True


def topology_orders(
    topology: tuple[SeamFamily, ...],
) -> list[tuple[tuple[EdgeRef, ...], ...]]:
    choices = []
    for family in topology:
        if len(family.short_edges) == 1:
            choices.append([family.short_edges])
        else:
            choices.append(
                list(itertools.permutations(family.short_edges))
            )
    return list(itertools.product(*choices))


def topology_signature(
    topology: tuple[SeamFamily, ...],
) -> tuple:
    return tuple(sorted(family.signature for family in topology))


def rough_rank_topologies(
    original: list[np.ndarray],
    centered: list[np.ndarray],
    centers: list[np.ndarray],
    topologies: list[tuple[SeamFamily, ...]],
    anchor_id: int,
    deadline: float | None,
    limit: int,
    exact_overlap: bool,
    order_hints: dict[tuple, tuple] | None = None,
    pair_cache: dict[tuple, tuple] | None = None,
    ranking_cache: dict[tuple, tuple] | None = None,
    texture_context: texture_matcher.TextureContext | None = None,
    source_texture_cache: dict | None = None,
) -> list[tuple[SeamFamily, ...]]:
    """Use unoptimized rigid propagation to rescue noisy valid topologies.

    Length error alone is a poor ranking signal after perspective distortion.
    A correct topology already looks much more rectangular after one pose-graph
    propagation, so this inexpensive stage ranks it before Gauss-Newton.
    """
    ranked = []
    total_piece_area = sum(
        legacy.polygon_area(piece) for piece in original
    )
    for topology in topologies:
        if deadline is not None and pytime.monotonic() >= deadline:
            break
        signature = topology_signature(topology)
        cache_key = (exact_overlap, signature)
        cached = (
            None
            if ranking_cache is None
            else ranking_cache.get(cache_key)
        )
        if cached is not None:
            best_score, best_orders = cached
            ranked.append((best_score, topology, best_orders))
            continue
        best_score = float("inf")
        best_orders = None
        for orders in topology_orders(topology):
            point_pairs = all_point_pairs(
                centered,
                topology,
                orders,
                pair_cache,
            )
            initial = initial_poses(
                centered,
                point_pairs,
                anchor_id,
            )
            if initial is None:
                continue
            placements = build_placements(
                original,
                centered,
                centers,
                initial,
            )
            if exact_overlap:
                rough_fill, overlap_ratio, width, height = (
                    layout_metrics(placements)
                )
            else:
                _, width, height, rectangle_area = (
                    legacy.minimum_rectangle(placements)
                )
                # Do not calculate pair intersections in this coarse stage.
                rough_fill = min(
                    1.0,
                    total_piece_area
                    / max(rectangle_area, 1e-9),
                )
                overlap_ratio = 0.0
            seam_residual = pose_residuals(initial, point_pairs)
            seam_rms = math.sqrt(
                float(np.mean(seam_residual ** 2))
            )
            dimension_penalty = 0.0 if legal_dimensions(
                width,
                height,
                texture_context,
            ) else 0.25
            score = (
                (1.0 - rough_fill)
                + (1.4 * overlap_ratio if exact_overlap else 0.0)
                + 0.035 * min(seam_rms, 3.0)
                + dimension_penalty
            )
            source_texture = texture_matcher.score_source_topology(
                texture_context,
                original,
                topology,
                orders,
                source_texture_cache,
            )
            score += (
                ROUGH_TEXTURE_RANK_WEIGHT
                * float(source_texture["score"])
                * float(source_texture["confidence"])
            )
            if score < best_score:
                best_score = score
                best_orders = orders
        if math.isfinite(best_score):
            if ranking_cache is not None:
                ranking_cache[cache_key] = (
                    best_score,
                    best_orders,
                )
            ranked.append((best_score, topology, best_orders))
    ranked.sort(key=lambda item: item[0])
    selected = ranked[:limit]
    if order_hints is not None:
        for _, topology, best_orders in selected:
            if best_orders is not None:
                order_hints[
                    topology_signature(topology)
                ] = best_orders
    return [topology for _, topology, _ in selected]


def _solve_geometry_once(
    pieces: list[np.ndarray],
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    allow_legacy_fallback: bool = True,
    texture_context: texture_matcher.TextureContext | None = None,
    fast_only: bool = False,
) -> tuple[Solution | None, int]:
    """Solve 1--4 randomly rotated pieces using global seam constraints."""
    global _LAST_DIAGNOSTICS
    _LAST_DIAGNOSTICS = {}
    if not 1 <= len(pieces) <= 4:
        return None, 0
    if len(pieces) == 1:
        return legacy.solve_geometry(pieces, max_seconds=max_seconds)

    started = pytime.monotonic()
    deadline = (
        started + max_seconds
        if max_seconds is not None and max_seconds > 0
        else None
    )
    if deadline is None:
        enumeration_deadline = None
        coarse_deadline = None
        overlap_deadline = None
    else:
        budget = deadline - started
        # Reserve time for every stage.  Previously topology enumeration could
        # consume the complete budget, leaving zero optimized candidates.
        enumeration_deadline = min(
            deadline,
            started + 0.16 * budget,
        )
        coarse_deadline = min(
            deadline,
            started + 0.70 * budget,
        )
        overlap_deadline = min(
            deadline,
            started + 0.86 * budget,
        )
    original = [
        legacy.counter_clockwise(
            np.asarray(piece, dtype=np.float64)
        )
        for piece in pieces
    ]
    texture_matcher.prepare_context_for_pieces(
        texture_context,
        original,
    )
    centers = [legacy.polygon_centroid(piece) for piece in original]
    centered = [
        piece - center
        for piece, center in zip(original, centers)
    ]
    families, _ = generate_seam_families(centered)
    pair_cache: dict[tuple, tuple] = {}
    for family in families:
        for order in itertools.permutations(family.short_edges):
            pair_cache[(family.signature, order)] = tuple(
                seam_point_pairs(centered, family, order)
            )
    families_finished = pytime.monotonic()
    full_topologies, strict_topology_count = enumerate_topologies(
        centered,
        families,
        deadline=enumeration_deadline,
        return_strict_count=True,
    )
    enumeration_rank_by_signature = {
        topology_signature(topology): rank
        for rank, topology in enumerate(full_topologies)
    }
    enumeration_finished = pytime.monotonic()
    anchor_id = max(
        range(len(centered)),
        key=lambda piece_id: legacy.polygon_area(centered[piece_id]),
    )
    enumerated_topology_count = len(full_topologies)
    relaxed_topologies = full_topologies[strict_topology_count:]
    fast_topologies = (
        full_topologies[
            :min(strict_topology_count, FAST_STRICT_TOPOLOGIES)
        ]
        + relaxed_topologies
    )
    ranking_cache: dict[tuple, tuple] = {}
    source_texture_cache: dict = {}
    coarse_seconds = 0.0
    overlap_seconds = 0.0
    pose_seconds = 0.0
    coarse_deadline_hit = False
    overlap_deadline_hit = False
    evaluated_total = 0
    candidates: list[LayoutCandidate] = []
    optimized_signatures = set()
    topologies: list[tuple[SeamFamily, ...]] = []
    early_accepted = False
    fast_path_accepted = False
    early_accept_reason = None

    def recommended_candidate_is_decisive(
        batch_candidates: list[LayoutCandidate],
    ) -> bool:
        nonlocal early_accept_reason
        if len(batch_candidates) < 2:
            return False
        recommended = sorted(
            batch_candidates,
            key=lambda candidate: candidate.score,
        )
        recommended_distinct = []
        recommended_signatures = set()
        for candidate in recommended:
            if (
                candidate.topology_signature
                in recommended_signatures
            ):
                continue
            recommended_signatures.add(
                candidate.topology_signature
            )
            recommended_distinct.append(candidate)
        if len(recommended_distinct) < 2:
            return False
        recommended_best = recommended_distinct[0]
        recommended_second = recommended_distinct[1]
        geometry_decisive = (
            recommended_best.iou >= EARLY_ACCEPT_IOU
            and recommended_best.overlap_ratio
            <= EARLY_ACCEPT_MAX_OVERLAP_RATIO
            and recommended_best.seam_rms_cm
            <= EARLY_ACCEPT_MAX_SEAM_RMS_CM
            and (
                recommended_second.score
                - recommended_best.score
                >= EARLY_ACCEPT_SCORE_MARGIN
            )
        )
        if geometry_decisive:
            early_accept_reason = "geometry_margin"
            return True
        if not white_card_mode(texture_context):
            return False
        texture_margin = (
            recommended_second.texture_score
            - recommended_best.texture_score
        )
        score_margin = (
            recommended_second.score
            - recommended_best.score
        )
        card_decisive = (
            recommended_best.iou >= CARD_EARLY_ACCEPT_IOU
            and recommended_best.overlap_ratio
            <= CARD_EARLY_ACCEPT_MAX_OVERLAP_RATIO
            and recommended_best.seam_rms_cm
            <= CARD_EARLY_ACCEPT_MAX_SEAM_RMS_CM
            and recommended_best.texture_confidence
            >= CARD_EARLY_ACCEPT_MIN_TEXTURE_CONFIDENCE
            and recommended_best.layout_perimeter_confidence
            >= CARD_EARLY_ACCEPT_MIN_PERIMETER_CONFIDENCE
            and recommended_best.layout_contact_segments
            >= CARD_EARLY_ACCEPT_MIN_CONTACT_SEGMENTS
            and texture_margin
            >= CARD_EARLY_ACCEPT_MIN_TEXTURE_MARGIN
            and score_margin
            >= CARD_EARLY_ACCEPT_MIN_SCORE_MARGIN
        )
        if card_decisive:
            early_accept_reason = "card_quality_gate"
            return True
        return False

    def evaluate_topology_batch(
        ranked_topologies: list[tuple[SeamFamily, ...]],
        order_hints: dict[tuple, tuple],
    ) -> tuple[list[LayoutCandidate], int, bool]:
        batch_candidates: list[LayoutCandidate] = []
        batch_evaluated = 0
        evaluation_jobs = []
        # First evaluate the best order already identified by overlap ranking
        # for every topology. Only then spend time on alternative directions.
        for topology in ranked_topologies:
            hint = order_hints.get(topology_signature(topology))
            if hint is not None:
                evaluation_jobs.append((topology, hint))
        recommended_job_count = len(evaluation_jobs)
        for topology in ranked_topologies:
            hint = order_hints.get(topology_signature(topology))
            for orders in topology_orders(topology):
                if orders != hint:
                    evaluation_jobs.append((topology, orders))

        for job_id, (topology, orders) in enumerate(evaluation_jobs):
            if (
                job_id == recommended_job_count
                and recommended_candidate_is_decisive(
                    batch_candidates
                )
            ):
                return batch_candidates, batch_evaluated, True
            if (
                deadline is not None
                and pytime.monotonic() >= deadline
            ):
                break
            point_pairs = all_point_pairs(
                centered,
                topology,
                orders,
                pair_cache,
            )
            initial = initial_poses(
                centered,
                point_pairs,
                anchor_id,
            )
            if initial is None:
                continue
            poses, seam_rms = optimize_poses(
                initial,
                point_pairs,
                anchor_id,
            )
            placements = build_placements(
                original,
                centered,
                centers,
                poses,
            )
            batch_evaluated += 1
            source_texture_result = (
                texture_matcher.score_source_topology(
                    texture_context,
                    original,
                    topology,
                    orders,
                    source_texture_cache,
                )
            )
            topology_source_texture_score = float(
                source_texture_result["score"]
            )
            source_seam_score = float(
                source_texture_result.get(
                    "seam_score",
                    topology_source_texture_score,
                )
            )
            source_perimeter_score = float(
                source_texture_result.get(
                    "perimeter_score",
                    topology_source_texture_score,
                )
            )
            for layout_variant, variant_placements in (
                card_layout_variants(
                    placements,
                    texture_context,
                )
            ):
                iou, overlap_ratio, width, height = layout_metrics(
                    variant_placements
                )
                if not legal_dimensions(
                    width,
                    height,
                    texture_context,
                ):
                    continue
                texture_result = texture_matcher.score_layout(
                    texture_context,
                    original,
                    variant_placements,
                    topology,
                    orders,
                )
                layout_texture_score = float(
                    texture_result["score"]
                )
                layout_seam_score = float(
                    texture_result.get(
                        "seam_score",
                        layout_texture_score,
                    )
                )
                layout_symmetry_score = float(
                    texture_result.get(
                        "symmetry_score",
                        layout_texture_score,
                    )
                )
                layout_perimeter_score = float(
                    texture_result.get(
                        "perimeter_score",
                        layout_texture_score,
                    )
                )
                layout_perimeter_confidence = float(
                    texture_result.get(
                        "perimeter_confidence",
                        0.0,
                    )
                )
                layout_contact_segments = int(
                    texture_result.get("contact_segments", 0)
                )
                if layout_variant == "topology":
                    source_texture_score = (
                        topology_source_texture_score
                    )
                    texture_score = (
                        SOURCE_TEXTURE_BLEND
                        * source_texture_score
                        + (1.0 - SOURCE_TEXTURE_BLEND)
                        * layout_texture_score
                    )
                    texture_confidence = float(
                        max(
                            texture_result["confidence"],
                            source_texture_result["confidence"],
                        )
                    )
                else:
                    # The exchanged columns have new physical contacts; the
                    # old topology source score is not evidence for them.
                    source_texture_score = layout_texture_score
                    texture_score = layout_texture_score
                    texture_confidence = float(
                        texture_result["confidence"]
                    )
                score = (
                    (1.0 - iou)
                    + 1.7 * overlap_ratio
                    + 0.12 * min(seam_rms, 2.0)
                    + TEXTURE_SCORE_WEIGHT
                    * texture_score
                    * texture_confidence
                )
                solution = Solution(
                    placements=variant_placements,
                    rectangularity=iou,
                    width_cm=width,
                    height_cm=height,
                    search_nodes=batch_evaluated,
                )
                batch_candidates.append(
                    LayoutCandidate(
                        solution=solution,
                        score=score,
                        iou=iou,
                        overlap_ratio=overlap_ratio,
                        seam_rms_cm=seam_rms,
                        texture_score=texture_score,
                        texture_confidence=texture_confidence,
                        source_texture_score=source_texture_score,
                        layout_texture_score=layout_texture_score,
                        source_seam_score=source_seam_score,
                        source_perimeter_score=source_perimeter_score,
                        layout_seam_score=layout_seam_score,
                        layout_symmetry_score=layout_symmetry_score,
                        layout_perimeter_score=layout_perimeter_score,
                        layout_perimeter_confidence=(
                            layout_perimeter_confidence
                        ),
                        layout_contact_segments=layout_contact_segments,
                        layout_variant=layout_variant,
                        topology_signature=topology_signature(
                            topology
                        ),
                    )
                )
        return batch_candidates, batch_evaluated, False

    search_passes = [
        ("fast", fast_topologies, FAST_COARSE_TOPOLOGIES),
    ]
    if not fast_only:
        search_passes.append(
            ("full", full_topologies, MAX_COARSE_TOPOLOGIES)
        )
    for pass_name, topology_pool, coarse_limit in search_passes:
        stage_started = pytime.monotonic()
        coarse_topologies = rough_rank_topologies(
            original,
            centered,
            centers,
            topology_pool,
            anchor_id,
            coarse_deadline,
            coarse_limit,
            False,
            pair_cache=pair_cache,
            ranking_cache=ranking_cache,
            texture_context=texture_context,
            source_texture_cache=source_texture_cache,
        )
        coarse_seconds += pytime.monotonic() - stage_started
        coarse_deadline_hit = bool(
            coarse_deadline is not None
            and pytime.monotonic() >= coarse_deadline
        )

        order_hints: dict[tuple, tuple] = {}
        stage_started = pytime.monotonic()
        ranked_topologies = rough_rank_topologies(
            original,
            centered,
            centers,
            coarse_topologies,
            anchor_id,
            overlap_deadline,
            MAX_TOPOLOGIES_TO_OPTIMIZE,
            True,
            order_hints,
            pair_cache,
            ranking_cache,
            texture_context,
            source_texture_cache,
        )
        overlap_seconds += pytime.monotonic() - stage_started
        overlap_deadline_hit = bool(
            overlap_deadline is not None
            and pytime.monotonic() >= overlap_deadline
        )

        stage_started = pytime.monotonic()
        (
            batch_candidates,
            batch_evaluated,
            batch_early_accepted,
        ) = evaluate_topology_batch(
            ranked_topologies,
            order_hints,
        )
        pose_seconds += pytime.monotonic() - stage_started
        evaluated_total += batch_evaluated
        # Keep fast-pass candidates.  The full pass can legitimately reach
        # its deadline before optimizing anything; replacing the list here
        # used to turn a valid fast candidate into ``best = None``.
        candidates.extend(batch_candidates)
        optimized_signatures.update(
            topology_signature(topology)
            for topology in ranked_topologies
        )
        topologies = ranked_topologies
        early_accepted = batch_early_accepted
        if pass_name == "fast" and batch_early_accepted:
            fast_path_accepted = True
            break
        if pass_name == "full":
            break

    candidates.sort(key=lambda candidate: candidate.score)
    distinct: list[LayoutCandidate] = []
    seen_topologies = set()
    for candidate in candidates:
        if candidate.topology_signature in seen_topologies:
            continue
        seen_topologies.add(candidate.topology_signature)
        distinct.append(candidate)

    best = distinct[0] if distinct else None
    second = distinct[1] if len(distinct) > 1 else None
    elapsed = pytime.monotonic() - started
    _LAST_DIAGNOSTICS = {
        "version": "v4.2-quality-stop",
        "seam_family_count": len(families),
        "enumerated_topology_count": enumerated_topology_count,
        "optimized_topology_count": len(optimized_signatures),
        "evaluated_layouts": evaluated_total,
        "search_mode": (
            "fast"
            if fast_path_accepted
            else ("fast_only" if fast_only else "full")
        ),
        "fast_pool_topology_count": len(fast_topologies),
        "early_accepted": early_accepted,
        "early_accept_reason": early_accept_reason,
        "elapsed_seconds": round(elapsed, 4),
        "stage_seconds": {
            "seam_families": round(
                families_finished - started,
                4,
            ),
            "topology_enumeration": round(
                enumeration_finished - families_finished,
                4,
            ),
            "coarse_ranking": round(
                coarse_seconds,
                4,
            ),
            "overlap_ranking": round(
                overlap_seconds,
                4,
            ),
            "pose_optimization": round(
                pose_seconds,
                4,
            ),
        },
        "stage_deadline_hit": {
            "topology_enumeration": bool(
                enumeration_deadline is not None
                and enumeration_finished >= enumeration_deadline
            ),
            "coarse_ranking": coarse_deadline_hit,
            "overlap_ranking": overlap_deadline_hit,
        },
        "timed_out": (
            deadline is not None and pytime.monotonic() >= deadline
        ),
        "best_score": None if best is None else round(best.score, 6),
        "best_iou": None if best is None else round(best.iou, 6),
        "best_overlap_ratio": (
            None
            if best is None
            else round(best.overlap_ratio, 6)
        ),
        "best_seam_rms_cm": (
            None
            if best is None
            else round(best.seam_rms_cm, 6)
        ),
        "best_texture_score": (
            None
            if best is None
            else round(best.texture_score, 6)
        ),
        "best_texture_confidence": (
            None
            if best is None
            else round(best.texture_confidence, 6)
        ),
        "best_source_texture_score": (
            None
            if best is None
            else round(best.source_texture_score, 6)
        ),
        "best_layout_texture_score": (
            None
            if best is None
            else round(best.layout_texture_score, 6)
        ),
        "best_source_seam_score": (
            None
            if best is None
            else round(best.source_seam_score, 6)
        ),
        "best_source_perimeter_score": (
            None
            if best is None
            else round(best.source_perimeter_score, 6)
        ),
        "best_layout_seam_score": (
            None
            if best is None
            else round(best.layout_seam_score, 6)
        ),
        "best_layout_symmetry_score": (
            None
            if best is None
            else round(best.layout_symmetry_score, 6)
        ),
        "best_layout_perimeter_score": (
            None
            if best is None
            else round(best.layout_perimeter_score, 6)
        ),
        "best_layout_perimeter_confidence": (
            None
            if best is None
            else round(best.layout_perimeter_confidence, 6)
        ),
        "best_layout_variant": (
            None if best is None else best.layout_variant
        ),
        "best_layout_contact_segments": (
            None
            if best is None
            else best.layout_contact_segments
        ),
        "white_card_confidence": (
            0.0
            if texture_context is None
            else round(
                texture_context.white_card_confidence,
                6,
            )
        ),
        "best_enumeration_rank": (
            None
            if best is None
            else enumeration_rank_by_signature.get(
                best.topology_signature
            )
        ),
        "second_score": (
            None if second is None else round(second.score, 6)
        ),
        "second_iou": (
            None if second is None else round(second.iou, 6)
        ),
        "second_texture_score": (
            None
            if second is None
            else round(second.texture_score, 6)
        ),
        "second_texture_confidence": (
            None
            if second is None
            else round(second.texture_confidence, 6)
        ),
        "second_source_texture_score": (
            None
            if second is None
            else round(second.source_texture_score, 6)
        ),
        "second_layout_texture_score": (
            None
            if second is None
            else round(second.layout_texture_score, 6)
        ),
        "second_layout_seam_score": (
            None
            if second is None
            else round(second.layout_seam_score, 6)
        ),
        "second_source_perimeter_score": (
            None
            if second is None
            else round(second.source_perimeter_score, 6)
        ),
        "second_layout_symmetry_score": (
            None
            if second is None
            else round(second.layout_symmetry_score, 6)
        ),
        "second_layout_perimeter_score": (
            None
            if second is None
            else round(second.layout_perimeter_score, 6)
        ),
        "second_layout_variant": (
            None if second is None else second.layout_variant
        ),
        "texture_score_margin": (
            None
            if best is None or second is None
            else round(
                second.texture_score - best.texture_score,
                6,
            )
        ),
        "layout_seam_score_margin": (
            None
            if best is None or second is None
            else round(
                second.layout_seam_score
                - best.layout_seam_score,
                6,
            )
        ),
        "score_margin": (
            None
            if best is None or second is None
            else round(second.score - best.score, 6)
        ),
    }

    def legacy_fallback(reason: str) -> tuple[Solution | None, int]:
        if not allow_legacy_fallback:
            _LAST_DIAGNOSTICS["fallback"] = "disabled"
            _LAST_DIAGNOSTICS["fallback_reason"] = reason
            _LAST_DIAGNOSTICS["fallback_nodes"] = 0
            return None, evaluated_total
        remaining_seconds = None
        if deadline is not None:
            remaining_seconds = max(
                0.05,
                deadline - pytime.monotonic(),
            )
        fallback_solution, fallback_nodes = legacy.solve_geometry(
            original,
            max_seconds=remaining_seconds,
        )
        finished = pytime.monotonic()
        _LAST_DIAGNOSTICS["fallback"] = "legacy_edge_dfs"
        _LAST_DIAGNOSTICS["fallback_reason"] = reason
        _LAST_DIAGNOSTICS["fallback_nodes"] = fallback_nodes
        _LAST_DIAGNOSTICS["elapsed_seconds"] = round(
            finished - started,
            4,
        )
        _LAST_DIAGNOSTICS["timed_out"] = bool(
            deadline is not None and finished >= deadline
        )
        return fallback_solution, evaluated_total + fallback_nodes

    if best is None:
        return legacy_fallback("no_global_candidate")
    if (
        best.iou < MIN_RECTANGLE_IOU
        or best.overlap_ratio > MAX_OVERLAP_RATIO
        or best.seam_rms_cm > MAX_SEAM_RMS_CM
    ):
        return legacy_fallback("global_quality_gate")
    if (
        second is not None
        and second.score - best.score < UNIQUE_SCORE_MARGIN
        and second.iou >= MIN_RECTANGLE_IOU
        and best.iou < 0.97
    ):
        score_margin = second.score - best.score
        texture_margin = (
            second.texture_score - best.texture_score
        )
        layout_seam_margin = (
            second.layout_seam_score
            - best.layout_seam_score
        )
        minimum_direct_contacts = max(
            len(original) - 1,
            DIRECT_SEAM_MIN_CONTACTS_PER_PIECE,
        )
        geometry_safe_for_texture = (
            texture_context is not None
            and best.texture_confidence >= TEXTURE_MIN_CONFIDENCE
            and second.texture_confidence >= TEXTURE_MIN_CONFIDENCE
            and best.iou >= TEXTURE_ACCEPT_IOU
            and best.overlap_ratio
            <= TEXTURE_ACCEPT_MAX_OVERLAP_RATIO
            and best.seam_rms_cm
            <= TEXTURE_ACCEPT_MAX_SEAM_RMS_CM
        )
        # Primary rule for patterned cards: compare only the pattern samples
        # crossing every actual contact in the completed rectangle.  Geometry
        # and half-turn symmetry may remain nearly tied for a top/bottom swap,
        # but the cut-line continuation cannot.
        direct_seam_resolved = (
            geometry_safe_for_texture
            # Blank white card regions can create a deceptively low direct
            # seam error.  White playing cards must instead pass the combined
            # cut-line, outer-border and 180-degree pattern score.
            and (
                texture_context is None
                or texture_context.white_card_confidence
                < WHITE_CARD_MODE_CONFIDENCE
            )
            and best.layout_contact_segments
            >= minimum_direct_contacts
            and layout_seam_margin
            >= DIRECT_SEAM_UNIQUE_MARGIN
        )
        texture_resolved = (
            geometry_safe_for_texture
            and score_margin >= TEXTURE_UNIQUE_SCORE_MARGIN
            and texture_margin >= TEXTURE_UNIQUE_RAW_MARGIN
        )
        if direct_seam_resolved:
            _LAST_DIAGNOSTICS[
                "ambiguity_resolved_by"
            ] = "direct_seam_texture"
        elif texture_resolved:
            _LAST_DIAGNOSTICS[
                "ambiguity_resolved_by"
            ] = (
                "card_pattern"
                if (
                    texture_context is not None
                    and texture_context.white_card_confidence
                    >= WHITE_CARD_MODE_CONFIDENCE
                )
                else "texture"
            )
        else:
            _LAST_DIAGNOSTICS["ambiguous"] = True
            # A geometry-only fallback can silently destroy flower alignment.
            # If the image contains strong seam texture, report uncertainty
            # instead of returning an unverified motion plan.
            if (
                texture_context is not None
                and best.texture_confidence
                >= TEXTURE_MIN_CONFIDENCE
            ):
                _LAST_DIAGNOSTICS["fallback"] = (
                    "disabled_for_textured_ambiguity"
                )
                _LAST_DIAGNOSTICS["fallback_reason"] = (
                    "ambiguous_global_candidate"
                )
                _LAST_DIAGNOSTICS["fallback_nodes"] = 0
                return None, evaluated_total
            return legacy_fallback("ambiguous_global_candidate")
    best.solution.search_nodes = evaluated_total
    return best.solution, evaluated_total


def solve_geometry(
    pieces: list[np.ndarray],
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    texture_context: texture_matcher.TextureContext | None = None,
) -> tuple[Solution | None, int]:
    """Try the proven size band first, then an emergency relaxed band.

    Mixing both bands in one bounded candidate pool allowed new small-layout
    hypotheses to evict previously solved cases.  Two passes preserve the
    original fast path while still accepting captures with moderate scale
    drift.  The total time remains bounded by ``max_seconds``.
    """
    global _LAST_DIAGNOSTICS
    global TARGET_MIN_SHORT_CM, TARGET_MAX_SHORT_CM
    global TARGET_MIN_LONG_CM, TARGET_MAX_LONG_CM

    started = pytime.monotonic()
    saved_limits = (
        TARGET_MIN_SHORT_CM,
        TARGET_MAX_SHORT_CM,
        TARGET_MIN_LONG_CM,
        TARGET_MAX_LONG_CM,
    )
    deadline = (
        None
        if max_seconds is None or max_seconds <= 0
        else started + max_seconds
    )
    normalized_pieces = [
        legacy.counter_clockwise(
            np.asarray(piece, dtype=np.float64)
        )
        for piece in pieces
    ]
    texture_matcher.prepare_context_for_pieces(
        texture_context,
        normalized_pieces,
    )

    # White playing cards have a known physical shape.  Sending them through
    # the old narrow-size pass first wastes 30% of the Maix time budget and,
    # for a slightly small A4 calibration, that pass cannot possibly succeed.
    # Run one bounded pass with the broad scale limits plus the card aspect
    # gate above.  Brown cardboard continues to use the two legacy size bands.
    if white_card_mode(texture_context):
        card_solution, card_nodes = _solve_geometry_once(
            pieces,
            max_seconds=max_seconds,
            texture_context=texture_context,
        )
        _LAST_DIAGNOSTICS["size_pass"] = "card_aspect"
        _LAST_DIAGNOSTICS["target_aspect_ratio_range"] = [
            WHITE_CARD_MIN_ASPECT_RATIO,
            WHITE_CARD_MAX_ASPECT_RATIO,
        ]
        _LAST_DIAGNOSTICS["elapsed_seconds"] = round(
            pytime.monotonic() - started,
            4,
        )
        _LAST_DIAGNOSTICS["timed_out"] = bool(
            deadline is not None
            and pytime.monotonic() >= deadline
        )
        return card_solution, card_nodes

    core_budget = (
        max_seconds * 0.30
        if max_seconds is not None and max_seconds > 0
        else max_seconds
    )
    try:
        (
            TARGET_MIN_SHORT_CM,
            TARGET_MAX_SHORT_CM,
            TARGET_MIN_LONG_CM,
            TARGET_MAX_LONG_CM,
        ) = (
            CORE_MIN_SHORT_CM,
            CORE_MAX_SHORT_CM,
            CORE_MIN_LONG_CM,
            CORE_MAX_LONG_CM,
        )
        core_solution, core_nodes = _solve_geometry_once(
            pieces,
            max_seconds=core_budget,
            allow_legacy_fallback=False,
            texture_context=texture_context,
        )
        core_diagnostics = dict(_LAST_DIAGNOSTICS)
        if core_solution is not None:
            _LAST_DIAGNOSTICS["size_pass"] = "core"
            _LAST_DIAGNOSTICS["elapsed_seconds"] = round(
                pytime.monotonic() - started,
                4,
            )
            return core_solution, core_nodes

        (
            TARGET_MIN_SHORT_CM,
            TARGET_MAX_SHORT_CM,
            TARGET_MIN_LONG_CM,
            TARGET_MAX_LONG_CM,
        ) = saved_limits
        remaining = (
            None
            if deadline is None
            else max(0.05, deadline - pytime.monotonic())
        )
        relaxed_solution, relaxed_nodes = _solve_geometry_once(
            pieces,
            max_seconds=remaining,
            texture_context=texture_context,
        )
        _LAST_DIAGNOSTICS["size_pass"] = "relaxed"
        _LAST_DIAGNOSTICS["core_attempt"] = {
            "best_iou": core_diagnostics.get("best_iou"),
            "fallback_reason": core_diagnostics.get(
                "fallback_reason"
            ),
            "evaluated_layouts": core_diagnostics.get(
                "evaluated_layouts"
            ),
        }
        _LAST_DIAGNOSTICS["elapsed_seconds"] = round(
            pytime.monotonic() - started,
            4,
        )
        _LAST_DIAGNOSTICS["timed_out"] = bool(
            deadline is not None
            and pytime.monotonic() >= deadline
        )
        return relaxed_solution, core_nodes + relaxed_nodes
    finally:
        (
            TARGET_MIN_SHORT_CM,
            TARGET_MAX_SHORT_CM,
            TARGET_MIN_LONG_CM,
            TARGET_MAX_LONG_CM,
        ) = saved_limits
