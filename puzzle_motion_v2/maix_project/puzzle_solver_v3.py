"""Global seam-topology puzzle solver for random piece rotations.

This module is intentionally separate from puzzle_solver.py while it is
validated on the PC. It uses only OpenCV and Numpy so the same file can run
on MaixCAM 2 after validation.

V3 阶段为：接缝族生成 -> 拓扑枚举 -> 初始刚体位姿 -> 连续优化 -> 候选评分。
``solve_geometry`` 是推荐入口，其余函数按阶段划分，便于定位超时和歧义。
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
MIN_EDGE_CM = 1.00

MAX_CHAIN_PARTS = 3
MAX_SEAM_FAMILIES = 64
# Two edges on each side of one seam are common in a 2x2 arrangement. Keep
# them in a separate bounded pool so they cannot displace proven 1xN matches.
MAX_CHAIN_TO_CHAIN_FAMILIES = 96
MAX_ENUMERATED_TOPOLOGIES = 700
# A valid placement can create an additional contact that is not needed to
# connect the piece graph.  Its two-family parent then has a false outer
# perimeter and fails the dimension prefilter, while the complete three-family
# topology can sit just below the generic heuristic cutoff.  Keep a bounded
# reserve for those fully declared contact topologies.
MAX_THREE_FAMILY_RESERVE_TOPOLOGIES = 128
MAX_CHAIN_TO_CHAIN_TOPOLOGIES = 192
MAX_CHAIN_TO_CHAIN_GEOMETRY_TOPOLOGIES = 64
MAX_CHAIN_TO_CHAIN_SOURCE_TOPOLOGIES = 128
MAX_TWO_BY_TWO_GRID_TOPOLOGIES = 128
MAX_TWO_BY_TWO_GRID_SOURCE_TOPOLOGIES = 96
MAX_FOUR_FAMILY_CYCLE_TOPOLOGIES = 128
MAX_FOUR_FAMILY_CYCLE_SOURCE_TOPOLOGIES = 96
MAX_RELAXED_TOPOLOGIES = 96
WHITE_CARD_GEOMETRY_TOPOLOGIES = 280
WHITE_CARD_SOURCE_POSE_TOPOLOGIES = 32
WHITE_CARD_NEIGHBOR_TOPOLOGIES = 120
WHITE_CARD_BORDER_TOPOLOGIES = 80
WHITE_CARD_NEIGHBOR_STRUCTURES = 8
# Sparse number cards have too little artwork for rich seam ranking, but their
# white outer frame is still reliable.  Fuse its ordinal rank with the
# geometry rank so neither lighting scale nor one lucky edge-length match can
# dominate topology selection.
SPARSE_CARD_BORDER_RANK_WEIGHT = 1.0
MAX_COARSE_TOPOLOGIES = 80
MAX_TOPOLOGIES_TO_OPTIMIZE = 16
MAX_CARD_GENERIC_SEARCH_TOPOLOGIES = 240
MAX_SPARSE_CARD_CHAIN_SEARCH_TOPOLOGIES = 64
# A long physical cut can be measured differently on opposite sides after
# sparse lens correction (capture 172040: 6.216 cm versus 5.636 cm).  Such a
# pair still carries far more structural information than the many short,
# similarly sized card edges, so give topologies containing it one bounded
# card-only ranking pass before the generic pool.
CARD_LONG_DIRECT_MIN_LENGTH_CM = 5.0
MAX_CARD_LONG_DIRECT_TOPOLOGIES = 192
FAST_STRICT_TOPOLOGIES = 120
FAST_COARSE_TOPOLOGIES = 32

HUBER_DELTA_CM = 0.28
POSE_OPTIMIZATION_ITERATIONS = 6
CHAIN_POSE_OPTIMIZATION_ITERATIONS = 4
POSE_DAMPING = 1e-3
# Polygon fitting perturbs area and perimeter independently.  A physically
# valid topology can therefore have a slightly negative quadratic
# discriminant before pose optimization.  Keep this pre-filter tolerant and
# leave the strict decision to the final IoU/overlap/seam quality gates.
MIN_PERIMETER_DISCRIMINANT_CM2 = -4.0
STRICT_MIN_PERIMETER_DISCRIMINANT_CM2 = -2.0
FOUR_FAMILY_CYCLE_MIN_PERIMETER_DISCRIMINANT_CM2 = -8.0
FOUR_FAMILY_CYCLE_DIRECT_ABS_TOLERANCE_CM = 0.65
FOUR_FAMILY_CYCLE_DIRECT_REL_TOLERANCE = 0.13

MIN_RECTANGLE_IOU = 0.89
MAX_OVERLAP_RATIO = 0.055
MAX_SEAM_RMS_CM = 0.50
UNIQUE_SCORE_MARGIN = 0.018
TEXTURE_SCORE_WEIGHT = 0.28
# Rich J/Q/K artwork should decide between geometrically valid rectangles.
# Sparse number cards retain the conservative generic texture weight.
RICH_CARD_TEXTURE_SCORE_WEIGHT = 0.45
SOURCE_TEXTURE_BLEND = 0.55
ROUGH_TEXTURE_RANK_WEIGHT = 0.22
# Geometry-only layouts need a deliberately conservative margin because
# several different seam topologies can produce similar rectangles.  Texture
# already supplies independent evidence, so reusing the geometry margin made
# patterned cards impossible to accept even when the seam pixels were active.
TEXTURE_UNIQUE_SCORE_MARGIN = 0.0025
# Real printed-card captures contain resampling and A4-rectification noise.
# Capture 102952 separates the user-confirmed P1/P2 half-column turn from the
# runner-up by 0.004815, so a 0.005 hard boundary rejects the correct layout
# for a numerically insignificant 0.000185.  Geometry, overlap, seam RMS and
# the mixed-score margin still have to pass independently below.
TEXTURE_UNIQUE_RAW_MARGIN = 0.004
# Column half-turn variants can make the blended texture value nearly equal
# even when every interpretable card cue agrees.  In that case require a
# substantially larger total-score lead plus consensus from the direct seam,
# half-turn pattern and exposed outer-border measurements.
CARD_COMPONENT_SCORE_MARGIN = 0.008
CARD_COMPONENT_EPSILON = 0.005
# Two directions of a nearly 180-degree-symmetric face-card print differ by
# as much as 0.075 in the combined seam/border value in real captures because
# blur and exposure affect the wide white edge strongly.  Treat that narrow
# band as an orientation tie and use a deterministic piece-group convention
# so motion plans do not alternate between equivalent column commands.
# The playing-card border is a hard visual constraint, not a soft tie-break.
# Keep only orientations whose exposed perimeter is very close to the
# whitest candidate before comparing their internal seams.
CARD_ORIENTATION_PERIMETER_DELTA = 0.025
# A standard face card is deliberately invariant under a global 180-degree
# turn.  Perimeter costs of those two target orientations should therefore
# be effectively identical; this epsilon only groups that final symmetry.
CARD_GLOBAL_HALF_TURN_PERIMETER_EPSILON = 0.005
CARD_ORIENTATION_MIN_PERIMETER_CONFIDENCE = 0.60
CARD_ORIENTATION_PERIMETER_WEIGHT = 1.00
# Once the four outer borders are white, the direct 180-degree artwork test
# is the strongest cue for deciding which half-card is upside down.
CARD_ORIENTATION_SYMMETRY_WEIGHT = 1.00
CARD_MAX_WORST_SIDE_SCORE = 0.42
CARD_MAX_WORST_EDGE_SCORE = 0.60
# The two values above describe the preferred clean white-frame quality.
# They must not be topology gates: several geometrically complete rectangles
# can exist and the card artwork is supposed to disambiguate them.  Only
# reject a candidate here when its perimeter measurement is so poor that it
# is no longer credible as a card at all.
CARD_ABSOLUTE_MAX_WORST_SIDE_SCORE = 0.75
CARD_ABSOLUTE_MAX_WORST_EDGE_SCORE = 0.90
# Mild ranking penalties keep cleaner borders ahead when pattern evidence is
# otherwise tied, without deleting a valid rectangle before seam matching.
CARD_WORST_SIDE_PENALTY_WEIGHT = 0.06
CARD_WORST_EDGE_PENALTY_WEIGHT = 0.04
# These guards do not enlarge seam-length matching or the topology search.
# They only absorb the final raster/polygon measurement jitter around the
# white-card quality gate.  Capture 20260730_171716 reconstructs the correct
# 6.17 x 9.24 cm card but measures IoU=0.91883, overlap=0.02567 and perimeter
# confidence=0.58073, narrowly straddling all three nominal boundaries.
CARD_FINAL_IOU_GUARD = 0.003
CARD_FINAL_OVERLAP_GUARD = 0.002
CARD_PERIMETER_CONFIDENCE_GUARD = 0.05
# A very small translation-only repair is cheaper and safer than weakening
# the final overlap gate.  It is used only for near-rectangular white-card
# candidates containing a measured one-to-many seam.
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
# Exchanging two already solved half-card columns can amplify millimetre-level
# contour errors at their shared boundary without changing the underlying
# topology.  Keep the global candidate gate at TEXTURE_ACCEPT_IOU, but let
# the bounded four-way orientation check reach the white-border measurement.
CARD_ORIENTATION_MIN_IOU = 0.92
CARD_FINAL_MIN_IOU = 0.92
CARD_FINAL_MAX_OVERLAP_RATIO = 0.025
TEXTURE_ACCEPT_MAX_SEAM_RMS_CM = 0.20
EARLY_ACCEPT_IOU = 0.94
EARLY_ACCEPT_MAX_OVERLAP_RATIO = 0.015
EARLY_ACCEPT_MAX_SEAM_RMS_CM = 0.20
EARLY_ACCEPT_SCORE_MARGIN = 0.06
# A clean Q1/Q2 rectangle is executable even when a batch contains only one
# distinct topology, so a runner-up margin cannot be computed. Keep this gate
# deliberately tighter than the normal final geometry limits.
Q12_ABSOLUTE_EARLY_IOU = 0.95
Q12_ABSOLUTE_EARLY_MAX_OVERLAP_RATIO = 0.005
Q12_ABSOLUTE_EARLY_MAX_SEAM_RMS_CM = 0.15
Q12_ABSOLUTE_EARLY_MIN_CONTACT_SEGMENTS = 3
# A patterned white card can be accepted earlier than a geometry-only puzzle,
# but only when several independent checks agree.  These thresholds are
# intentionally stricter than the final acceptance gate so an early exit can
# never be triggered by 180-degree symmetry alone.
CARD_EARLY_ACCEPT_IOU = 0.94
CARD_EARLY_ACCEPT_MAX_OVERLAP_RATIO = 0.02
CARD_EARLY_ACCEPT_MAX_SEAM_RMS_CM = 0.20
CARD_EARLY_ACCEPT_MIN_TEXTURE_CONFIDENCE = 0.90
CARD_EARLY_ACCEPT_MIN_PERIMETER_CONFIDENCE = 0.90
CARD_EARLY_ACCEPT_MIN_TEXTURE_MARGIN = 0.05
CARD_EARLY_ACCEPT_MIN_SCORE_MARGIN = 0.005
CARD_EARLY_ACCEPT_MIN_CONTACT_SEGMENTS = 4
CARD_SINGLE_HALF_TURN_MIN_POLYGON_IOU = 0.88
CARD_SINGLE_HALF_TURN_MAX_VARIANTS = 8

# These broad limits are used only by the playing-card mode, whose standard
# card dimensions differ from the Question 1/2 target rectangle.
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
# Four hand-cut pieces measured through two cumulative chains carry more
# perimeter bias than direct card edges. These limits remain guarded by the
# generic rectangle, overlap, seam and white-border checks.
CHAIN_CARD_MIN_ASPECT_RATIO = 1.35
CHAIN_CARD_MAX_ASPECT_RATIO = 1.85
CHAIN_CARD_MIN_IOU = 0.89
TWO_BY_TWO_GRID_MIN_IOU = 0.875
CHAIN_CARD_MAX_OVERLAP_RATIO = 0.055
CHAIN_CARD_MAX_SEAM_RMS_CM = 0.50
CHAIN_CARD_MIN_PERIMETER_CONFIDENCE = 0.55
WHITE_CARD_MODE_CONFIDENCE = 0.50
RICH_CARD_MODE_CONFIDENCE = 0.45
# Sparse number-card artwork can fall below the normal card classifier while
# still carrying substantially more ink than the blank white geometry task.
# This gate is used only after an excellent card-aspect rectangle has failed
# the Question 1/2 dimensions and passed the normal card border validation.
SPARSE_CARD_RECOVERY_MIN_CONFIDENCE = 0.15
SPARSE_CARD_RECOVERY_MIN_INK_FRACTION = 0.06
SPARSE_CARD_RECOVERY_MIN_VERY_DARK_FRACTION = 0.001
SPARSE_CARD_RECOVERY_MIN_VIVID_PRINT_FRACTION = 0.001
SPARSE_CARD_RECOVERY_MIN_WHITE_FRACTION = 0.45
SPARSE_CARD_RECOVERY_MIN_SHORT_CM = 4.80
SPARSE_CARD_RECOVERY_MAX_SHORT_CM = 7.20
SPARSE_CARD_RECOVERY_MIN_LONG_CM = 7.50
SPARSE_CARD_RECOVERY_MAX_LONG_CM = 10.50
SPARSE_CARD_RECOVERY_MIN_ASPECT_RATIO = 1.40
SPARSE_CARD_RECOVERY_MAX_ASPECT_RATIO = 1.68
MAX_CARD_SOURCE_POSE_RANK_INPUT = 192

# Questions 1 and 2 use the specified rectangle dimensions directly.  This
# excludes size-compatible false layouts before their pose search expands.
CORE_MIN_SHORT_CM = 5.00
CORE_MAX_SHORT_CM = 9.00
CORE_MIN_LONG_CM = 9.00
CORE_MAX_LONG_CM = 12.00

DEFAULT_MAX_SECONDS = 14.0
# The global topology ranker is intentionally bounded, but it used to consume
# the complete Maix deadline before invoking the proven edge-DFS fallback.
# Reserve enough wall time for DFS to examine a clean four-piece capture and
# then run the same card-border/artwork verification on its rectangle.
CARD_LEGACY_RESERVE_SECONDS = 3.0
# Compatibility names used by the MaixCAM UI.  Keeping these here lets the
# old solver remain untouched and makes switching versions a one-line import.
MAX_SEARCH_SECONDS = DEFAULT_MAX_SECONDS

_LAST_DIAGNOSTICS: dict = {}


@dataclass(frozen=True, order=True)
class EdgeRef:
    """碎片边的稳定引用，用于跨阶段传递匹配关系。"""
    piece_id: int
    edge_id: int


@dataclass(frozen=True)
class SeamFamily:
    """一组可解释为同一接缝的边匹配（含 T 形长边拆分情况）。"""
    long_edge: EdgeRef
    short_edges: tuple[EdgeRef, ...]
    length_error_cm: float
    normalized_error: float
    used_mask: int
    internal_length_cm: float
    chain_a_edges: tuple[EdgeRef, ...] = ()
    chain_b_edges: tuple[EdgeRef, ...] = ()

    @property
    def is_chain_to_chain(self) -> bool:
        return bool(self.chain_a_edges and self.chain_b_edges)

    @property
    def edges(self) -> tuple[EdgeRef, ...]:
        if self.is_chain_to_chain:
            return self.chain_a_edges + self.chain_b_edges
        return (self.long_edge,) + self.short_edges

    @property
    def pieces(self) -> frozenset[int]:
        return frozenset(edge.piece_id for edge in self.edges)

    @property
    def signature(self) -> tuple:
        if self.is_chain_to_chain:
            edges = tuple(sorted(self.edges))
            return (
                edges[0],
                edges[1:],
                self.chain_a_edges,
                self.chain_b_edges,
            )
        return (
            self.long_edge,
            tuple(sorted(self.short_edges)),
        )

    @property
    def split_count(self) -> int:
        if self.is_chain_to_chain:
            return (
                len(self.chain_a_edges)
                + len(self.chain_b_edges)
                - 2
            )
        return len(self.short_edges) - 1


@dataclass
class LayoutCandidate:
    """候选布局及其 IoU、接缝、纹理和周边评分。"""
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
    layout_perimeter_worst_side_score: float
    layout_perimeter_worst_edge_score: float
    layout_contact_segments: int
    layout_variant: str
    topology_signature: tuple
    uses_chain_to_chain: bool
    uses_two_by_two_grid: bool


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
    """按边长容差生成完整边、拆分边和链到链接缝候选。"""
    lengths, bit_index = build_edge_tables(pieces)
    edges = [
        EdgeRef(piece_id, edge_id)
        for piece_id, piece_lengths in enumerate(lengths)
        for edge_id in range(len(piece_lengths))
        if piece_lengths[edge_id] >= MIN_EDGE_CM
    ]
    families: list[SeamFamily] = []
    chain_to_chain_families: list[SeamFamily] = []

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

    # Two-piece chain to two-piece chain. This cannot be represented as four
    # direct edge pairs: the chain junctions generally occur at different
    # positions, so each edge can cover only part of an edge on the other
    # side. Restrict this family to four distinct pieces and retain ordering
    # for the pose stage rather than multiplying topology candidates here.
    if len(pieces) == 4:
        edges_by_piece = [
            [edge for edge in edges if edge.piece_id == piece_id]
            for piece_id in range(4)
        ]
        partitions = (
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            ((0, 3), (1, 2)),
        )
        for side_a_ids, side_b_ids in partitions:
            for chosen in itertools.product(*edges_by_piece):
                by_piece = {
                    edge.piece_id: edge for edge in chosen
                }
                side_a = tuple(
                    sorted(by_piece[piece_id] for piece_id in side_a_ids)
                )
                side_b = tuple(
                    sorted(by_piece[piece_id] for piece_id in side_b_ids)
                )
                if side_b < side_a:
                    side_a, side_b = side_b, side_a
                length_a = sum(
                    lengths[edge.piece_id][edge.edge_id]
                    for edge in side_a
                )
                length_b = sum(
                    lengths[edge.piece_id][edge.edge_id]
                    for edge in side_b
                )
                matches, normalized_error = tolerance_match(
                    length_a,
                    length_b,
                    CHAIN_ABS_TOLERANCE_CM,
                    CHAIN_REL_TOLERANCE,
                )
                if not matches:
                    continue
                refs = side_a + side_b
                chain_to_chain_families.append(
                    SeamFamily(
                        # Legacy fields remain populated for callers that
                        # inspect families without understanding 2x2 yet.
                        long_edge=side_a[0],
                        short_edges=side_b,
                        length_error_cm=abs(length_a - length_b),
                        normalized_error=normalized_error,
                        used_mask=edge_mask(refs, bit_index),
                        internal_length_cm=0.5 * (
                            length_a + length_b
                        ),
                        chain_a_edges=side_a,
                        chain_b_edges=side_b,
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
            + 0.08 * family.split_count,
            -family.internal_length_cm,
        ),
    )
    chain_deduplicated = {
        family.signature: family
        for family in chain_to_chain_families
    }
    chain_result = sorted(
        chain_deduplicated.values(),
        key=lambda family: (
            family.normalized_error
            + 0.08 * family.split_count,
            -family.internal_length_cm,
            family.signature,
        ),
    )
    return (
        result[:MAX_SEAM_FAMILIES]
        + chain_result[:MAX_CHAIN_TO_CHAIN_FAMILIES],
        lengths,
    )


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
        family_edges = family.edges
        root_id = family_edges[0].piece_id
        used_pieces.add(root_id)
        for edge in family_edges[1:]:
            used_pieces.add(edge.piece_id)
            union(root_id, edge.piece_id)
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
    # This pre-filter estimates dimensions from noisy cut-edge perimeters;
    # keep it tolerant and enforce the strict Q1/Q2 range after pose fitting.
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
            family.split_count for family in families
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
        for edge in family.edges:
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
    texture_context: texture_matcher.TextureContext | None = None,
    source_pieces: list[np.ndarray] | None = None,
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
        family_edges = family.edges
        root_piece = family_edges[0].piece_id
        piece_mask = 1 << root_piece
        adjacency = 0
        edge_masks = [0] * piece_count
        edge_masks[root_piece] |= 1 << family_edges[0].edge_id
        for edge in family_edges[1:]:
            other_piece = edge.piece_id
            piece_mask |= 1 << other_piece
            adjacency |= 1 << (
                root_piece * piece_count + other_piece
            )
            adjacency |= 1 << (
                other_piece * piece_count + root_piece
            )
            edge_masks[other_piece] |= 1 << edge.edge_id
        records.append(
            (
                family,
                piece_mask,
                adjacency,
                tuple(edge_masks),
                0.25 * family.normalized_error
                + 0.05 * family.split_count,
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
        min_discriminant: float = MIN_PERIMETER_DISCRIMINANT_CM2,
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
        if discriminant < min_discriminant:
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

    edge_border_costs = (
        None
        if texture_context is None
        else texture_context.edge_border_costs
    )
    use_card_border_rank = bool(
        white_card_mode(texture_context)
        and edge_border_costs
    )

    def outer_border_score(edge_masks: list[int]) -> float:
        if not use_card_border_rank:
            return 0.0
        weighted_error = 0.0
        total_length = 0.0
        for piece_id, piece in enumerate(pieces):
            used_edges = edge_masks[piece_id]
            for edge_id in range(len(piece)):
                if used_edges & (1 << edge_id):
                    continue
                result = edge_border_costs.get((piece_id, edge_id))
                if result is None:
                    continue
                edge_error, edge_length = result
                weighted_error += edge_error * edge_length
                total_length += edge_length
        if total_length <= 1e-8:
            return 1.0
        return weighted_error / total_length

    source_pose_family_cache: dict[tuple, tuple[float, int]] = {}

    def source_pose_family_score(
        family: SeamFamily,
    ) -> tuple[float, int]:
        cached = source_pose_family_cache.get(family.signature)
        if cached is not None:
            return cached
        if source_pieces is None:
            return float("inf"), 0
        family_errors = []
        if family.is_chain_to_chain:
            order_errors = []
            for side_a in itertools.permutations(
                family.chain_a_edges
            ):
                for side_b in itertools.permutations(
                    family.chain_b_edges
                ):
                    pairs = seam_point_pairs(
                        source_pieces,
                        family,
                        side_a + side_b,
                    )
                    if not pairs:
                        continue
                    order_errors.append(
                        sum(
                            float(
                                np.linalg.norm(
                                    first_point - second_point
                                )
                            )
                            for (
                                _,
                                first_point,
                                _,
                                second_point,
                            ) in pairs
                        )
                        / (
                            len(pairs)
                            * max(family.internal_length_cm, 1.0)
                        )
                    )
            if order_errors:
                family_errors.append(min(order_errors))
        else:
            long_piece = source_pieces[
                family.long_edge.piece_id
            ]
            long_first = long_piece[family.long_edge.edge_id]
            long_second = long_piece[
                (family.long_edge.edge_id + 1) % len(long_piece)
            ]
            long_vector = long_second - long_first
            long_length = float(np.linalg.norm(long_vector))
            if long_length > 1e-8:
                long_unit = long_vector / long_length
                long_midpoint = (long_first + long_second) * 0.5
                for edge in family.short_edges:
                    short_piece = source_pieces[edge.piece_id]
                    short_first = short_piece[edge.edge_id]
                    short_second = short_piece[
                        (edge.edge_id + 1) % len(short_piece)
                    ]
                    short_vector = short_second - short_first
                    short_length = float(np.linalg.norm(short_vector))
                    if short_length <= 1e-8:
                        continue
                    short_unit = short_vector / short_length
                    parallel_error = abs(
                        float(
                            long_unit[0] * short_unit[1]
                            - long_unit[1] * short_unit[0]
                        )
                    )
                    short_midpoint = (
                        short_first + short_second
                    ) * 0.5
                    distance_scale = max(
                        1.0,
                        0.5 * (long_length + short_length),
                    )
                    midpoint_error = (
                        float(
                            np.linalg.norm(
                                long_midpoint - short_midpoint
                            )
                        )
                        / distance_scale
                    )
                    family_errors.append(
                        parallel_error + 0.35 * midpoint_error
                    )
        result = (
            float(sum(family_errors)),
            len(family_errors),
        )
        source_pose_family_cache[family.signature] = result
        return result

    def source_pose_score(
        topology: tuple[SeamFamily, ...],
    ) -> float:
        """Prefer contacts already close in the captured source pose.

        This is only a small white-card reserve, not a hard constraint.  It
        makes the common "pull the completed card apart by a few centimetres"
        case reach the expensive geometric ranker early, while randomly
        rotated pieces still use the unchanged geometry/neighbor/border
        candidate pools.
        """
        if source_pieces is None:
            return float("inf")
        error_sum = 0.0
        error_count = 0
        for family in topology:
            family_sum, family_count = source_pose_family_score(
                family
            )
            error_sum += family_sum
            error_count += family_count
        if not error_count:
            return float("inf")
        return error_sum / error_count

    accepted: list[
        tuple[float, float, tuple, tuple[SeamFamily, ...]]
    ] = []
    relaxed_accepted: list[
        tuple[float, float, tuple, tuple[SeamFamily, ...]]
    ] = []
    cycle_accepted: list[
        tuple[float, float, tuple, tuple[SeamFamily, ...]]
    ] = []
    cycle_relaxed_accepted: list[
        tuple[float, float, tuple, tuple[SeamFamily, ...]]
    ] = []
    grid_accepted: list[
        tuple[float, float, tuple, tuple[SeamFamily, ...]]
    ] = []
    grid_relaxed_accepted: list[
        tuple[float, float, tuple, tuple[SeamFamily, ...]]
    ] = []
    checked = 0

    # With at most four pieces, a connected contact topology always has a
    # spanning description using at most three seam families. Exhaustively
    # checking 2- and 3-family combinations is deterministic and avoids the
    # branch-order failures of the old beam/DFS search.
    for family_count in (1, 2, 3):
        eligible_records = (
            records
            if family_count < 3
            else [
                record
                for record in records
                if not record[0].is_chain_to_chain
            ]
        )
        for selected_records in itertools.combinations(
            eligible_records,
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
                (
                    heuristic,
                    outer_border_score(edge_masks),
                    signature,
                    selected_tuple,
                )
            )
        if (
            deadline is not None
            and pytime.monotonic() >= deadline
        ):
            break

    # A true 2x2 arrangement needs three declared contacts: one two-edge
    # chain against another two-edge chain, plus one direct contact joining
    # the two pieces on each side.  The generic three-family loop excludes
    # chain families to avoid C(n, 3) blow-up, so enumerate this bounded
    # structure explicitly.
    if (
        piece_count == 4
        and (deadline is None or pytime.monotonic() < deadline)
    ):
        direct_pair_records: dict[tuple[int, int], list[tuple]] = {}
        for record in records:
            family = record[0]
            family_edges = family.edges
            if family.is_chain_to_chain or len(family_edges) != 2:
                continue
            pair = tuple(
                sorted(edge.piece_id for edge in family_edges)
            )
            if pair[0] != pair[1]:
                direct_pair_records.setdefault(pair, []).append(record)

        seen_grid_signatures = set()
        for chain_record in records:
            chain_family = chain_record[0]
            if not chain_family.is_chain_to_chain:
                continue
            side_a_ids = tuple(
                sorted(
                    edge.piece_id
                    for edge in chain_family.chain_a_edges
                )
            )
            side_b_ids = tuple(
                sorted(
                    edge.piece_id
                    for edge in chain_family.chain_b_edges
                )
            )
            side_a_records = direct_pair_records.get(side_a_ids, [])
            side_b_records = direct_pair_records.get(side_b_ids, [])
            for side_a_record, side_b_record in itertools.product(
                side_a_records,
                side_b_records,
            ):
                checked += 1
                if (
                    checked & 0x3F == 0
                    and deadline is not None
                    and pytime.monotonic() >= deadline
                ):
                    break
                selected_records = (
                    chain_record,
                    side_a_record,
                    side_b_record,
                )
                used_mask = 0
                edge_masks = [0] * piece_count
                internal_length = 0.0
                heuristic = 0.0
                selected = []
                conflict = False
                for record in selected_records:
                    (
                        family,
                        _,
                        _,
                        family_edge_masks,
                        family_heuristic,
                    ) = record
                    if used_mask & family.used_mask:
                        conflict = True
                        break
                    used_mask |= family.used_mask
                    internal_length += family.internal_length_cm
                    heuristic += family_heuristic
                    selected.append(family)
                    for piece_id in range(piece_count):
                        edge_masks[piece_id] |= (
                            family_edge_masks[piece_id]
                        )
                if conflict or any(
                    edge_masks[piece_id].bit_count() != 2
                    for piece_id in range(piece_count)
                ):
                    continue
                plausibility_class = dimension_plausibility_class(
                    internal_length
                )
                if not plausibility_class:
                    continue
                selected_tuple = tuple(selected)
                signature = tuple(
                    sorted(family.signature for family in selected)
                )
                if signature in seen_grid_signatures:
                    continue
                seen_grid_signatures.add(signature)
                item = (
                    heuristic,
                    outer_border_score(edge_masks),
                    signature,
                    selected_tuple,
                )
                if plausibility_class == 2:
                    grid_accepted.append(item)
                else:
                    grid_relaxed_accepted.append(item)
            if deadline is not None and pytime.monotonic() >= deadline:
                break

    # Four pieces meeting around an interior region form a four-contact cycle.
    # Any three contacts connect all pieces, but treating the omitted fourth
    # contact as outer perimeter corrupts the rectangle dimension prefilter.
    # Enumerate only the three possible four-piece cycles rather than the
    # prohibitively large general C(family_count, 4) search. Pieces may have
    # additional outer-border vertices; exactly two edges per piece are
    # consumed by the cycle.
    if (
        piece_count == 4
        and (deadline is None or pytime.monotonic() < deadline)
    ):
        pair_records: dict[tuple[int, int], list[tuple]] = {}
        for record in records:
            family = record[0]
            family_edges = family.edges
            if family.is_chain_to_chain or len(family_edges) != 2:
                continue
            pair = tuple(
                sorted(edge.piece_id for edge in family_edges)
            )
            if pair[0] == pair[1]:
                continue
            pair_records.setdefault(pair, []).append(record)

        # Polygon fitting can shorten one side of a visually exact contact
        # just beyond the generic direct-edge tolerance. Add those matches
        # only to the bounded cycle search; do not enlarge the global seam
        # family pool.
        cycle_lengths, cycle_bit_index = build_edge_tables(pieces)
        existing_cycle_signatures = {
            record[0].signature
            for group in pair_records.values()
            for record in group
        }
        for first_piece in range(piece_count):
            for second_piece in range(first_piece + 1, piece_count):
                pair = (first_piece, second_piece)
                for first_edge in range(len(pieces[first_piece])):
                    first_length = cycle_lengths[first_piece][first_edge]
                    if first_length < MIN_EDGE_CM:
                        continue
                    for second_edge in range(len(pieces[second_piece])):
                        second_length = cycle_lengths[
                            second_piece
                        ][second_edge]
                        if second_length < MIN_EDGE_CM:
                            continue
                        matches, normalized_error = tolerance_match(
                            first_length,
                            second_length,
                            FOUR_FAMILY_CYCLE_DIRECT_ABS_TOLERANCE_CM,
                            FOUR_FAMILY_CYCLE_DIRECT_REL_TOLERANCE,
                        )
                        if not matches:
                            continue
                        first_ref = EdgeRef(first_piece, first_edge)
                        second_ref = EdgeRef(second_piece, second_edge)
                        refs = (first_ref, second_ref)
                        family = SeamFamily(
                            long_edge=first_ref,
                            short_edges=(second_ref,),
                            length_error_cm=abs(
                                first_length - second_length
                            ),
                            normalized_error=normalized_error,
                            used_mask=edge_mask(refs, cycle_bit_index),
                            internal_length_cm=0.5 * (
                                first_length + second_length
                            ),
                        )
                        if family.signature in existing_cycle_signatures:
                            continue
                        existing_cycle_signatures.add(family.signature)
                        edge_masks = [0] * piece_count
                        edge_masks[first_piece] = 1 << first_edge
                        edge_masks[second_piece] = 1 << second_edge
                        adjacency = (
                            1 << (
                                first_piece * piece_count
                                + second_piece
                            )
                        ) | (
                            1 << (
                                second_piece * piece_count
                                + first_piece
                            )
                        )
                        pair_records.setdefault(pair, []).append(
                            (
                                family,
                                (1 << first_piece)
                                | (1 << second_piece),
                                adjacency,
                                tuple(edge_masks),
                                0.25 * normalized_error,
                            )
                        )

        cycle_orders = (
            (0, 1, 2, 3),
            (0, 1, 3, 2),
            (0, 2, 1, 3),
        )
        seen_cycle_signatures = set()
        for order in cycle_orders:
            pair_keys = tuple(
                tuple(sorted((order[index], order[(index + 1) % 4])))
                for index in range(4)
            )
            record_groups = [
                pair_records.get(pair_key, [])
                for pair_key in pair_keys
            ]
            if any(not group for group in record_groups):
                continue
            for selected_records in itertools.product(*record_groups):
                checked += 1
                if (
                    checked & 0x3F == 0
                    and deadline is not None
                    and pytime.monotonic() >= deadline
                ):
                    break
                used_mask = 0
                edge_masks = [0] * piece_count
                internal_length = 0.0
                heuristic = 0.0
                selected = []
                conflict = False
                for record in selected_records:
                    (
                        family,
                        _,
                        _,
                        family_edge_masks,
                        family_heuristic,
                    ) = record
                    if used_mask & family.used_mask:
                        conflict = True
                        break
                    used_mask |= family.used_mask
                    internal_length += family.internal_length_cm
                    heuristic += family_heuristic
                    selected.append(family)
                    for piece_id in range(piece_count):
                        edge_masks[piece_id] |= (
                            family_edge_masks[piece_id]
                        )
                if conflict:
                    continue
                plausibility_class = dimension_plausibility_class(
                    internal_length,
                    FOUR_FAMILY_CYCLE_MIN_PERIMETER_DISCRIMINANT_CM2,
                )
                if not plausibility_class:
                    continue
                if any(
                    edge_masks[piece_id].bit_count()
                    != 2
                    for piece_id in range(piece_count)
                ):
                    continue
                selected_tuple = tuple(selected)
                signature = tuple(
                    sorted(family.signature for family in selected)
                )
                if signature in seen_cycle_signatures:
                    continue
                seen_cycle_signatures.add(signature)
                item = (
                    heuristic,
                    outer_border_score(edge_masks),
                    signature,
                    selected_tuple,
                )
                if plausibility_class == 2:
                    cycle_accepted.append(item)
                else:
                    cycle_relaxed_accepted.append(item)
            if deadline is not None and pytime.monotonic() >= deadline:
                break

    # Preserve the proven 1xN ranking exactly. Chain-to-chain layouts use
    # their dedicated reserve below instead of shifting old candidates out
    # of the generic and three-family quotas.
    accepted.sort(
        key=lambda item: (
            any(
                family.is_chain_to_chain
                for family in item[3]
            ),
            item[0],
            item[2],
        )
    )
    relaxed_accepted.sort(
        key=lambda item: (
            any(
                family.is_chain_to_chain
                for family in item[3]
            ),
            item[0],
            item[2],
        )
    )
    generic_accepted = [
        item
        for item in accepted
        if not any(
            family.is_chain_to_chain
            for family in item[3]
        )
    ]
    selection_has_time = (
        deadline is None or pytime.monotonic() < deadline
    )
    if rich_card_mode(texture_context) and selection_has_time:
        geometry_limit = min(
            WHITE_CARD_GEOMETRY_TOPOLOGIES,
            MAX_ENUMERATED_TOPOLOGIES,
        )
        # Source-pose scoring is substantially more expensive than the
        # geometry key.  It is only a reserve, so rank a bounded prefix and
        # never let it consume the following coarse/overlap stages.
        source_pose_pool = generic_accepted[
            :MAX_CARD_SOURCE_POSE_RANK_INPUT
        ]
        source_pose_ranked = sorted(
            source_pose_pool,
            key=lambda item: (
                source_pose_score(item[3]),
                item[0],
                item[2],
            ),
        )
        selected_items = list(
            source_pose_ranked[
                :min(
                    WHITE_CARD_SOURCE_POSE_TOPOLOGIES,
                    MAX_ENUMERATED_TOPOLOGIES,
                )
            ]
        )
        selected_signatures = {
            item[2] for item in selected_items
        }
        for item in generic_accepted[:geometry_limit]:
            if item[2] in selected_signatures:
                continue
            selected_items.append(item)
            selected_signatures.add(item[2])
        # Edge fitting noise often changes only which parallel edge realizes
        # a fixed piece-contact graph.  Capture 112159's true T-junction is
        # such a sibling of an early but visibly wrong topology.  Explore a
        # small round-robin neighborhood of the first contact structures so
        # one large family cannot consume the complete reserve.
        def contact_structure(item) -> tuple:
            return tuple(
                sorted(
                    (
                        family.edges[0].piece_id,
                        tuple(
                            sorted(
                                edge.piece_id
                                for edge in family.edges[1:]
                            )
                        ),
                    )
                    for family in item[3]
                )
            )

        seed_structures = []
        for item in generic_accepted:
            if (
                deadline is not None
                and pytime.monotonic() >= deadline
            ):
                break
            structure = contact_structure(item)
            if structure in seed_structures:
                continue
            seed_structures.append(structure)
            if (
                len(seed_structures)
                >= WHITE_CARD_NEIGHBOR_STRUCTURES
            ):
                break
        neighbor_buckets = [
            [
                item
                for item in generic_accepted
                if (
                    item[2] not in selected_signatures
                    and contact_structure(item) == structure
                )
            ]
            for structure in seed_structures
        ]
        neighbor_added = 0
        neighbor_cursor = 0
        while (
            neighbor_added < WHITE_CARD_NEIGHBOR_TOPOLOGIES
            and neighbor_buckets
            and (
                deadline is None
                or pytime.monotonic() < deadline
            )
        ):
            next_buckets = []
            for bucket in neighbor_buckets:
                if neighbor_cursor < len(bucket):
                    item = bucket[neighbor_cursor]
                    if item[2] not in selected_signatures:
                        selected_items.append(item)
                        selected_signatures.add(item[2])
                        neighbor_added += 1
                    next_buckets.append(bucket)
                    if (
                        neighbor_added
                        >= WHITE_CARD_NEIGHBOR_TOPOLOGIES
                    ):
                        break
            neighbor_buckets = next_buckets
            neighbor_cursor += 1
        if deadline is None or pytime.monotonic() < deadline:
            border_ranked = sorted(
                generic_accepted,
                key=lambda item: (item[1], item[0], item[2]),
            )
            for item in border_ranked:
                if item[2] in selected_signatures:
                    continue
                selected_items.append(item)
                selected_signatures.add(item[2])
                if len(selected_items) >= MAX_ENUMERATED_TOPOLOGIES:
                    break
    elif use_card_border_rank and selection_has_time:
        border_ranked = sorted(
            generic_accepted,
            key=lambda item: (item[1], item[0], item[2]),
        )
        geometry_rank = {
            item[2]: rank
            for rank, item in enumerate(generic_accepted)
        }
        border_rank = {
            item[2]: rank
            for rank, item in enumerate(border_ranked)
        }
        sparse_card_ranked = sorted(
            generic_accepted,
            key=lambda item: (
                geometry_rank[item[2]]
                + SPARSE_CARD_BORDER_RANK_WEIGHT
                * border_rank[item[2]],
                item[0],
                item[2],
            ),
        )
        selected_items = sparse_card_ranked[
            :MAX_ENUMERATED_TOPOLOGIES
        ]
        selected_signatures = {
            item[2] for item in selected_items
        }
    else:
        selected_items = generic_accepted[
            :MAX_ENUMERATED_TOPOLOGIES
        ]
        selected_signatures = {
            item[2] for item in selected_items
        }

    chain_items = [
        item
        for item in accepted
        if any(
            family.is_chain_to_chain
            for family in item[3]
        )
    ]
    chain_selected = []
    if source_pieces is not None:
        chain_selected.extend(
            sorted(
                chain_items,
                key=lambda item: (
                    source_pose_score(item[3]),
                    item[0],
                    item[2],
                ),
            )[:MAX_CHAIN_TO_CHAIN_SOURCE_TOPOLOGIES]
        )
    chain_selected.extend(
        chain_items[:MAX_CHAIN_TO_CHAIN_GEOMETRY_TOPOLOGIES]
    )
    chain_to_chain_added = 0
    for item in chain_selected:
        if item[2] in selected_signatures:
            continue
        selected_items.append(item)
        selected_signatures.add(item[2])
        chain_to_chain_added += 1
        if chain_to_chain_added >= MAX_CHAIN_TO_CHAIN_TOPOLOGIES:
            break

    grid_accepted.sort(key=lambda item: (item[0], item[2]))
    grid_selected = []
    if source_pieces is not None:
        grid_selected.extend(
            sorted(
                grid_accepted,
                key=lambda item: (
                    source_pose_score(item[3]),
                    item[0],
                    item[2],
                ),
            )[:MAX_TWO_BY_TWO_GRID_SOURCE_TOPOLOGIES]
        )
    grid_selected.extend(grid_accepted)
    two_by_two_grid_added = 0
    for item in grid_selected:
        if item[2] in selected_signatures:
            continue
        selected_items.append(item)
        selected_signatures.add(item[2])
        two_by_two_grid_added += 1
        if two_by_two_grid_added >= MAX_TWO_BY_TWO_GRID_TOPOLOGIES:
            break

    cycle_accepted.sort(key=lambda item: (item[0], item[2]))
    cycle_selected = []
    if source_pieces is not None:
        cycle_selected.extend(
            sorted(
                cycle_accepted,
                key=lambda item: (
                    source_pose_score(item[3]),
                    item[0],
                    item[2],
                ),
            )[:MAX_FOUR_FAMILY_CYCLE_SOURCE_TOPOLOGIES]
        )
    cycle_selected.extend(cycle_accepted)
    four_family_cycle_added = 0
    for item in cycle_selected:
        if item[2] in selected_signatures:
            continue
        selected_items.append(item)
        selected_signatures.add(item[2])
        four_family_cycle_added += 1
        if (
            four_family_cycle_added
            >= MAX_FOUR_FAMILY_CYCLE_TOPOLOGIES
        ):
            break

    three_family_added = 0
    for item in generic_accepted:
        if (
            len(item[3]) < 3
            or item[2] in selected_signatures
        ):
            continue
        selected_items.append(item)
        selected_signatures.add(item[2])
        three_family_added += 1
        if (
            three_family_added
            >= MAX_THREE_FAMILY_RESERVE_TOPOLOGIES
        ):
            break
    strict_topologies = [
        topology
        for _, _, _, topology in selected_items
    ]
    relaxed_topologies = [
        topology
        for _, _, _, topology in relaxed_accepted[
            :MAX_RELAXED_TOPOLOGIES
        ]
    ]
    relaxed_cycle_topologies = [
        topology
        for _, _, _, topology in sorted(
            cycle_relaxed_accepted,
            key=lambda item: (
                source_pose_score(item[3]),
                item[0],
                item[2],
            ),
        )[:MAX_RELAXED_TOPOLOGIES]
    ]
    relaxed_grid_topologies = [
        topology
        for _, _, _, topology in sorted(
            grid_relaxed_accepted,
            key=lambda item: (
                source_pose_score(item[3]),
                item[0],
                item[2],
            ),
        )[:MAX_RELAXED_TOPOLOGIES]
    ]
    combined = (
        strict_topologies
        + relaxed_topologies
        + relaxed_grid_topologies
        + relaxed_cycle_topologies
    )
    if return_strict_count:
        return combined, len(strict_topologies)
    return combined


def seam_point_pairs(
    pieces: list[np.ndarray],
    family: SeamFamily,
    short_order: tuple[EdgeRef, ...],
) -> list[tuple[int, np.ndarray, int, np.ndarray]]:
    if family.is_chain_to_chain:
        side_a_count = len(family.chain_a_edges)
        side_a = short_order[:side_a_count]
        side_b = short_order[side_a_count:]

        def chain_intervals(chain):
            edge_lengths = []
            for edge in chain:
                first, second = edge_vertices(
                    pieces[edge.piece_id],
                    edge,
                )
                edge_lengths.append(
                    legacy.edge_length(first, second)
                )
            total = max(sum(edge_lengths), 1e-9)
            boundaries = [0.0]
            for length in edge_lengths:
                boundaries.append(
                    boundaries[-1] + length / total
                )
            boundaries[-1] = 1.0
            return edge_lengths, boundaries

        _, boundaries_a = chain_intervals(side_a)
        _, boundaries_b = chain_intervals(side_b)
        boundaries = sorted(
            set(boundaries_a + boundaries_b)
        )
        pairs = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            midpoint = 0.5 * (start + end)
            index_a = next(
                index
                for index in range(len(side_a))
                if midpoint <= boundaries_a[index + 1] + 1e-9
            )
            index_b = next(
                index
                for index in range(len(side_b))
                if midpoint <= boundaries_b[index + 1] + 1e-9
            )
            edge_a = side_a[index_a]
            edge_b = side_b[index_b]
            width_a = max(
                boundaries_a[index_a + 1]
                - boundaries_a[index_a],
                1e-9,
            )
            width_b = max(
                boundaries_b[index_b + 1]
                - boundaries_b[index_b],
                1e-9,
            )
            first_a, second_a = edge_vertices(
                pieces[edge_a.piece_id],
                edge_a,
            )
            first_b, second_b = edge_vertices(
                pieces[edge_b.piece_id],
                edge_b,
            )
            for position in (start, end):
                fraction_a = (
                    position - boundaries_a[index_a]
                ) / width_a
                fraction_b = (
                    position - boundaries_b[index_b]
                ) / width_b
                pairs.append(
                    (
                        edge_a.piece_id,
                        legacy.edge_point(
                            first_a,
                            second_a,
                            fraction_a,
                        ),
                        edge_b.piece_id,
                        legacy.edge_point(
                            second_b,
                            first_b,
                            fraction_b,
                        ),
                    )
                )
        return pairs

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
    iterations: int = POSE_OPTIMIZATION_ITERATIONS,
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
    for _ in range(iterations):
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
    include_half_turns: bool = False,
) -> list[tuple[str, list[Placement]]]:
    """Test the bounded playing-card symmetries left by geometry.

    Four-piece card cuts commonly produce two already coherent half-card
    columns.  Edge-topology enumeration can lock those columns to the wrong
    left/right side because both geometric arrangements have the same shape.
    It can also put the two pieces of one column in the wrong vertical slots,
    or put either rectangular two-piece column upside down.  The former is
    invisible to geometry when the pieces have matching card-cut shapes, but
    leaves printed artwork on the finished card border.

    The alternatives are rigid motions only:

    * exchange the two columns by translation;
    * exchange the two vertical slots inside either column by translation;
    * turn either complete column by 180 degrees;
    * turn a single near-centrally-symmetric fragment by 180 degrees.

    An arbitrary polygon still cannot turn in place without breaking the
    rectangle. Single-piece variants therefore require high polygon overlap
    after the turn and remain bounded. Every generated layout is checked
    again for dimensions, IoU and overlap before texture scoring.
    """
    variants: list[tuple[str, list[Placement]]] = []
    if not white_card_mode(texture_context) or len(placements) != 4:
        return [("topology", placements)]

    def column_groups(layout):
        points = np.concatenate(
            [
                np.asarray(item.vertices, dtype=np.float64)
                for item in layout
            ],
            axis=0,
        ).astype(np.float32)
        box = cv2.boxPoints(cv2.minAreaRect(points)).astype(np.float64)
        first_axis = box[1] - box[0]
        second_axis = box[3] - box[0]
        # In landscape card coordinates the two half-card columns are
        # separated along the completed card's long axis.
        split_axis = (
            first_axis
            if np.linalg.norm(first_axis) >= np.linalg.norm(second_axis)
            else second_axis
        )
        axis_length = float(np.linalg.norm(split_axis))
        if axis_length < 1e-8:
            return None
        axis = split_axis / axis_length
        projections = [
            float(
                np.dot(
                    np.mean(
                        np.asarray(item.vertices, dtype=np.float64),
                        axis=0,
                    ),
                    axis,
                )
            )
            for item in layout
        ]
        order = np.argsort(projections)
        low_ids = frozenset(int(index) for index in order[:2])
        high_ids = frozenset(int(index) for index in order[2:])
        low_points = np.concatenate(
            [
                np.asarray(layout[index].vertices, dtype=np.float64)
                for index in sorted(low_ids)
            ],
            axis=0,
        )
        high_points = np.concatenate(
            [
                np.asarray(layout[index].vertices, dtype=np.float64)
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
        # Reject layouts that do not actually consist of two separated
        # card-width groups.  This prevents arbitrary pieces being turned.
        if high_min < low_max - 0.35:
            return None
        return (
            axis,
            low_ids,
            high_ids,
            low_min,
            low_max,
            high_min,
            high_max,
        )

    def translate_columns(layout, groups):
        (
            axis,
            low_ids,
            high_ids,
            low_min,
            low_max,
            high_min,
            high_max,
        ) = groups
        low_delta = (high_max - low_max) * axis
        high_delta = (low_min - high_min) * axis
        result = []
        for index, item in enumerate(layout):
            delta = low_delta if index in low_ids else high_delta
            result.append(
                Placement(
                    piece_id=item.piece_id,
                    vertices=np.asarray(item.vertices) + delta,
                    rotation=np.asarray(item.rotation).copy(),
                    translation=np.asarray(item.translation) + delta,
                    used_edges=item.used_edges,
                    edge_coverage=item.edge_coverage,
                )
            )
        return result

    def swap_column_rows(layout, group_ids):
        """Exchange the two occupied slots in one already coherent column."""
        if len(group_ids) != 2:
            return layout
        first, second = sorted(group_ids)
        first_center = np.mean(
            np.asarray(layout[first].vertices, dtype=np.float64),
            axis=0,
        )
        second_center = np.mean(
            np.asarray(layout[second].vertices, dtype=np.float64),
            axis=0,
        )
        deltas = {
            first: second_center - first_center,
            second: first_center - second_center,
        }
        result = []
        for index, item in enumerate(layout):
            delta = deltas.get(index)
            if delta is None:
                result.append(
                    Placement(
                        piece_id=item.piece_id,
                        vertices=np.asarray(item.vertices).copy(),
                        rotation=np.asarray(item.rotation).copy(),
                        translation=np.asarray(item.translation).copy(),
                        used_edges=item.used_edges,
                        edge_coverage=item.edge_coverage,
                    )
                )
                continue
            result.append(
                Placement(
                    piece_id=item.piece_id,
                    vertices=np.asarray(item.vertices) + delta,
                    rotation=np.asarray(item.rotation).copy(),
                    translation=np.asarray(item.translation) + delta,
                    used_edges=item.used_edges,
                    edge_coverage=item.edge_coverage,
                )
            )
        return result

    def half_turn_group(layout, group_ids):
        group_points = np.concatenate(
            [
                np.asarray(layout[index].vertices, dtype=np.float64)
                for index in sorted(group_ids)
            ],
            axis=0,
        ).astype(np.float32)
        center = np.asarray(
            cv2.minAreaRect(group_points)[0],
            dtype=np.float64,
        )
        result = []
        for index, item in enumerate(layout):
            if index in group_ids:
                # target = R*source+t; a half turn about C gives
                # target' = -R*source + (2*C-t).
                result.append(
                    Placement(
                        piece_id=item.piece_id,
                        vertices=(
                            2.0 * center
                            - np.asarray(
                                item.vertices,
                                dtype=np.float64,
                            )
                        ),
                        rotation=-np.asarray(
                            item.rotation,
                            dtype=np.float64,
                        ),
                        translation=(
                            2.0 * center
                            - np.asarray(
                                item.translation,
                                dtype=np.float64,
                            )
                        ),
                        used_edges=item.used_edges,
                        edge_coverage=item.edge_coverage,
                    )
                )
            else:
                result.append(
                    Placement(
                        piece_id=item.piece_id,
                        vertices=np.asarray(item.vertices).copy(),
                        rotation=np.asarray(item.rotation).copy(),
                        translation=np.asarray(item.translation).copy(),
                        used_edges=item.used_edges,
                        edge_coverage=item.edge_coverage,
                    )
                )
        return result

    def single_half_turn_similarity(item):
        vertices = np.asarray(item.vertices, dtype=np.float64)
        center = np.asarray(
            cv2.minAreaRect(vertices.astype(np.float32))[0],
            dtype=np.float64,
        )
        turned = 2.0 * center - vertices
        area = abs(float(cv2.contourArea(vertices.astype(np.float32))))
        turned_area = abs(
            float(cv2.contourArea(turned.astype(np.float32)))
        )
        intersection, _ = cv2.intersectConvexConvex(
            vertices.astype(np.float32),
            turned.astype(np.float32),
        )
        union = area + turned_area - float(intersection)
        return float(intersection) / max(union, 1e-9)

    seen = set()

    def append_if_legal(name, layout, always=False):
        signature = tuple(
            np.round(
                np.asarray(item.vertices, dtype=np.float64),
                3,
            ).tobytes()
            for item in layout
        )
        if signature in seen:
            return
        if not always:
            iou, overlap, width, height = layout_metrics(layout)
            if (
                not legal_dimensions(width, height, texture_context)
                or iou < MIN_RECTANGLE_IOU
                or overlap > MAX_OVERLAP_RATIO
            ):
                return
        seen.add(signature)
        variants.append((name, layout))

    def row_swap_name(group_ids):
        return "column_row_swap_P{}".format(
            "_P".join(
                str(int(placements[index].piece_id) + 1)
                for index in sorted(group_ids)
            )
        )

    groups = column_groups(placements)
    # During global topology search retain only the five rigid slot
    # arrangements. Half turns are a second-stage orientation decision;
    # mixing them into topology ranking lets an accidental seam from another
    # topology steal the solution.
    base_layouts = []

    def append_base_layout(name, layout):
        append_if_legal(name, layout, always=name == "topology")
        base_groups = column_groups(layout)
        if include_half_turns and base_groups is not None:
            base_layouts.append((name, layout, base_groups))

    append_base_layout("topology", placements)
    if groups is not None:
        swapped = translate_columns(placements, groups)
        append_base_layout("column_swap", swapped)
        low_ids = groups[1]
        high_ids = groups[2]
        low_swapped = swap_column_rows(placements, low_ids)
        high_swapped = swap_column_rows(placements, high_ids)
        append_base_layout(row_swap_name(low_ids), low_swapped)
        append_base_layout(row_swap_name(high_ids), high_swapped)
        both_swapped = swap_column_rows(low_swapped, high_ids)
        append_base_layout("both_column_row_swap", both_swapped)

    # Test either column independently.  Turning both columns is only a
    # global 180-degree rotation of the entire finished card; the target
    # frame already has that free global orientation, so emitting it would
    # add four unnecessary actuator rotations and can hide which individual
    # column is actually reversed.
    if not include_half_turns:
        return variants
    for base_name, base_layout, base_groups in base_layouts:
        if base_groups is None:
            continue
        low_ids = base_groups[1]
        high_ids = base_groups[2]
        for turn_groups in (
            (low_ids,),
            (high_ids,),
        ):
            turned = base_layout
            turned_piece_ids = []
            for group_ids in turn_groups:
                turned = half_turn_group(turned, group_ids)
                turned_piece_ids.extend(
                    int(base_layout[index].piece_id) + 1
                    for index in sorted(group_ids)
                )
            suffix = "_half_turn_P{}".format(
                "_P".join(
                    str(piece_id)
                    for piece_id in sorted(turned_piece_ids)
                )
            )
            append_if_legal(base_name + suffix, turned)

    # A nearly centrally symmetric fragment can occupy the same geometric
    # slot after an individual half turn while carrying different artwork.
    # Enumerate only the strongest bounded cases; arbitrary polygons still
    # require their complete two-piece column to turn together.
    single_turns = []
    for base_name, base_layout, _ in base_layouts:
        for index, item in enumerate(base_layout):
            similarity = single_half_turn_similarity(item)
            if similarity < CARD_SINGLE_HALF_TURN_MIN_POLYGON_IOU:
                continue
            single_turns.append((
                -similarity,
                base_name,
                base_layout,
                index,
                int(item.piece_id) + 1,
            ))
    single_turns.sort(key=lambda value: (value[0], value[1], value[4]))
    for _, base_name, base_layout, index, piece_id in single_turns[
        :CARD_SINGLE_HALF_TURN_MAX_VARIANTS
    ]:
        turned = half_turn_group(base_layout, frozenset((index,)))
        append_if_legal(
            "{}_single_half_turn_P{}".format(base_name, piece_id),
            turned,
        )

    return variants


def refine_card_column_orientation(
    candidate: LayoutCandidate,
    original: list[np.ndarray],
    texture_context: texture_matcher.TextureContext | None,
) -> tuple[LayoutCandidate, list[dict]]:
    """Choose card artwork orientation only after geometry chose a topology.

    A half turn of one complete two-piece column leaves the rectangle and its
    edge topology unchanged.  It must therefore not participate in the large
    global topology search.  At this bounded second stage we compare at most
    fifteen rigid variants of the already selected rectangle and use the
    pattern measured across its *actual* internal contacts as the primary
    direction cue. Near-symmetric single-piece turns are also bounded by
    polygon overlap before entering this scoring stage.
    """
    if not white_card_mode(texture_context):
        return candidate, []

    alternatives = []
    evaluated_variants = []
    minimum_contacts = max(
        len(original) - 1,
        DIRECT_SEAM_MIN_CONTACTS_PER_PIECE,
    )
    for local_name, placements in card_layout_variants(
        candidate.solution.placements,
        texture_context,
        include_half_turns=True,
    ):
        iou, overlap, width, height = layout_metrics(placements)
        minimum_iou = (
            TWO_BY_TWO_GRID_MIN_IOU
            if candidate.uses_two_by_two_grid
            else (
                CHAIN_CARD_MIN_IOU
                if candidate.uses_chain_to_chain
                else CARD_ORIENTATION_MIN_IOU - CARD_FINAL_IOU_GUARD
            )
        )
        maximum_overlap = (
            CHAIN_CARD_MAX_OVERLAP_RATIO
            if candidate.uses_chain_to_chain
            else (
                CARD_FINAL_MAX_OVERLAP_RATIO
                + CARD_FINAL_OVERLAP_GUARD
            )
        )
        if (
            not legal_dimensions(
                width,
                height,
                texture_context,
                candidate.uses_chain_to_chain,
            )
            or iou < minimum_iou
            or overlap > maximum_overlap
        ):
            evaluated_variants.append(
                {
                    "variant": local_name,
                    "accepted": False,
                    "reject_reason": "geometry",
                    "iou": round(float(iou), 6),
                    "overlap": round(float(overlap), 6),
                }
            )
            continue
        result = texture_matcher.score_layout(
            texture_context,
            original,
            placements,
            None,
            None,
        )
        confidence = float(result.get("confidence", 0.0))
        contacts = int(result.get("contact_segments", 0))
        perimeter_confidence = float(
            result.get("perimeter_confidence", 0.0)
        )
        if (
            confidence < TEXTURE_MIN_CONFIDENCE
            or contacts < minimum_contacts
            or perimeter_confidence
            < (
                CARD_ORIENTATION_MIN_PERIMETER_CONFIDENCE
                - CARD_PERIMETER_CONFIDENCE_GUARD
            )
        ):
            evaluated_variants.append(
                {
                    "variant": local_name,
                    "accepted": False,
                    "reject_reason": (
                        "texture_or_perimeter_confidence"
                    ),
                    "texture_confidence": round(confidence, 6),
                    "contacts": contacts,
                    "perimeter_confidence": round(
                        perimeter_confidence,
                        6,
                    ),
                }
            )
            continue
        seam_score = float(
            result.get("seam_score", result.get("score", 1.0))
        )
        symmetry_score = float(
            result.get("symmetry_score", seam_score)
        )
        perimeter_score = float(
            result.get("perimeter_score", seam_score)
        )
        # A complete playing card must have its white border on all four
        # outside edges.  This is as important as internal cut continuity:
        # capture 142730 has a deceptively good blank internal seam in the
        # wrong direction, while its patterned outer long edge exposes the
        # mistake immediately.  Half-turn symmetry remains only a tie-break.
        orientation_score = (
            seam_score
            + CARD_ORIENTATION_PERIMETER_WEIGHT
            * perimeter_score
            + CARD_ORIENTATION_SYMMETRY_WEIGHT
            * symmetry_score
        )
        if local_name == "topology":
            name = candidate.layout_variant
        elif candidate.layout_variant == "topology":
            name = local_name
        else:
            name = candidate.layout_variant + "__" + local_name
        alternatives.append(
            {
                "name": name,
                "placements": placements,
                "iou": iou,
                "overlap": overlap,
                "width": width,
                "height": height,
                "result": result,
                "orientation_score": orientation_score,
            }
        )
        evaluated_variants.append(
            {
                "variant": name,
                "accepted": True,
                "orientation_score": round(
                    float(orientation_score),
                    6,
                ),
                "seam_score": round(seam_score, 6),
                "perimeter_score": round(perimeter_score, 6),
                "perimeter_worst_side_score": round(
                    float(
                        result.get(
                            "perimeter_worst_side_score",
                            1.0,
                        )
                    ),
                    6,
                ),
                "perimeter_worst_edge_score": round(
                    float(
                        result.get(
                            "perimeter_worst_edge_score",
                            1.0,
                        )
                    ),
                    6,
                ),
                "perimeter_side_scores": [
                    round(float(value), 6)
                    for value in result.get(
                        "perimeter_side_scores",
                        [],
                    )
                ],
            }
        )

    if not alternatives:
        return candidate, []
    alternatives.sort(
        key=lambda item: (
            float(
                item["result"].get(
                    "perimeter_score",
                    1.0,
                )
            ),
            item["orientation_score"],
            -item["iou"],
            item["overlap"],
        )
    )
    best_perimeter = float(
        alternatives[0]["result"].get(
            "perimeter_score",
            1.0,
        )
    )
    white_border_candidates = [
        item
        for item in alternatives
        if float(
            item["result"].get(
                "perimeter_score",
                1.0,
            )
        )
        <= best_perimeter + CARD_ORIENTATION_PERIMETER_DELTA
    ]
    white_border_candidates.sort(
        key=lambda item: (
            item["orientation_score"],
            -item["iou"],
            item["overlap"],
        )
    )
    selected = white_border_candidates[0]

    # Do not normalize the result by piece IDs.  Two single-column half turns
    # are not a harmless global card turn: only one keeps the Q/J/K artwork
    # continuous.  The assembled-image score above must make this decision.
    result = selected["result"]
    layout_score = float(result.get("score", 1.0))
    confidence = float(result.get("confidence", 0.0))
    candidate.solution = Solution(
        placements=selected["placements"],
        rectangularity=float(selected["iou"]),
        width_cm=float(selected["width"]),
        height_cm=float(selected["height"]),
        search_nodes=candidate.solution.search_nodes,
    )
    candidate.iou = float(selected["iou"])
    candidate.overlap_ratio = float(selected["overlap"])
    candidate.texture_score = layout_score
    candidate.texture_confidence = confidence
    candidate.source_texture_score = layout_score
    candidate.layout_texture_score = layout_score
    candidate.layout_seam_score = float(
        result.get("seam_score", layout_score)
    )
    candidate.layout_symmetry_score = float(
        result.get("symmetry_score", layout_score)
    )
    candidate.layout_perimeter_score = float(
        result.get("perimeter_score", layout_score)
    )
    candidate.layout_perimeter_confidence = float(
        result.get("perimeter_confidence", 0.0)
    )
    candidate.layout_perimeter_worst_side_score = float(
        result.get("perimeter_worst_side_score", 1.0)
    )
    candidate.layout_perimeter_worst_edge_score = float(
        result.get("perimeter_worst_edge_score", 1.0)
    )
    candidate.layout_contact_segments = int(
        result.get("contact_segments", 0)
    )
    candidate.layout_variant = str(selected["name"])
    candidate.score = (
        (1.0 - candidate.iou)
        + 1.7 * candidate.overlap_ratio
        + 0.12 * min(candidate.seam_rms_cm, 2.0)
        + texture_rank_weight(texture_context)
        * candidate.texture_score
        * candidate.texture_confidence
    )
    return candidate, evaluated_variants


def card_candidate_from_solution(
    solution: Solution,
    original: list[np.ndarray],
    texture_context: texture_matcher.TextureContext,
) -> LayoutCandidate:
    """Wrap a DFS rectangle so it receives the normal card visual checks."""
    iou, overlap, width, height = layout_metrics(solution.placements)
    result = texture_matcher.score_layout(
        texture_context,
        original,
        solution.placements,
        None,
        None,
    )
    texture_score = float(result.get("score", 1.0))
    texture_confidence = float(result.get("confidence", 0.0))
    perimeter_score = float(
        result.get("perimeter_score", texture_score)
    )
    solution.rectangularity = iou
    solution.width_cm = width
    solution.height_cm = height
    score = (
        (1.0 - iou)
        + 1.7 * overlap
        + texture_rank_weight(texture_context)
        * texture_score
        * texture_confidence
    )
    return LayoutCandidate(
        solution=solution,
        score=score,
        iou=iou,
        overlap_ratio=overlap,
        # DFS places every new piece by an explicit edge transform.  It has
        # no global least-squares seam residual; overlap and final IoU are
        # the appropriate independent geometry checks for this path.
        seam_rms_cm=0.0,
        texture_score=texture_score,
        texture_confidence=texture_confidence,
        source_texture_score=texture_score,
        layout_texture_score=texture_score,
        source_seam_score=float(
            result.get("seam_score", texture_score)
        ),
        source_perimeter_score=perimeter_score,
        layout_seam_score=float(
            result.get("seam_score", texture_score)
        ),
        layout_symmetry_score=float(
            result.get("symmetry_score", texture_score)
        ),
        layout_perimeter_score=perimeter_score,
        layout_perimeter_confidence=float(
            result.get("perimeter_confidence", 0.0)
        ),
        layout_perimeter_worst_side_score=float(
            result.get("perimeter_worst_side_score", 1.0)
        ),
        layout_perimeter_worst_edge_score=float(
            result.get("perimeter_worst_edge_score", 1.0)
        ),
        layout_contact_segments=int(
            result.get("contact_segments", 0)
        ),
        layout_variant="legacy_card_dfs",
        topology_signature=("legacy_card_dfs",),
        uses_chain_to_chain=False,
        uses_two_by_two_grid=False,
    )


def card_candidate_is_executable(candidate: LayoutCandidate) -> bool:
    """Apply guarded geometry and only the absolute border sanity limits."""
    return bool(
        candidate.iou
        >= CARD_FINAL_MIN_IOU - CARD_FINAL_IOU_GUARD
        and candidate.overlap_ratio
        <= CARD_FINAL_MAX_OVERLAP_RATIO + CARD_FINAL_OVERLAP_GUARD
        and candidate.texture_confidence >= TEXTURE_MIN_CONFIDENCE
        and candidate.layout_perimeter_confidence
        >= (
            CARD_ORIENTATION_MIN_PERIMETER_CONFIDENCE
            - CARD_PERIMETER_CONFIDENCE_GUARD
        )
        and candidate.layout_perimeter_worst_side_score
        <= CARD_ABSOLUTE_MAX_WORST_SIDE_SCORE
        and candidate.layout_perimeter_worst_edge_score
        <= CARD_ABSOLUTE_MAX_WORST_EDGE_SCORE
    )


def layout_metrics(
    placements: list[Placement],
    rectangle_metrics: tuple[float, float, float] | None = None,
    total_area: float | None = None,
) -> tuple[float, float, float, float]:
    if rectangle_metrics is None:
        _, width, height, rectangle_area = legacy.minimum_rectangle(
            placements
        )
    else:
        width, height, rectangle_area = rectangle_metrics
    if rectangle_area <= 1e-9:
        return 0.0, float("inf"), width, height
    if total_area is None:
        total_area = sum(
            legacy.polygon_area(placement.vertices)
            for placement in placements
        )

    # Each polygon participates in three pair checks for a four-piece layout.
    # Prepare invariant OpenCV inputs once while preserving the exact overlap
    # implementation and pair accumulation order used by the legacy helper.
    prepared = []
    native_convex_intersection = hasattr(
        cv2,
        "intersectConvexConvex",
    )
    for placement in placements:
        vertices = placement.vertices
        vertices_float = np.asarray(vertices, dtype=np.float32)
        prepared.append(
            (
                vertices,
                np.min(vertices, axis=0),
                np.max(vertices, axis=0),
                vertices_float,
                bool(
                    native_convex_intersection
                    and cv2.isContourConvex(vertices_float)
                ),
            )
        )

    overlap_area = 0.0
    for first_index, first in enumerate(prepared):
        for second in prepared[first_index + 1:]:
            common_min = np.maximum(first[1], second[1])
            common_max = np.minimum(first[2], second[2])
            if np.any(common_max <= common_min):
                continue
            if first[4] and second[4]:
                intersection_area, _ = cv2.intersectConvexConvex(
                    first[3],
                    second[3],
                )
                overlap_area += float(intersection_area)
            else:
                overlap_area += legacy.overlap_area_raster(
                    first[0],
                    second[0],
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


def rich_card_mode(
    texture_context: texture_matcher.TextureContext | None,
) -> bool:
    """Whether artwork is rich enough to justify expanded card ranking."""
    return bool(
        white_card_mode(texture_context)
        and texture_context is not None
        and texture_context.texture_richness
        >= RICH_CARD_MODE_CONFIDENCE
    )


def texture_rank_weight(
    texture_context: texture_matcher.TextureContext | None,
) -> float:
    """Use artwork strongly only when the capture contains rich information."""
    if rich_card_mode(texture_context):
        return RICH_CARD_TEXTURE_SCORE_WEIGHT
    return TEXTURE_SCORE_WEIGHT


def legal_dimensions(
    width: float,
    height: float,
    texture_context: texture_matcher.TextureContext | None = None,
    allow_chain_to_chain: bool = False,
) -> bool:
    short_side, long_side = sorted((width, height))
    if not (
        TARGET_MIN_SHORT_CM <= short_side <= TARGET_MAX_SHORT_CM
        and TARGET_MIN_LONG_CM <= long_side <= TARGET_MAX_LONG_CM
    ):
        return False
    if white_card_mode(texture_context):
        aspect_ratio = long_side / max(short_side, 1e-9)
        if allow_chain_to_chain:
            return (
                CHAIN_CARD_MIN_ASPECT_RATIO
                <= aspect_ratio
                <= CHAIN_CARD_MAX_ASPECT_RATIO
            )
        return (
            WHITE_CARD_MIN_ASPECT_RATIO
            <= aspect_ratio
            <= WHITE_CARD_MAX_ASPECT_RATIO
        )
    return True


def question_1_2_dimensions(width: float, height: float) -> bool:
    """Hard target-size gate for the first two geometry questions."""
    short_side, long_side = sorted((width, height))
    return (
        CORE_MIN_SHORT_CM <= short_side <= CORE_MAX_SHORT_CM
        and CORE_MIN_LONG_CM <= long_side <= CORE_MAX_LONG_CM
    )


def topology_orders(
    topology: tuple[SeamFamily, ...],
    order_cache: dict[tuple, list] | None = None,
) -> list[tuple[tuple[EdgeRef, ...], ...]]:
    cache_key = tuple(sorted(family.signature for family in topology))
    if order_cache is not None:
        cached = order_cache.get(cache_key)
        if cached is not None:
            return cached
    choices = []
    for family in topology:
        if family.is_chain_to_chain:
            choices.append(
                [
                    side_a + side_b
                    for side_a in itertools.permutations(
                        family.chain_a_edges
                    )
                    for side_b in itertools.permutations(
                        family.chain_b_edges
                    )
                ]
            )
        elif len(family.short_edges) == 1:
            choices.append([family.short_edges])
        else:
            choices.append(
                list(itertools.permutations(family.short_edges))
            )
    orders = list(itertools.product(*choices))
    if order_cache is not None:
        order_cache[cache_key] = orders
    return orders


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
    order_cache: dict[tuple, list] | None = None,
    pose_cache: dict[tuple, tuple] | None = None,
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
        best_pose_record = None
        for orders in topology_orders(topology, order_cache):
            pose_key = (signature, orders)
            pose_record = (
                None
                if pose_cache is None
                else pose_cache.get(pose_key)
            )
            if pose_record is None:
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
                _, width, height, rectangle_area = (
                    legacy.minimum_rectangle(placements)
                )
                rough_fill = min(
                    1.0,
                    total_piece_area
                    / max(rectangle_area, 1e-9),
                )
                seam_residual = pose_residuals(initial, point_pairs)
                seam_rms = math.sqrt(
                    float(np.mean(seam_residual ** 2))
                )
                pose_record = (
                    point_pairs,
                    initial,
                    placements,
                    width,
                    height,
                    rectangle_area,
                    rough_fill,
                    seam_rms,
                )
            else:
                (
                    point_pairs,
                    initial,
                    placements,
                    width,
                    height,
                    rectangle_area,
                    rough_fill,
                    seam_rms,
                ) = pose_record
            if exact_overlap:
                rough_fill, overlap_ratio, width, height = (
                    layout_metrics(
                        placements,
                        (width, height, rectangle_area),
                        total_piece_area,
                    )
                )
            else:
                overlap_ratio = 0.0
            dimension_penalty = 0.0 if legal_dimensions(
                width,
                height,
                texture_context,
                any(
                    family.is_chain_to_chain
                    for family in topology
                ),
            ) else 0.25
            score = (
                (1.0 - rough_fill)
                + (1.4 * overlap_ratio if exact_overlap else 0.0)
                + 0.035 * min(seam_rms, 3.0)
                + dimension_penalty
            )
            # Keep the first pass genuinely coarse.  Sampling source pixels
            # for every one of the 700+ topology/order combinations dominated
            # Maix runtime (about five seconds in capture 180811) and caused
            # the ranker to hit its deadline before reaching the valid
            # topology.  Texture remains active in the exact-overlap 80 -> 8
            # pass and in final candidate verification, where it is useful
            # and bounded.
            if exact_overlap:
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
                best_pose_record = pose_record
        if math.isfinite(best_score):
            if (
                pose_cache is not None
                and best_orders is not None
                and best_pose_record is not None
            ):
                pose_cache[
                    (signature, best_orders)
                ] = best_pose_record
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
    # 每个阶段有独立截止时间，确保拓扑枚举不会耗尽预算而跳过姿态优化。
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
        if family.is_chain_to_chain:
            orders = [
                side_a + side_b
                for side_a in itertools.permutations(
                    family.chain_a_edges
                )
                for side_b in itertools.permutations(
                    family.chain_b_edges
                )
            ]
        else:
            orders = itertools.permutations(family.short_edges)
        for order in orders:
            pair_cache[(family.signature, order)] = tuple(
                seam_point_pairs(centered, family, order)
            )
    families_finished = pytime.monotonic()
    full_topologies, strict_topology_count = enumerate_topologies(
        centered,
        families,
        deadline=enumeration_deadline,
        return_strict_count=True,
        texture_context=texture_context,
        source_pieces=original,
    )
    enumeration_rank_by_signature = {
        topology_signature(topology): rank
        for rank, topology in enumerate(full_topologies)
    }
    chain_rank_by_signature = {
        topology_signature(topology): rank
        for rank, topology in enumerate(
            [
                item
                for item in full_topologies
                if any(
                    family.split_count > 0
                    for family in item
                )
            ]
        )
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
    order_cache: dict[tuple, list] = {}
    pose_cache: dict[tuple, tuple] = {}
    coarse_seconds = 0.0
    overlap_seconds = 0.0
    pose_seconds = 0.0
    coarse_deadline_hit = False
    overlap_deadline_hit = False
    evaluated_total = 0
    candidates: list[LayoutCandidate] = []
    optimized_signatures = set()
    topologies: list[tuple[SeamFamily, ...]] = []
    chain_priority_topology_count = 0
    two_by_two_grid_topology_count = 0
    four_family_cycle_topology_count = 0
    card_search_pool_count = 0
    card_long_direct_topology_count = 0
    early_accepted = False
    fast_path_accepted = False
    early_accept_reason = None
    last_search_pass = "none"

    def recommended_candidate_is_decisive(
        batch_candidates: list[LayoutCandidate],
    ) -> bool:
        nonlocal early_accept_reason
        grid_candidates = [
            candidate
            for candidate in batch_candidates
            if (
                candidate.uses_two_by_two_grid
                and candidate.iou >= TWO_BY_TWO_GRID_MIN_IOU
                and candidate.overlap_ratio <= 0.02
                and candidate.seam_rms_cm <= 0.25
                and candidate.layout_perimeter_confidence >= 0.60
                and candidate.layout_perimeter_worst_side_score <= 0.55
                and candidate.layout_contact_segments >= 3
            )
        ]
        if grid_candidates:
            early_accept_reason = "two_by_two_grid_quality_gate"
            return True
        high_quality_card_candidates = [
            candidate
            for candidate in batch_candidates
            if (
                white_card_mode(texture_context)
                and candidate.iou >= CARD_EARLY_ACCEPT_IOU
                and candidate.overlap_ratio
                <= CARD_EARLY_ACCEPT_MAX_OVERLAP_RATIO
                and candidate.seam_rms_cm
                <= CARD_EARLY_ACCEPT_MAX_SEAM_RMS_CM
                and candidate.texture_confidence
                >= CARD_EARLY_ACCEPT_MIN_TEXTURE_CONFIDENCE
                and candidate.layout_perimeter_confidence >= 0.75
                and candidate.layout_perimeter_worst_side_score <= 0.45
                and candidate.layout_contact_segments
                >= CARD_EARLY_ACCEPT_MIN_CONTACT_SEGMENTS
            )
        ]
        if high_quality_card_candidates:
            early_accept_reason = "absolute_card_quality_gate"
            return True
        absolute_q12_candidates = [
            candidate
            for candidate in batch_candidates
            if (
                not white_card_mode(texture_context)
                and question_1_2_dimensions(
                    candidate.solution.width_cm,
                    candidate.solution.height_cm,
                )
                and candidate.iou >= Q12_ABSOLUTE_EARLY_IOU
                and candidate.overlap_ratio
                <= Q12_ABSOLUTE_EARLY_MAX_OVERLAP_RATIO
                and candidate.seam_rms_cm
                <= Q12_ABSOLUTE_EARLY_MAX_SEAM_RMS_CM
                and candidate.layout_contact_segments
                >= Q12_ABSOLUTE_EARLY_MIN_CONTACT_SEGMENTS
            )
        ]
        if absolute_q12_candidates:
            early_accept_reason = "absolute_q12_geometry_gate"
            return True
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
        if geometry_decisive and not white_card_mode(texture_context):
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
            for orders in topology_orders(topology, order_cache):
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
            pose_record = pose_cache.get(
                (topology_signature(topology), orders)
            )
            if pose_record is None:
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
            else:
                point_pairs, initial = pose_record[:2]
            poses, seam_rms = optimize_poses(
                initial,
                point_pairs,
                anchor_id,
                (
                    CHAIN_POSE_OPTIMIZATION_ITERATIONS
                    if any(
                        family.is_chain_to_chain
                        for family in topology
                    )
                    else POSE_OPTIMIZATION_ITERATIONS
                ),
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
                    any(
                        family.is_chain_to_chain
                        for family in topology
                    ),
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
                    + texture_rank_weight(texture_context)
                    * texture_score
                    * texture_confidence
                )
                layout_perimeter_worst_side_score = float(
                    texture_result.get(
                        "perimeter_worst_side_score",
                        1.0,
                    )
                )
                layout_perimeter_worst_edge_score = float(
                    texture_result.get(
                        "perimeter_worst_edge_score",
                        1.0,
                    )
                )
                # Keep every geometrically complete rectangle for the
                # artwork/contact comparison.  Local white-frame defects are
                # a soft preference, not proof that the topology is wrong.
                score += (
                    CARD_WORST_SIDE_PENALTY_WEIGHT
                    * max(
                        0.0,
                        layout_perimeter_worst_side_score
                        - CARD_MAX_WORST_SIDE_SCORE,
                    )
                    + CARD_WORST_EDGE_PENALTY_WEIGHT
                    * max(
                        0.0,
                        layout_perimeter_worst_edge_score
                        - CARD_MAX_WORST_EDGE_SCORE,
                    )
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
                        layout_perimeter_worst_side_score=(
                            layout_perimeter_worst_side_score
                        ),
                        layout_perimeter_worst_edge_score=(
                            layout_perimeter_worst_edge_score
                        ),
                        layout_contact_segments=layout_contact_segments,
                        layout_variant=layout_variant,
                        topology_signature=topology_signature(
                            topology
                        ),
                        uses_chain_to_chain=any(
                            family.is_chain_to_chain
                            for family in topology
                        ),
                        uses_two_by_two_grid=(
                            len(topology) == 3
                            and any(
                                family.is_chain_to_chain
                                for family in topology
                            )
                        ),
                    )
                )
        return batch_candidates, batch_evaluated, False

    # A white-card topology selected from the geometry/card-border union is
    # not guaranteed to live in the first FAST_STRICT_TOPOLOGIES entries.
    # Capture 20260730_180811 is the minimal counterexample: its correct
    # topology is enumeration rank 151 and reconstructs at IoU=0.974, but the
    # Maix fast pass exhausted the shared coarse/overlap deadlines while
    # examining only the first 120 strict entries.  The subsequent full pass
    # therefore optimized zero new topologies.
    #
    # In card mode run one bounded rank over the complete, already-pruned
    # topology union.  This also applies to sparse number cards: capture
    # 013153's correct topology is the eighth full-pool result, while the
    # fast-prefix pass consumed enough Maix time that only seven full-pool
    # results reached pose optimization.  This does not enlarge any candidate
    # quota or tolerance; it only removes the redundant prefix pass.
    # Geometry-only puzzles retain the lower-latency fast-then-full path.
    if white_card_mode(texture_context) and not fast_only:
        grid_priority = [
            topology
            for topology in full_topologies
            if (
                len(topology) == 3
                and any(
                    family.is_chain_to_chain
                    for family in topology
                )
            )
        ]
        grid_priority_signatures = {
            topology_signature(topology)
            for topology in grid_priority
        }
        two_by_two_grid_topology_count = len(grid_priority)
        chain_priority = [
            topology
            for topology in full_topologies
            if (
                topology_signature(topology)
                not in grid_priority_signatures
                and any(
                family.is_chain_to_chain
                for family in topology
                )
            )
        ]
        chain_priority_signatures = {
            topology_signature(topology)
            for topology in chain_priority
        }
        chain_priority_topology_count = len(chain_priority)
        cycle_priority = [
            topology
            for topology in full_topologies
            if len(topology) == 4
        ]
        cycle_priority_signatures = {
            topology_signature(topology)
            for topology in cycle_priority
        }
        four_family_cycle_topology_count = len(cycle_priority)
        generic_card_topologies = [
            topology
            for topology in full_topologies
            if (
                topology_signature(topology)
                not in chain_priority_signatures
                and topology_signature(topology)
                not in grid_priority_signatures
                and topology_signature(topology)
                not in cycle_priority_signatures
            )
        ]
        generic_card_pool = generic_card_topologies[
            :MAX_CARD_GENERIC_SEARCH_TOPOLOGIES
        ]
        search_passes = []
        if rich_card_mode(texture_context):
            card_topology_pool = chain_priority + generic_card_pool
            long_direct_topologies = []
            long_direct_signatures = set()
            for topology in full_topologies:
                if not any(
                    not family.is_chain_to_chain
                    and family.split_count == 0
                    and family.internal_length_cm
                    >= CARD_LONG_DIRECT_MIN_LENGTH_CM
                    for family in topology
                ):
                    continue
                signature = topology_signature(topology)
                if signature in long_direct_signatures:
                    continue
                long_direct_topologies.append(topology)
                long_direct_signatures.add(signature)
                if (
                    len(long_direct_topologies)
                    >= MAX_CARD_LONG_DIRECT_TOPOLOGIES
                ):
                    break
            if long_direct_topologies:
                card_long_direct_topology_count = len(
                    long_direct_topologies
                )
                search_passes.append(
                    (
                        "card_long_direct",
                        long_direct_topologies,
                        min(
                            len(long_direct_topologies),
                            MAX_COARSE_TOPOLOGIES,
                        ),
                    )
                )
            generic_fast_topologies = generic_card_topologies[
                :FAST_STRICT_TOPOLOGIES
            ]
            if generic_fast_topologies:
                search_passes.append(
                    (
                        "card_generic_fast",
                        generic_fast_topologies,
                        FAST_COARSE_TOPOLOGIES,
                    )
                )
            if grid_priority:
                search_passes.append(
                    (
                        "two_by_two_grid",
                        grid_priority,
                        min(
                            len(grid_priority),
                            MAX_COARSE_TOPOLOGIES,
                        ),
                    )
                )
            if cycle_priority:
                search_passes.append(
                    (
                        "four_contact_cycle",
                        cycle_priority,
                        len(cycle_priority),
                    )
                )
        else:
            # Sparse artwork cannot reliably promote the correct topology.
            # Ranking prefix/grid/full batches separately repeats the most
            # expensive geometry work and can consume the Maix deadline before
            # a valid generic topology (capture 050911: rank 232) is optimized.
            # Rank every topology class together once instead.
            card_topology_pool = (
                grid_priority
                + cycle_priority
                + chain_priority[
                    :MAX_SPARSE_CARD_CHAIN_SEARCH_TOPOLOGIES
                ]
                + generic_card_pool
            )
        card_search_pool_count = len(card_topology_pool)
        search_passes.append(
            ("card_full", card_topology_pool, MAX_COARSE_TOPOLOGIES)
        )
    else:
        search_passes = [
            ("fast", fast_topologies, FAST_COARSE_TOPOLOGIES),
        ]
        if not fast_only:
            search_passes.append(
                ("full", full_topologies, MAX_COARSE_TOPOLOGIES)
            )
    for pass_name, topology_pool, coarse_limit in search_passes:
        last_search_pass = pass_name
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
            order_cache=order_cache,
            pose_cache=pose_cache,
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
            order_cache,
            pose_cache,
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
        if (
            pass_name in (
                "fast",
                "card_generic_fast",
                "two_by_two_grid",
                "four_contact_cycle",
            )
            and batch_early_accepted
        ):
            fast_path_accepted = pass_name == "fast"
            break
        if (
            pass_name == "card_long_direct"
            and batch_early_accepted
            and early_accept_reason == "card_quality_gate"
        ):
            # A long-edge batch exists to rescue distorted geometry.  Do not
            # let rectangle quality alone skip the wider artwork comparison;
            # only a texture-margin decision is strong enough to stop here.
            break
        if pass_name == "full":
            break

    card_border_floor = None
    card_border_rejected = 0
    card_gate_probe = None
    card_gate_guard_used = False
    if white_card_mode(texture_context):
        if candidates:
            probe = min(
                candidates,
                key=lambda candidate: (
                    max(0.0, CARD_FINAL_MIN_IOU - candidate.iou)
                    + max(
                        0.0,
                        candidate.overlap_ratio
                        - CARD_FINAL_MAX_OVERLAP_RATIO,
                    )
                    + max(
                        0.0,
                        CARD_ORIENTATION_MIN_PERIMETER_CONFIDENCE
                        - candidate.layout_perimeter_confidence,
                    ),
                    candidate.score,
                ),
            )
            nominal_failures = []
            probe_minimum_iou = (
                TWO_BY_TWO_GRID_MIN_IOU
                if probe.uses_two_by_two_grid
                else (
                    CHAIN_CARD_MIN_IOU
                    if probe.uses_chain_to_chain
                    else CARD_FINAL_MIN_IOU
                )
            )
            if probe.iou < probe_minimum_iou:
                nominal_failures.append("iou")
            if (
                probe.overlap_ratio
                > CARD_FINAL_MAX_OVERLAP_RATIO
            ):
                nominal_failures.append("overlap")
            if (
                probe.layout_perimeter_confidence
                < CARD_ORIENTATION_MIN_PERIMETER_CONFIDENCE
            ):
                nominal_failures.append("perimeter_confidence")
            card_gate_probe = {
                "iou": round(probe.iou, 6),
                "overlap_ratio": round(
                    probe.overlap_ratio,
                    6,
                ),
                "seam_rms_cm": round(probe.seam_rms_cm, 6),
                "aspect_ratio": round(
                    max(
                        probe.solution.width_cm,
                        probe.solution.height_cm,
                    )
                    / max(
                        min(
                            probe.solution.width_cm,
                            probe.solution.height_cm,
                        ),
                        1e-9,
                    ),
                    6,
                ),
                "uses_chain_to_chain": probe.uses_chain_to_chain,
                "uses_two_by_two_grid": probe.uses_two_by_two_grid,
                "enumeration_rank": enumeration_rank_by_signature.get(
                    probe.topology_signature
                ),
                "chain_enumeration_rank": (
                    chain_rank_by_signature.get(
                        probe.topology_signature
                    )
                    if probe.uses_chain_to_chain
                    else None
                ),
                "perimeter_confidence": round(
                    probe.layout_perimeter_confidence,
                    6,
                ),
                "worst_side_score": round(
                    probe.layout_perimeter_worst_side_score,
                    6,
                ),
                "worst_edge_score": round(
                    probe.layout_perimeter_worst_edge_score,
                    6,
                ),
                "nominal_failures": nominal_failures,
            }
        measurable_card_candidates = [
            candidate
            for candidate in candidates
            if (
                candidate.iou
                >= (
                    TWO_BY_TWO_GRID_MIN_IOU
                    if candidate.uses_two_by_two_grid
                    else (
                        CHAIN_CARD_MIN_IOU
                        if candidate.uses_chain_to_chain
                        else CARD_FINAL_MIN_IOU
                        - CARD_FINAL_IOU_GUARD
                    )
                )
                and candidate.overlap_ratio
                <= (
                    CHAIN_CARD_MAX_OVERLAP_RATIO
                    if candidate.uses_chain_to_chain
                    else CARD_FINAL_MAX_OVERLAP_RATIO
                    + CARD_FINAL_OVERLAP_GUARD
                )
                and candidate.layout_perimeter_confidence
                >= (
                    CHAIN_CARD_MIN_PERIMETER_CONFIDENCE
                    if candidate.uses_chain_to_chain
                    else (
                        CARD_ORIENTATION_MIN_PERIMETER_CONFIDENCE
                        - CARD_PERIMETER_CONFIDENCE_GUARD
                    )
                )
            )
        ]
        if measurable_card_candidates:
            card_gate_guard_used = any(
                candidate.iou < CARD_FINAL_MIN_IOU
                or candidate.overlap_ratio
                > CARD_FINAL_MAX_OVERLAP_RATIO
                or candidate.layout_perimeter_confidence
                < CARD_ORIENTATION_MIN_PERIMETER_CONFIDENCE
                for candidate in measurable_card_candidates
            )
            card_border_floor = min(
                candidate.layout_perimeter_worst_side_score
                for candidate in measurable_card_candidates
            )
            # Do not collapse the candidate set to the single cleanest white
            # border.  Multiple rectangles are expected; direct pattern
            # continuity across their physical contacts selects the answer.
            verified_card_candidates = [
                candidate
                for candidate in measurable_card_candidates
                if (
                    candidate.layout_perimeter_worst_side_score
                    <= CARD_ABSOLUTE_MAX_WORST_SIDE_SCORE
                    and candidate.layout_perimeter_worst_edge_score
                    <= CARD_ABSOLUTE_MAX_WORST_EDGE_SCORE
                )
            ]
            card_border_rejected = (
                len(candidates) - len(verified_card_candidates)
            )
            candidates = verified_card_candidates
        else:
            card_border_rejected = len(candidates)
            candidates = []

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
        "version": "v5.24-long-direct-card",
        "seam_family_count": len(families),
        "chain_to_chain_family_count": sum(
            family.is_chain_to_chain for family in families
        ),
        "enumerated_topology_count": enumerated_topology_count,
        "topology_selection": (
            "geometry_plus_source_pose_plus_card_border"
            if rich_card_mode(texture_context)
            else (
                "geometry_plus_sparse_card_border"
                if white_card_mode(texture_context)
                else "geometry"
            )
        ),
        "geometry_topology_quota": (
            WHITE_CARD_GEOMETRY_TOPOLOGIES
            if rich_card_mode(texture_context)
            else MAX_ENUMERATED_TOPOLOGIES
        ),
        "sparse_card_border_rank_weight": (
            SPARSE_CARD_BORDER_RANK_WEIGHT
            if (
                white_card_mode(texture_context)
                and not rich_card_mode(texture_context)
            )
            else 0.0
        ),
        "three_family_topology_reserve_quota": (
            MAX_THREE_FAMILY_RESERVE_TOPOLOGIES
        ),
        "chain_to_chain_topology_reserve_quota": (
            MAX_CHAIN_TO_CHAIN_TOPOLOGIES
        ),
        "chain_to_chain_geometry_quota": (
            MAX_CHAIN_TO_CHAIN_GEOMETRY_TOPOLOGIES
        ),
        "chain_to_chain_source_pose_quota": (
            MAX_CHAIN_TO_CHAIN_SOURCE_TOPOLOGIES
        ),
        "chain_priority_topology_count": (
            chain_priority_topology_count
        ),
        "two_by_two_grid_topology_count": (
            two_by_two_grid_topology_count
        ),
        "four_family_cycle_topology_count": (
            four_family_cycle_topology_count
        ),
        "card_search_pool_count": card_search_pool_count,
        "card_long_direct_topology_count": (
            card_long_direct_topology_count
        ),
        "card_border_topology_quota": (
            WHITE_CARD_BORDER_TOPOLOGIES
            if rich_card_mode(texture_context)
            else 0
        ),
        "source_pose_topology_quota": (
            WHITE_CARD_SOURCE_POSE_TOPOLOGIES
            if rich_card_mode(texture_context)
            else 0
        ),
        "card_neighbor_topology_quota": (
            WHITE_CARD_NEIGHBOR_TOPOLOGIES
            if rich_card_mode(texture_context)
            else 0
        ),
        "card_border_worst_side_floor": (
            None
            if card_border_floor is None
            else round(card_border_floor, 6)
        ),
        "card_border_rejected_layouts": card_border_rejected,
        "card_gate_guard_band_used": card_gate_guard_used,
        "card_gate_nearest_candidate": card_gate_probe,
        "optimized_topology_count": len(optimized_signatures),
        "evaluated_layouts": evaluated_total,
        "search_mode": (
            "fast"
            if fast_path_accepted
            else (
                "fast_only"
                if fast_only
                else (
                    last_search_pass
                    if white_card_mode(texture_context)
                    else "full"
                )
            )
        ),
        "card_texture_mode": (
            "rich"
            if rich_card_mode(texture_context)
            else (
                "sparse"
                if white_card_mode(texture_context)
                else "none"
            )
        ),
        "texture_rank_weight": texture_rank_weight(texture_context),
        "texture_richness": round(
            float(
                0.0
                if texture_context is None
                else texture_context.texture_richness
            ),
            6,
        ),
        "ink_fraction": round(
            float(
                0.0
                if texture_context is None
                else texture_context.ink_fraction
            ),
            6,
        ),
        "dark_ink_fraction": round(
            float(
                0.0
                if texture_context is None
                else texture_context.dark_ink_fraction
            ),
            6,
        ),
        "very_dark_ink_fraction": round(
            float(
                0.0
                if texture_context is None
                else texture_context.very_dark_ink_fraction
            ),
            6,
        ),
        "vivid_print_fraction": round(
            float(
                0.0
                if texture_context is None
                else texture_context.vivid_print_fraction
            ),
            6,
        ),
        "white_material_fraction": round(
            float(
                0.0
                if texture_context is None
                else texture_context.white_material_fraction
            ),
            6,
        ),
        "chromatic_fraction": round(
            float(
                0.0
                if texture_context is None
                else texture_context.chromatic_fraction
            ),
            6,
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
        "best_layout_perimeter_worst_side_score": (
            None
            if best is None
            else round(
                best.layout_perimeter_worst_side_score,
                6,
            )
        ),
        "best_layout_perimeter_worst_edge_score": (
            None
            if best is None
            else round(
                best.layout_perimeter_worst_edge_score,
                6,
            )
        ),
        "best_layout_variant": (
            None if best is None else best.layout_variant
        ),
        "best_layout_contact_segments": (
            None
            if best is None
            else best.layout_contact_segments
        ),
        "orientation_refined_by": (
            None
        ),
        "best_orientation_candidates": [],
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
        "best_chain_enumeration_rank": (
            None
            if best is None
            else chain_rank_by_signature.get(
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
        best.iou
        < (
            TWO_BY_TWO_GRID_MIN_IOU
            if best.uses_two_by_two_grid
            else MIN_RECTANGLE_IOU
        )
        or best.overlap_ratio > MAX_OVERLAP_RATIO
        or best.seam_rms_cm > MAX_SEAM_RMS_CM
    ):
        return legacy_fallback("global_quality_gate")
    if (
        white_card_mode(texture_context)
        and (
            best.iou < (
                TWO_BY_TWO_GRID_MIN_IOU
                if best.uses_two_by_two_grid
                else (
                    CHAIN_CARD_MIN_IOU
                    if best.uses_chain_to_chain
                    else CARD_FINAL_MIN_IOU - CARD_FINAL_IOU_GUARD
                )
            )
            or best.overlap_ratio
            > (
                CHAIN_CARD_MAX_OVERLAP_RATIO
                if best.uses_chain_to_chain
                else (
                    CARD_FINAL_MAX_OVERLAP_RATIO
                    + CARD_FINAL_OVERLAP_GUARD
                )
            )
            or best.seam_rms_cm
            > (
                CHAIN_CARD_MAX_SEAM_RMS_CM
                if best.uses_chain_to_chain
                else TEXTURE_ACCEPT_MAX_SEAM_RMS_CM
            )
            or best.layout_perimeter_worst_side_score
            > CARD_ABSOLUTE_MAX_WORST_SIDE_SCORE
            or best.layout_perimeter_worst_edge_score
            > CARD_ABSOLUTE_MAX_WORST_EDGE_SCORE
        )
    ):
        # Pattern evidence may rank already plausible card layouts, but it
        # must never make a mechanically poor rectangle executable.
        _LAST_DIAGNOSTICS["fallback"] = (
            "disabled_for_textured_geometry"
        )
        _LAST_DIAGNOSTICS["fallback_reason"] = (
            "card_geometry_quality_gate"
        )
        _LAST_DIAGNOSTICS["fallback_nodes"] = 0
        return None, evaluated_total
    card_runner_up_eligible = bool(
        second is not None
        and (
            not white_card_mode(texture_context)
            or (
                second.iou >= TEXTURE_ACCEPT_IOU
                and second.overlap_ratio
                <= TEXTURE_ACCEPT_MAX_OVERLAP_RATIO
                and second.seam_rms_cm
                <= TEXTURE_ACCEPT_MAX_SEAM_RMS_CM
                and second.texture_confidence
                >= TEXTURE_MIN_CONFIDENCE
            )
        )
    )
    if (
        second is not None
        and white_card_mode(texture_context)
        and second.score - best.score < UNIQUE_SCORE_MARGIN
        and second.iou >= MIN_RECTANGLE_IOU
        and not card_runner_up_eligible
    ):
        _LAST_DIAGNOSTICS["ambiguity_runner_up_rejected"] = (
            "below_texture_geometry_floor"
        )
    if (
        second is not None
        and second.score - best.score < UNIQUE_SCORE_MARGIN
        and second.iou >= MIN_RECTANGLE_IOU
        # A weaker card-shaped runner-up must not turn an otherwise verified
        # card into an ambiguity.  The texture comparison below requires
        # TEXTURE_ACCEPT_IOU, but the former outer condition admitted any
        # generic 0.89-IoU rectangle.  Capture 012106 therefore discarded a
        # 0.962-IoU, low-overlap card because its 0.935-IoU alternative was
        # numerically close in the mixed score.
        and card_runner_up_eligible
        and best.iou < 0.97
        and not (
            white_card_mode(texture_context)
            and early_accepted
            # Every card-specific early gate already verifies rectangle,
            # overlap, seam, border and contact quality.  The former code
            # exempted only the score-margin gate, then re-rejected an
            # equally verified 2x2-grid candidate as ambiguous.  Keep the
            # artwork-orientation refinement below, but do not send these
            # protected card candidates to geometry-only DFS.
            and early_accept_reason in (
                "two_by_two_grid_quality_gate",
                "absolute_card_quality_gate",
                "card_quality_gate",
            )
        )
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
        card_components_resolved = (
            geometry_safe_for_texture
            and texture_context is not None
            and texture_context.white_card_confidence
            >= WHITE_CARD_MODE_CONFIDENCE
            and score_margin >= CARD_COMPONENT_SCORE_MARGIN
            and best.layout_contact_segments
            >= minimum_direct_contacts
            and best.layout_seam_score
            <= second.layout_seam_score + CARD_COMPONENT_EPSILON
            and best.layout_symmetry_score
            <= second.layout_symmetry_score + CARD_COMPONENT_EPSILON
            and best.layout_perimeter_score
            <= second.layout_perimeter_score + CARD_COMPONENT_EPSILON
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
        elif card_components_resolved:
            _LAST_DIAGNOSTICS[
                "ambiguity_resolved_by"
            ] = "card_component_consensus"
        else:
            _LAST_DIAGNOSTICS["ambiguous"] = True
            # Rectangle assembly is the primary task.  If multiple card
            # topologies survive every mechanical and white-border gate,
            # keep the best mixed-score rectangle even when artwork cannot
            # uniquely separate it from the runner-up.  Texture remains a
            # ranking penalty, but may not turn a safe rectangle into a hard
            # failure.  The strict geometry gate above is intentionally not
            # relaxed by this fallback.
            if (
                texture_context is not None
                and best.texture_confidence
                >= TEXTURE_MIN_CONFIDENCE
            ):
                _LAST_DIAGNOSTICS["fallback"] = (
                    "geometry_card_fallback"
                )
                _LAST_DIAGNOSTICS["fallback_reason"] = (
                    "texture_ambiguity"
                )
                _LAST_DIAGNOSTICS["fallback_nodes"] = 0
                _LAST_DIAGNOSTICS[
                    "texture_fallback_used"
                ] = True
            else:
                return legacy_fallback("ambiguous_global_candidate")

    if white_card_mode(texture_context):
        # The topology has now passed all rectangle, ambiguity and texture
        # gates.  Only at this point compare the bounded card artwork
        # variants.  Its score may choose the orientation inside this one
        # topology, but is deliberately not allowed to reopen topology
        # ranking or turn a protected early accept into a false ambiguity.
        topology_score = best.score
        best, variant_diagnostics = refine_card_column_orientation(
            best,
            original,
            texture_context,
        )
        _LAST_DIAGNOSTICS.update(
            {
                "topology_score_before_orientation": round(
                    topology_score,
                    6,
                ),
                "best_score": round(best.score, 6),
                "best_iou": round(best.iou, 6),
                "best_overlap_ratio": round(
                    best.overlap_ratio,
                    6,
                ),
                "best_texture_score": round(
                    best.texture_score,
                    6,
                ),
                "best_texture_confidence": round(
                    best.texture_confidence,
                    6,
                ),
                "best_source_texture_score": round(
                    best.source_texture_score,
                    6,
                ),
                "best_layout_texture_score": round(
                    best.layout_texture_score,
                    6,
                ),
                "best_layout_seam_score": round(
                    best.layout_seam_score,
                    6,
                ),
                "best_layout_symmetry_score": round(
                    best.layout_symmetry_score,
                    6,
                ),
                "best_layout_perimeter_score": round(
                    best.layout_perimeter_score,
                    6,
                ),
                "best_layout_perimeter_confidence": round(
                    best.layout_perimeter_confidence,
                    6,
                ),
                "best_layout_variant": best.layout_variant,
                "best_layout_contact_segments": (
                    best.layout_contact_segments
                ),
                "orientation_refined_by": (
                    "direct_card_artwork"
                ),
                "best_orientation_candidates": (
                    variant_diagnostics
                ),
            }
        )
    best.solution.search_nodes = evaluated_total
    return best.solution, evaluated_total


def solve_geometry(
    pieces: list[np.ndarray],
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    texture_context: texture_matcher.TextureContext | None = None,
) -> tuple[Solution | None, int]:
    """在时间预算内完成拓扑搜索、姿态优化和候选排序。

    白色扑克牌和纸板碎片使用不同尺寸门控；失败时保留诊断并尝试兼容回退。
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
    # gate above. Give V3 nearly the complete budget and retain only a short
    # final slice for edge DFS; unlike the former last-millisecond fallback,
    # its result still passes the same artwork and white-perimeter checks.
    # Brown cardboard uses only the specified Question 1/2 size band.
    if white_card_mode(texture_context):
        reserve_seconds = (
            CARD_LEGACY_RESERVE_SECONDS
            if max_seconds is None or max_seconds <= 0
            else min(
                CARD_LEGACY_RESERVE_SECONDS,
                max(0.50, max_seconds * 0.50),
            )
        )
        global_seconds = (
            max_seconds
            if max_seconds is None or max_seconds <= 0
            else max(0.05, max_seconds - reserve_seconds)
        )
        card_solution, card_nodes = _solve_geometry_once(
            pieces,
            max_seconds=global_seconds,
            allow_legacy_fallback=False,
            texture_context=texture_context,
        )
        _LAST_DIAGNOSTICS["v3_budget_seconds"] = (
            None
            if global_seconds is None or global_seconds <= 0
            else round(global_seconds, 3)
        )
        _LAST_DIAGNOSTICS["dfs_reserve_seconds"] = round(
            reserve_seconds,
            3,
        )
        global_diagnostics = dict(_LAST_DIAGNOSTICS)
        if card_solution is None:
            remaining_seconds = (
                reserve_seconds
                if deadline is None
                else max(0.05, deadline - pytime.monotonic())
            )
            legacy_solution, legacy_nodes = legacy.solve_geometry(
                normalized_pieces,
                max_seconds=remaining_seconds,
            )
            card_nodes += legacy_nodes
            _LAST_DIAGNOSTICS["fallback"] = (
                "reserved_legacy_card_dfs"
            )
            _LAST_DIAGNOSTICS["fallback_reason"] = (
                global_diagnostics.get("fallback_reason")
                or "global_no_executable_candidate"
            )
            _LAST_DIAGNOSTICS["fallback_nodes"] = legacy_nodes
            _LAST_DIAGNOSTICS["fallback_reserved_seconds"] = round(
                reserve_seconds,
                3,
            )
            if legacy_solution is not None:
                legacy_candidate = card_candidate_from_solution(
                    legacy_solution,
                    normalized_pieces,
                    texture_context,
                )
                (
                    legacy_candidate,
                    orientation_diagnostics,
                ) = refine_card_column_orientation(
                    legacy_candidate,
                    normalized_pieces,
                    texture_context,
                )
                executable = card_candidate_is_executable(
                    legacy_candidate
                )
                _LAST_DIAGNOSTICS[
                    "fallback_card_validation"
                ] = {
                    "accepted": executable,
                    "iou": round(legacy_candidate.iou, 6),
                    "overlap_ratio": round(
                        legacy_candidate.overlap_ratio,
                        6,
                    ),
                    "texture_confidence": round(
                        legacy_candidate.texture_confidence,
                        6,
                    ),
                    "perimeter_confidence": round(
                        legacy_candidate.layout_perimeter_confidence,
                        6,
                    ),
                    "worst_side_score": round(
                        legacy_candidate
                        .layout_perimeter_worst_side_score,
                        6,
                    ),
                    "worst_edge_score": round(
                        legacy_candidate
                        .layout_perimeter_worst_edge_score,
                        6,
                    ),
                    "variant": legacy_candidate.layout_variant,
                }
                _LAST_DIAGNOSTICS[
                    "best_orientation_candidates"
                ] = orientation_diagnostics
                if executable:
                    legacy_candidate.solution.search_nodes = card_nodes
                    card_solution = legacy_candidate.solution
                    _LAST_DIAGNOSTICS.update(
                        {
                            "best_score": round(
                                legacy_candidate.score,
                                6,
                            ),
                            "best_iou": round(
                                legacy_candidate.iou,
                                6,
                            ),
                            "best_overlap_ratio": round(
                                legacy_candidate.overlap_ratio,
                                6,
                            ),
                            "best_seam_rms_cm": 0.0,
                            "best_texture_score": round(
                                legacy_candidate.texture_score,
                                6,
                            ),
                            "best_texture_confidence": round(
                                legacy_candidate.texture_confidence,
                                6,
                            ),
                            "best_layout_seam_score": round(
                                legacy_candidate.layout_seam_score,
                                6,
                            ),
                            "best_layout_symmetry_score": round(
                                legacy_candidate
                                .layout_symmetry_score,
                                6,
                            ),
                            "best_layout_perimeter_score": round(
                                legacy_candidate
                                .layout_perimeter_score,
                                6,
                            ),
                            "best_layout_perimeter_confidence": round(
                                legacy_candidate
                                .layout_perimeter_confidence,
                                6,
                            ),
                            "best_layout_perimeter_worst_side_score": (
                                round(
                                    legacy_candidate
                                    .layout_perimeter_worst_side_score,
                                    6,
                                )
                            ),
                            "best_layout_perimeter_worst_edge_score": (
                                round(
                                    legacy_candidate
                                    .layout_perimeter_worst_edge_score,
                                    6,
                                )
                            ),
                            "best_layout_variant": (
                                legacy_candidate.layout_variant
                            ),
                            "best_layout_contact_segments": (
                                legacy_candidate
                                .layout_contact_segments
                            ),
                            "orientation_refined_by": (
                                "reserved_dfs_direct_card_artwork"
                            ),
                        }
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

    # Reserve time for the proven edge-DFS fallback.  On MaixCAM, a difficult
    # but legal four-piece layout can exhaust global ranking without leaving
    # enough time for DFS to recover the same strict Q1/Q2 rectangle.
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

        # A broader candidate pool is needed to recover valid topologies whose
        # cut-edge perimeter is biased before pose optimization.  It is search
        # only: the returned layout still passes the hard Q1/Q2 dimensions.
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
        candidate_solution, candidate_nodes = _solve_geometry_once(
            pieces,
            max_seconds=remaining,
            texture_context=texture_context,
        )
        strict_solution = (
            candidate_solution
            if candidate_solution is not None
            and question_1_2_dimensions(
                candidate_solution.width_cm,
                candidate_solution.height_cm,
            )
            else None
        )
        sparse_card_recovered = False
        sparse_print_evidence = bool(
            texture_context is not None
            and texture_context.white_material_fraction
            >= SPARSE_CARD_RECOVERY_MIN_WHITE_FRACTION
            and (
                (
                    texture_context.white_card_confidence
                    >= SPARSE_CARD_RECOVERY_MIN_CONFIDENCE
                    and texture_context.ink_fraction
                    >= SPARSE_CARD_RECOVERY_MIN_INK_FRACTION
                )
                or texture_context.very_dark_ink_fraction
                >= SPARSE_CARD_RECOVERY_MIN_VERY_DARK_FRACTION
                or texture_context.vivid_print_fraction
                >= SPARSE_CARD_RECOVERY_MIN_VIVID_PRINT_FRACTION
            )
        )
        if (
            candidate_solution is not None
            and texture_context is not None
            and sparse_print_evidence
        ):
            short_side = min(
                candidate_solution.width_cm,
                candidate_solution.height_cm,
            )
            long_side = max(
                candidate_solution.width_cm,
                candidate_solution.height_cm,
            )
            aspect_ratio = long_side / max(short_side, 1e-9)
            if (
                SPARSE_CARD_RECOVERY_MIN_SHORT_CM
                <= short_side
                <= SPARSE_CARD_RECOVERY_MAX_SHORT_CM
                and SPARSE_CARD_RECOVERY_MIN_LONG_CM
                <= long_side
                <= SPARSE_CARD_RECOVERY_MAX_LONG_CM
                and SPARSE_CARD_RECOVERY_MIN_ASPECT_RATIO
                <= aspect_ratio
                <= SPARSE_CARD_RECOVERY_MAX_ASPECT_RATIO
            ):
                recovered_candidate = card_candidate_from_solution(
                    candidate_solution,
                    normalized_pieces,
                    texture_context,
                )
                recovered_candidate.layout_variant = (
                    "sparse_card_recovery"
                )
                recovered_candidate.topology_signature = (
                    "sparse_card_recovery",
                )
                (
                    recovered_candidate,
                    orientation_diagnostics,
                ) = refine_card_column_orientation(
                    recovered_candidate,
                    normalized_pieces,
                    texture_context,
                )
                if card_candidate_is_executable(recovered_candidate):
                    strict_solution = recovered_candidate.solution
                    sparse_card_recovered = True
                    _LAST_DIAGNOSTICS.update(
                        {
                            "card_texture_mode": "sparse_recovery",
                            "sparse_card_recovery": True,
                            "sparse_card_recovery_aspect_ratio": round(
                                aspect_ratio,
                                6,
                            ),
                            "sparse_card_recovery_very_dark_fraction": round(
                                texture_context.very_dark_ink_fraction,
                                6,
                            ),
                            "sparse_card_recovery_vivid_print_fraction": round(
                                texture_context.vivid_print_fraction,
                                6,
                            ),
                            "best_score": round(
                                recovered_candidate.score,
                                6,
                            ),
                            "best_iou": round(
                                recovered_candidate.iou,
                                6,
                            ),
                            "best_overlap_ratio": round(
                                recovered_candidate.overlap_ratio,
                                6,
                            ),
                            "best_texture_score": round(
                                recovered_candidate.texture_score,
                                6,
                            ),
                            "best_texture_confidence": round(
                                recovered_candidate.texture_confidence,
                                6,
                            ),
                            "best_layout_perimeter_confidence": round(
                                recovered_candidate
                                .layout_perimeter_confidence,
                                6,
                            ),
                            "best_layout_perimeter_worst_side_score": round(
                                recovered_candidate
                                .layout_perimeter_worst_side_score,
                                6,
                            ),
                            "best_layout_perimeter_worst_edge_score": round(
                                recovered_candidate
                                .layout_perimeter_worst_edge_score,
                                6,
                            ),
                            "best_layout_variant": (
                                recovered_candidate.layout_variant
                            ),
                            "orientation_refined_by": (
                                "sparse_card_recovery_artwork"
                            ),
                            "best_orientation_candidates": (
                                orientation_diagnostics
                            ),
                        }
                    )
        _LAST_DIAGNOSTICS["size_pass"] = (
            "sparse_card_recovery"
            if sparse_card_recovered
            else "question_1_2_strict"
        )
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
        return strict_solution, core_nodes + candidate_nodes
    finally:
        (
            TARGET_MIN_SHORT_CM,
            TARGET_MAX_SHORT_CM,
            TARGET_MIN_LONG_CM,
            TARGET_MAX_LONG_CM,
        ) = saved_limits
