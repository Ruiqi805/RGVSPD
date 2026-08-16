"""Risk-guided visual-semantic progressive decision logic."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .config import REFERENCE_CONFIG, dino_weight_for_shot

SEMANTIC_VIEWS = ("attrseek", "attrbank", "finedefics")


@dataclass(frozen=True)
class VisualDecision:
    """The visual-stage output required before optional semantic verification."""

    class_ids: tuple[str, ...]
    dino_prediction: str
    clip_prediction: str
    visual_prediction: str
    fused_probabilities: tuple[float, ...]
    routed: bool
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class DecisionResult:
    """Final prediction plus an auditable record of the decision path."""

    prediction: str
    visual_prediction: str
    dino_prediction: str
    clip_prediction: str
    routed: bool
    candidates: tuple[str, ...]
    semantic_votes: tuple[tuple[str, str | None], ...]
    majority_count: int
    reason: str
    fused_probabilities: tuple[float, ...]


def _class_ids(values: Sequence[str]) -> tuple[str, ...]:
    class_ids = tuple(str(value) for value in values)
    if not class_ids:
        raise ValueError("class_ids must not be empty")
    if any(not value for value in class_ids):
        raise ValueError("class_ids must be non-empty strings")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("class_ids must be unique")
    return class_ids


def _normalize(values: Sequence[float], expected_length: int, name: str) -> tuple[float, ...]:
    if len(values) != expected_length:
        raise ValueError(f"{name} has length {len(values)}; expected {expected_length}")
    scores = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in scores):
        raise ValueError(f"{name} must contain only finite values")
    if any(value < 0.0 for value in scores):
        raise ValueError(f"{name} must contain only non-negative values")
    total = sum(scores)
    if total <= 0.0:
        raise ValueError(f"{name} must have a positive sum")
    return tuple(value / total for value in scores)


def fuse_probabilities(
    dino_probabilities: Sequence[float],
    clip_probabilities: Sequence[float],
    dino_weight: float,
) -> tuple[float, ...]:
    """Fuse two aligned class distributions with a convex combination."""

    if len(dino_probabilities) != len(clip_probabilities):
        raise ValueError("DINO and CLIP probability vectors must have equal length")
    alpha = float(dino_weight)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("dino_weight must be a finite value in [0, 1]")
    dino = _normalize(dino_probabilities, len(dino_probabilities), "dino_probabilities")
    clip = _normalize(clip_probabilities, len(clip_probabilities), "clip_probabilities")
    fused = tuple(alpha * p_dino + (1.0 - alpha) * p_clip for p_dino, p_clip in zip(dino, clip))
    total = sum(fused)
    return tuple(value / total for value in fused)


def _rank(class_ids: Sequence[str], probabilities: Sequence[float]) -> tuple[str, ...]:
    indices = sorted(range(len(class_ids)), key=lambda index: (-probabilities[index], index))
    return tuple(class_ids[index] for index in indices)


def prepare_visual_decision(
    *,
    class_ids: Sequence[str],
    dino_probabilities: Sequence[float],
    clip_probabilities: Sequence[float],
    dino_weight: float,
    candidate_count: int = REFERENCE_CONFIG.candidate_count,
) -> VisualDecision:
    """Fuse visual scores, route by Top-1 disagreement, and build candidates."""

    ids = _class_ids(class_ids)
    if int(candidate_count) <= 0:
        raise ValueError("candidate_count must be positive")
    dino = _normalize(dino_probabilities, len(ids), "dino_probabilities")
    clip = _normalize(clip_probabilities, len(ids), "clip_probabilities")
    fused = fuse_probabilities(dino, clip, dino_weight)
    dino_prediction = _rank(ids, dino)[0]
    clip_prediction = _rank(ids, clip)[0]
    ranked = _rank(ids, fused)
    visual_prediction = ranked[0]
    routed = dino_prediction != clip_prediction
    candidates = ranked[: min(int(candidate_count), len(ids))]
    return VisualDecision(
        class_ids=ids,
        dino_prediction=dino_prediction,
        clip_prediction=clip_prediction,
        visual_prediction=visual_prediction,
        fused_probabilities=fused,
        routed=routed,
        candidates=candidates,
    )


def finalize_decision(
    visual: VisualDecision,
    semantic_votes: Mapping[str, str | None] | None = None,
) -> DecisionResult:
    """Apply two-of-three semantic majority or fused-visual fallback."""

    supplied = dict(semantic_votes or {})
    unknown_views = set(supplied) - set(SEMANTIC_VIEWS)
    if unknown_views:
        names = ", ".join(sorted(unknown_views))
        raise ValueError(f"unknown semantic view(s): {names}")
    votes = tuple(
        (view, None if supplied.get(view) is None else str(supplied[view]))
        for view in SEMANTIC_VIEWS
    )

    if not visual.routed:
        return DecisionResult(
            prediction=visual.visual_prediction,
            visual_prediction=visual.visual_prediction,
            dino_prediction=visual.dino_prediction,
            clip_prediction=visual.clip_prediction,
            routed=False,
            candidates=visual.candidates,
            semantic_votes=votes,
            majority_count=0,
            reason="visual_agreement",
            fused_probabilities=visual.fused_probabilities,
        )

    candidate_set = set(visual.candidates)
    valid_votes = [vote for _, vote in votes if vote in candidate_set]
    counts = Counter(valid_votes)
    ranked_counts = sorted(
        counts.items(),
        key=lambda item: (-item[1], visual.candidates.index(item[0])),
    )
    if ranked_counts and ranked_counts[0][1] >= REFERENCE_CONFIG.semantic_majority_threshold:
        prediction, count = ranked_counts[0]
        reason = "semantic_majority"
    else:
        prediction = visual.visual_prediction
        count = ranked_counts[0][1] if ranked_counts else 0
        reason = "visual_fallback_no_majority"

    return DecisionResult(
        prediction=prediction,
        visual_prediction=visual.visual_prediction,
        dino_prediction=visual.dino_prediction,
        clip_prediction=visual.clip_prediction,
        routed=True,
        candidates=visual.candidates,
        semantic_votes=votes,
        majority_count=count,
        reason=reason,
        fused_probabilities=visual.fused_probabilities,
    )


def decide(
    *,
    class_ids: Sequence[str],
    dino_probabilities: Sequence[float],
    clip_probabilities: Sequence[float],
    dino_weight: float,
    semantic_votes: Mapping[str, str | None] | None = None,
    candidate_count: int = REFERENCE_CONFIG.candidate_count,
) -> DecisionResult:
    """Run the complete decision rule from aligned visual probabilities."""

    visual = prepare_visual_decision(
        class_ids=class_ids,
        dino_probabilities=dino_probabilities,
        clip_probabilities=clip_probabilities,
        dino_weight=dino_weight,
        candidate_count=candidate_count,
    )
    return finalize_decision(visual, semantic_votes)


def decide_for_shot(
    *,
    class_ids: Sequence[str],
    dino_probabilities: Sequence[float],
    clip_probabilities: Sequence[float],
    shot: int,
    semantic_votes: Mapping[str, str | None] | None = None,
    candidate_count: int = REFERENCE_CONFIG.candidate_count,
) -> DecisionResult:
    """Run the decision rule with the reference shot-dependent fusion weight."""

    return decide(
        class_ids=class_ids,
        dino_probabilities=dino_probabilities,
        clip_probabilities=clip_probabilities,
        dino_weight=dino_weight_for_shot(shot),
        semantic_votes=semantic_votes,
        candidate_count=candidate_count,
    )
