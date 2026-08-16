"""End-to-end routing from visual probabilities to semantic majority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import REFERENCE_CONFIG, dino_weight_for_shot
from .core import (
    SEMANTIC_VIEWS,
    DecisionResult,
    finalize_decision,
    prepare_visual_decision,
)
from .prompts import Candidate, build_reference_prompt, parse_semantic_response
from .semantic import JsonSemanticCache, semantic_cache_key


class SemanticGenerator(Protocol):
    """Minimal interface implemented by the Qwen verifier and test doubles."""

    model_id_or_path: str

    def generate(self, image_path: str | Path, prompt: str, *, view: str) -> str: ...


@dataclass(frozen=True)
class PipelineResult:
    """Final decision together with semantic audit material."""

    decision: DecisionResult
    prompts: tuple[tuple[str, str], ...]
    raw_responses: tuple[tuple[str, str], ...]


def _semantic_candidates(
    *,
    class_ids: Sequence[str],
    class_names: Mapping[str, str],
    fused_probabilities: Sequence[float],
    candidate_ids: Sequence[str],
    dino_prediction: str,
    clip_prediction: str,
) -> tuple[Candidate, ...]:
    probability_by_id = dict(zip(map(str, class_ids), map(float, fused_probabilities)))
    result = []
    for rank, class_id in enumerate(candidate_ids, start=1):
        support = []
        if class_id == dino_prediction:
            support.append("dino")
        if class_id == clip_prediction:
            support.append("clip")
        result.append(
            Candidate(
                class_id=class_id,
                class_name=class_names.get(class_id, class_id),
                visual_rank=rank,
                visual_probability=probability_by_id[class_id],
                branch_support="both" if len(support) == 2 else support[0] if support else "neither",
            )
        )
    return tuple(result)


def run_pipeline_decision(
    *,
    image_path: str | Path,
    class_ids: Sequence[str],
    class_names: Mapping[str, str],
    dino_probabilities: Sequence[float],
    clip_probabilities: Sequence[float],
    shot: int,
    verifier: SemanticGenerator | None,
    attribute_bank: Mapping[str, Mapping[str, Any]] | None = None,
    semantic_cache: JsonSemanticCache | None = None,
    candidate_count: int = REFERENCE_CONFIG.candidate_count,
) -> PipelineResult:
    """Run disagreement routing and the three-view semantic decision.

    Passing ``verifier=None`` performs a visual-only run while preserving the
    exact router and fallback behavior.
    """

    visual = prepare_visual_decision(
        class_ids=class_ids,
        dino_probabilities=dino_probabilities,
        clip_probabilities=clip_probabilities,
        dino_weight=dino_weight_for_shot(shot),
        candidate_count=candidate_count,
    )
    if not visual.routed or verifier is None:
        return PipelineResult(finalize_decision(visual), (), ())

    candidates = _semantic_candidates(
        class_ids=visual.class_ids,
        class_names=class_names,
        fused_probabilities=visual.fused_probabilities,
        candidate_ids=visual.candidates,
        dino_prediction=visual.dino_prediction,
        clip_prediction=visual.clip_prediction,
    )
    prompts: list[tuple[str, str]] = []
    raw_responses: list[tuple[str, str]] = []
    votes: dict[str, str | None] = {}
    model_id = str(getattr(verifier, "model_id_or_path", type(verifier).__name__))
    for view in SEMANTIC_VIEWS:
        prompt = build_reference_prompt(
            view,
            candidates,
            dino_prediction=visual.dino_prediction,
            clip_prediction=visual.clip_prediction,
            attribute_bank=attribute_bank,
        )
        prompts.append((view, prompt))
        key = semantic_cache_key(
            image_path=image_path,
            prompt=prompt,
            view=view,
            model_id=model_id,
        )
        raw = semantic_cache.get(key) if semantic_cache is not None else None
        if raw is None:
            raw = verifier.generate(image_path, prompt, view=view)
            if semantic_cache is not None:
                semantic_cache.put(key, raw)
        raw_responses.append((view, raw))
        votes[view] = parse_semantic_response(raw, visual.candidates)
    return PipelineResult(finalize_decision(visual, votes), tuple(prompts), tuple(raw_responses))
