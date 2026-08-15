"""Candidate-constrained semantic prompt views and strict response parsing."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from .core import SEMANTIC_VIEWS


_VIEW_INSTRUCTIONS = {
    "attrseek": (
        "Use only observable image evidence: object structure, local parts, texture, "
        "color patterns, and spatial characteristics. Select the candidate best "
        "supported by visible cues."
    ),
    "attrbank": (
        "Reason about the attributes implied by each candidate class name, then match "
        "those attributes against visible image evidence. Do not rely on unsupported context."
    ),
    "finedefics": (
        "Compare discriminative evidence and counter-evidence between the visually "
        "similar candidates. Reject candidates contradicted by the image."
    ),
}


def build_semantic_prompt(
    view: str,
    candidate_ids: Sequence[str],
    class_names: Mapping[str, str],
) -> str:
    """Build one semantic view; the image is supplied separately to the LVLM."""

    if view not in SEMANTIC_VIEWS:
        raise ValueError(f"unknown semantic view {view!r}")
    ids = tuple(str(value) for value in candidate_ids)
    if not ids:
        raise ValueError("candidate_ids must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique")
    lines = [f"{rank}. class_id={class_id}; class_name={class_names.get(class_id, class_id)}" for rank, class_id in enumerate(ids, 1)]
    candidates = "\n".join(lines)
    return (
        "Choose exactly one class from the candidate list for the supplied query image.\n"
        f"View instruction: {_VIEW_INSTRUCTIONS[view]}\n\n"
        f"Candidates:\n{candidates}\n\n"
        "Return strict JSON only, with no markdown or extra keys:\n"
        '{"best_class_id":"<one exact candidate class_id>"}'
    )


def parse_semantic_response(raw_response: str, candidate_ids: Sequence[str]) -> str | None:
    """Return a valid candidate ID, or ``None`` for malformed/out-of-set output."""

    try:
        payload = json.loads(str(raw_response))
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"best_class_id"}:
        return None
    value = payload.get("best_class_id")
    if not isinstance(value, str):
        return None
    allowed = {str(candidate_id) for candidate_id in candidate_ids}
    return value if value in allowed else None
