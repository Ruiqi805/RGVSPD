"""Candidate-constrained semantic prompt views and strict response parsing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class Candidate:
    """One fused-visual candidate exposed to the semantic verifier."""

    class_id: str
    class_name: str
    visual_rank: int
    visual_probability: float
    branch_support: str = "neither"


def _candidate_block(candidates: Sequence[Candidate]) -> str:
    return "\n".join(
        f"{item.visual_rank}. class_id={item.class_id} | "
        f"class_name={item.class_name} | visual_rank={item.visual_rank} | "
        f"visual_prob={item.visual_probability:.6f} | "
        f"branch_support={item.branch_support}"
        for item in candidates
    )


def _schema_tail(candidates: Sequence[Candidate]) -> str:
    ids = ", ".join(item.class_id for item in candidates)
    return (
        "\nReturn only this JSON object:\n"
        '{"best_class_id":"<one candidate class_id>"}\n\n'
        f"best_class_id must be exactly one of: {ids}\n"
        "No markdown. No reasoning. No text outside JSON."
    )


def build_reference_prompt(
    view: str,
    candidates: Sequence[Candidate],
    *,
    dino_prediction: str,
    clip_prediction: str,
    attribute_bank: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Build the paper-facing AttrSeek, AttrBank, or FineDefics prompt.

    Prompts contain predictions and candidates only. Ground-truth labels are never
    accepted by this API, which makes accidental test-label leakage harder.
    """

    if view not in SEMANTIC_VIEWS:
        raise ValueError(f"unknown semantic view {view!r}")
    items = tuple(candidates)
    if not items:
        raise ValueError("candidates must not be empty")
    ids = [item.class_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate class_ids must be unique")
    context = (
        "You are a candidate-constrained visual recognition verifier.\n"
        "DINOv2 is a structural visual expert. CLIP-LPP is a "
        "semantic-aligned visual expert.\n"
        f"DINOv2 prediction: class_id={dino_prediction}\n"
        f"CLIP-LPP prediction: class_id={clip_prediction}\n\n"
        "The experts disagree. Use visible evidence from the supplied image.\n"
        f"Choose only from the fused Top-{len(items)} candidates.\n"
        "If evidence is ambiguous, prefer the higher visual-rank candidate.\n\n"
        f"Candidates:\n{_candidate_block(items)}\n"
    )
    if view == "attrseek":
        task = (
            "\nAttrSeek: look for concrete visible attributes before deciding: "
            "overall shape, distinctive parts, texture, color pattern, material, "
            "posture, and fine-grained object cues. Choose the candidate best "
            "supported by those visible attributes."
        )
    elif view == "attrbank":
        bank = attribute_bank or {}
        bank_lines = []
        for item in items:
            entry = bank.get(item.class_id, {})
            visible = entry.get("visible_attributes", [f"visible features of {item.class_name}"])
            cues = entry.get("discriminative_cues", [f"discriminative cues of {item.class_name}"])
            negatives = entry.get("negative_cues", ["do not decide from background alone"])
            bank_lines.append(
                f"class_id={item.class_id} | visible_attributes: {'; '.join(map(str, visible))} | "
                f"discriminative_cues: {'; '.join(map(str, cues))} | "
                f"negative_cues: {'; '.join(map(str, negatives))}"
            )
        task = (
            "\nAttrBank: compare visible image evidence against each candidate's "
            "attribute-bank entry. Penalize background-only support and reject "
            "candidates contradicted by negative cues.\nAttribute bank:\n"
            + "\n".join(bank_lines)
        )
    else:
        task = (
            "\nFineDefics: treat the candidates as visually similar hard negatives. "
            "Compare expected discriminative attributes with the image, reject "
            "contradicted candidates, and select the class with the strongest "
            "fine-grained visible alignment. Do not rely on class-name plausibility "
            "or prior frequency."
        )
    return context + task + _schema_tail(items)


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
