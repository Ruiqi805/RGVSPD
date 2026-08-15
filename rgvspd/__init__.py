"""Public API for the RG-VSPD decision core."""

from .config import REFERENCE_CONFIG, ReferenceConfig, dino_weight_for_shot
from .core import (
    SEMANTIC_VIEWS,
    DecisionResult,
    VisualDecision,
    decide,
    decide_for_shot,
    finalize_decision,
    fuse_probabilities,
    prepare_visual_decision,
)
from .prompts import build_semantic_prompt, parse_semantic_response

__all__ = [
    "REFERENCE_CONFIG",
    "SEMANTIC_VIEWS",
    "DecisionResult",
    "ReferenceConfig",
    "VisualDecision",
    "build_semantic_prompt",
    "decide",
    "decide_for_shot",
    "dino_weight_for_shot",
    "finalize_decision",
    "fuse_probabilities",
    "parse_semantic_response",
    "prepare_visual_decision",
]
