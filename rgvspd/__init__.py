"""Public API for the RG-VSPD reference implementation."""

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
from .data import ImageRecord, load_manifest, select_fewshot_support
from .pipeline import PipelineResult, run_pipeline_decision
from .prompts import (
    Candidate,
    build_reference_prompt,
    build_semantic_prompt,
    parse_semantic_response,
)

__all__ = [
    "REFERENCE_CONFIG",
    "SEMANTIC_VIEWS",
    "Candidate",
    "DecisionResult",
    "ImageRecord",
    "PipelineResult",
    "ReferenceConfig",
    "VisualDecision",
    "build_reference_prompt",
    "build_semantic_prompt",
    "decide",
    "decide_for_shot",
    "dino_weight_for_shot",
    "finalize_decision",
    "fuse_probabilities",
    "load_manifest",
    "parse_semantic_response",
    "prepare_visual_decision",
    "run_pipeline_decision",
    "select_fewshot_support",
]
