"""Reference constants for the RG-VSPD-DCQ instantiation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceConfig:
    """Fixed settings used by the reference DINO-CLIP-Qwen instantiation."""

    dino_backbone: str = "DINOv2 ViT-L/14"
    clip_backbone: str = "CLIP ViT-B/16"
    semantic_model: str = "Qwen3.5-4B"
    dino_linear_probe_weight: float = 0.5
    dino_prototype_weight: float = 0.5
    clip_support_prototype_weight: float = 0.12
    clip_text_weight: float = 0.88
    clip_temperature: float = 100.0
    clip_prompt_template_count: int = 13
    candidate_count: int = 5
    semantic_majority_threshold: int = 2
    support_set_seeds: tuple[int, int, int] = (1, 21, 93)
    dino_weight_schedule: tuple[tuple[int, float], ...] = (
        (1, 0.1),
        (2, 0.2),
        (16, 0.5),
    )


REFERENCE_CONFIG = ReferenceConfig()


def dino_weight_for_shot(shot: int) -> float:
    """Return the pre-specified DINO fusion weight for a reported shot setting."""

    schedule = dict(REFERENCE_CONFIG.dino_weight_schedule)
    try:
        return schedule[int(shot)]
    except (KeyError, ValueError) as exc:
        supported = ", ".join(str(value) for value in sorted(schedule))
        raise ValueError(f"unsupported shot setting {shot!r}; choose one of: {supported}") from exc
