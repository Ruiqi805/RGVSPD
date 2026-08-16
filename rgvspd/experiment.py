"""Reference experiment runner from a manifest to predictions and metrics."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .config import REFERENCE_CONFIG, dino_weight_for_shot
from .data import (
    ImageRecord,
    assert_disjoint_images,
    class_names,
    load_manifest,
    records_for_split,
    select_fewshot_support,
)
from .pipeline import run_pipeline_decision
from .semantic import JsonSemanticCache, QwenSemanticVerifier
from .visual import (
    CLIPEncoder,
    DinoV2Encoder,
    clip_lpp_probabilities,
    dino_lp_proto_probabilities,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime settings that do not change the locked method definition."""

    manifest: Path
    output_dir: Path
    shot: int
    seed: int
    eval_split: str = "test"
    support_split: str = "train"
    device: str = "cuda"
    batch_size: int = 16
    linear_epochs: int = 80
    qwen_model: str = REFERENCE_CONFIG.semantic_model
    local_files_only: bool = False
    visual_only: bool = False
    max_eval: int = 0
    attribute_bank: Path | None = None


def _paths(records: Sequence[ImageRecord]) -> tuple[str, ...]:
    return tuple(str(record.image_path) for record in records)


def _cached_features(
    cache_path: Path,
    image_paths: Sequence[str],
    encoder: Callable[[Sequence[str]], np.ndarray],
) -> np.ndarray:
    expected = np.asarray(list(image_paths), dtype=np.str_)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if "paths" in cached and "features" in cached and np.array_equal(cached["paths"], expected):
                return np.asarray(cached["features"], dtype=np.float32)
    features = np.asarray(encoder(image_paths), dtype=np.float32)
    if features.shape[0] != len(image_paths):
        raise RuntimeError(f"encoder returned {features.shape[0]} rows for {len(image_paths)} images")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, paths=expected, features=features)
    return features


def _cached_text_embeddings(
    cache_path: Path,
    class_ids: Sequence[str],
    names: dict[str, str],
    encoder: Callable[[], np.ndarray],
) -> np.ndarray:
    ids = np.asarray(list(class_ids), dtype=np.str_)
    labels = np.asarray([names[class_id] for class_id in class_ids], dtype=np.str_)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if (
                "class_ids" in cached
                and "class_names" in cached
                and "features" in cached
                and np.array_equal(cached["class_ids"], ids)
                and np.array_equal(cached["class_names"], labels)
            ):
                return np.asarray(cached["features"], dtype=np.float32)
    features = np.asarray(encoder(), dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, class_ids=ids, class_names=labels, features=features)
    return features


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_attribute_bank(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("attribute bank must be a JSON object")
    classes = payload.get("classes", payload)
    if not isinstance(classes, dict):
        raise TypeError("attribute bank 'classes' field must be an object")
    return {str(key): value for key, value in classes.items() if isinstance(value, dict)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    temporary.replace(path)


def run_reference_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Execute the locked reference pipeline and return aggregate metrics."""

    if config.shot not in dict(REFERENCE_CONFIG.dino_weight_schedule):
        raise ValueError(f"unsupported shot setting: {config.shot}")
    if config.seed not in REFERENCE_CONFIG.support_set_seeds:
        raise ValueError(f"seed must be one of {REFERENCE_CONFIG.support_set_seeds}")
    records = load_manifest(config.manifest)
    names = class_names(records)
    support = select_fewshot_support(
        records,
        shot=config.shot,
        seed=config.seed,
        source_split=config.support_split,
    )
    evaluation = records_for_split(records, config.eval_split)
    if not evaluation:
        raise ValueError(f"no records found for evaluation split {config.eval_split!r}")
    if config.max_eval > 0:
        evaluation = evaluation[: config.max_eval]
    integrity = assert_disjoint_images(support, evaluation)
    class_ids = tuple(sorted(names))
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    support_labels = np.asarray([class_to_index[item.class_id] for item in support], dtype=np.int64)
    evaluation_labels = np.asarray([class_to_index[item.class_id] for item in evaluation], dtype=np.int64)

    cache_dir = config.output_dir / "feature_cache"
    dino_encoder = DinoV2Encoder(device=config.device)
    support_dino = _cached_features(
        cache_dir / f"dino_support_shot{config.shot}_seed{config.seed}.npz",
        _paths(support),
        partial(dino_encoder.encode, batch_size=config.batch_size),
    )
    evaluation_dino = _cached_features(
        cache_dir / f"dino_{config.eval_split}.npz",
        _paths(evaluation),
        partial(dino_encoder.encode, batch_size=config.batch_size),
    )
    del dino_encoder
    _release_accelerator_memory()

    clip_encoder = CLIPEncoder(device=config.device, local_files_only=config.local_files_only)
    support_clip = _cached_features(
        cache_dir / f"clip_support_shot{config.shot}_seed{config.seed}.npz",
        _paths(support),
        partial(clip_encoder.encode_images, batch_size=config.batch_size),
    )
    evaluation_clip = _cached_features(
        cache_dir / f"clip_{config.eval_split}.npz",
        _paths(evaluation),
        partial(clip_encoder.encode_images, batch_size=config.batch_size),
    )
    text_embeddings = _cached_text_embeddings(
        cache_dir / "clip_text_13_templates.npz",
        class_ids,
        names,
        partial(clip_encoder.encode_class_prompts, class_ids, names),
    )
    del clip_encoder
    _release_accelerator_memory()

    dino_probabilities = dino_lp_proto_probabilities(
        support_dino,
        support_labels,
        evaluation_dino,
        len(class_ids),
        device=config.device,
        epochs=config.linear_epochs,
        seed=config.seed,
    )
    clip_probabilities = clip_lpp_probabilities(
        support_clip,
        support_labels,
        evaluation_clip,
        text_embeddings,
    )
    _release_accelerator_memory()

    verifier = None
    semantic_cache = None
    if not config.visual_only:
        verifier = QwenSemanticVerifier(
            config.qwen_model,
            local_files_only=config.local_files_only,
        )
        semantic_cache = JsonSemanticCache(config.output_dir / "semantic_cache.json")
    attribute_bank = _load_attribute_bank(config.attribute_bank)
    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(evaluation):
        result = run_pipeline_decision(
            image_path=record.image_path,
            class_ids=class_ids,
            class_names=names,
            dino_probabilities=dino_probabilities[index],
            clip_probabilities=clip_probabilities[index],
            shot=config.shot,
            verifier=verifier,
            attribute_bank=attribute_bank,
            semantic_cache=semantic_cache,
        )
        decision = result.decision
        predictions.append(
            {
                "image_id": record.image_id,
                "class_id": record.class_id,
                "prediction": decision.prediction,
                "correct": decision.prediction == record.class_id,
                "dino_prediction": decision.dino_prediction,
                "clip_prediction": decision.clip_prediction,
                "visual_prediction": decision.visual_prediction,
                "routed": decision.routed,
                "reason": decision.reason,
                "candidates": list(decision.candidates),
                "semantic_votes": dict(decision.semantic_votes),
                "majority_count": decision.majority_count,
            }
        )

    total = len(predictions)
    routed = sum(int(row["routed"]) for row in predictions)
    metrics = {
        "num_evaluation_images": total,
        "accuracy": sum(int(row["correct"]) for row in predictions) / total,
        "dino_accuracy": float(np.mean(np.argmax(dino_probabilities, axis=1) == evaluation_labels)),
        "clip_lpp_accuracy": float(np.mean(np.argmax(clip_probabilities, axis=1) == evaluation_labels)),
        "routed_count": routed,
        "routed_fraction": routed / total,
        "shot": config.shot,
        "seed": config.seed,
        "dino_fusion_weight": dino_weight_for_shot(config.shot),
        "visual_only": config.visual_only,
    }
    public_config = asdict(config)
    public_config["manifest"] = str(config.manifest)
    public_config["output_dir"] = str(config.output_dir)
    public_config["attribute_bank"] = str(config.attribute_bank) if config.attribute_bank else None
    _write_json(config.output_dir / "run_config.json", public_config)
    _write_json(
        config.output_dir / "support_selection.json",
        {
            "shot": config.shot,
            "seed": config.seed,
            "images": [
                {"image_id": record.image_id, "class_id": record.class_id}
                for record in support
            ],
        },
    )
    _write_json(config.output_dir / "split_integrity.json", integrity)
    _write_json(config.output_dir / "metrics.json", metrics)
    _write_json(config.output_dir / "predictions.json", predictions)
    return metrics
