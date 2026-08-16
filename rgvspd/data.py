"""Manifest loading and deterministic few-shot support selection."""

from __future__ import annotations

import csv
import hashlib
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    """A single image and its label in a dataset manifest."""

    image_path: Path
    class_id: str
    class_name: str
    split: str
    image_id: str


def load_manifest(path: str | Path, *, require_images: bool = True) -> tuple[ImageRecord, ...]:
    """Load a CSV manifest.

    Required columns are ``image_path``, ``class_id``, and ``split``.
    ``class_name`` and ``image_id`` are optional. Relative image paths are
    resolved against the manifest directory.
    """

    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "class_id", "split"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
        records: list[ImageRecord] = []
        for row_index, row in enumerate(reader, start=2):
            raw_path = str(row.get("image_path", "")).strip()
            class_id = str(row.get("class_id", "")).strip()
            split = str(row.get("split", "")).strip().lower()
            if not raw_path or not class_id or not split:
                raise ValueError(f"manifest row {row_index} has an empty required value")
            image_path = Path(raw_path).expanduser()
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            image_path = image_path.resolve()
            if require_images and not image_path.is_file():
                raise FileNotFoundError(f"manifest row {row_index} image not found: {image_path}")
            class_name = str(row.get("class_name", "")).strip() or class_id
            image_id = str(row.get("image_id", "")).strip() or raw_path.replace("\\", "/")
            records.append(ImageRecord(image_path, class_id, class_name, split, image_id))
    if not records:
        raise ValueError("manifest must contain at least one record")
    return tuple(records)


def records_for_split(records: Iterable[ImageRecord], split: str) -> tuple[ImageRecord, ...]:
    """Return one split in deterministic image-id order."""

    wanted = str(split).strip().lower()
    selected = [record for record in records if record.split == wanted]
    return tuple(sorted(selected, key=lambda item: (item.class_id, item.image_id)))


def class_names(records: Iterable[ImageRecord]) -> dict[str, str]:
    """Build and validate the class-id to class-name mapping."""

    result: dict[str, str] = {}
    for record in records:
        previous = result.setdefault(record.class_id, record.class_name)
        if previous != record.class_name:
            raise ValueError(
                f"class_id {record.class_id!r} has inconsistent names: "
                f"{previous!r} and {record.class_name!r}"
            )
    return result


def _class_seed(seed: int, class_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{class_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def select_fewshot_support(
    records: Sequence[ImageRecord],
    *,
    shot: int,
    seed: int,
    source_split: str = "train",
) -> tuple[ImageRecord, ...]:
    """Select exactly ``shot`` support images per class without test access."""

    count = int(shot)
    if count <= 0:
        raise ValueError("shot must be positive")
    groups: dict[str, list[ImageRecord]] = {}
    for record in records_for_split(records, source_split):
        groups.setdefault(record.class_id, []).append(record)
    if not groups:
        raise ValueError(f"no records found for source split {source_split!r}")
    selected: list[ImageRecord] = []
    for class_id in sorted(groups):
        candidates = groups[class_id]
        if len(candidates) < count:
            raise ValueError(
                f"class_id {class_id!r} has {len(candidates)} {source_split} "
                f"images; {count} required"
            )
        rng = random.Random(_class_seed(seed, class_id))
        chosen_indices = sorted(rng.sample(range(len(candidates)), count))
        selected.extend(candidates[index] for index in chosen_indices)
    return tuple(selected)


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an image file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_disjoint_images(
    support: Sequence[ImageRecord],
    evaluation: Sequence[ImageRecord],
) -> dict[str, object]:
    """Reject path or byte-identical overlap between support and evaluation."""

    support_paths = {record.image_path.resolve() for record in support}
    evaluation_paths = {record.image_path.resolve() for record in evaluation}
    path_overlap = support_paths & evaluation_paths
    if path_overlap:
        raise ValueError(f"support/evaluation path overlap detected: {next(iter(path_overlap))}")
    support_hashes = {file_sha256(path) for path in support_paths}
    evaluation_hashes = {file_sha256(path) for path in evaluation_paths}
    hash_overlap = support_hashes & evaluation_hashes
    if hash_overlap:
        raise ValueError("support/evaluation byte-identical image overlap detected")
    digest = hashlib.sha256("\n".join(sorted(support_hashes)).encode("ascii")).hexdigest()
    return {
        "support_images": len(support_paths),
        "evaluation_images": len(evaluation_paths),
        "support_set_sha256": digest,
        "path_overlap": 0,
        "content_overlap": 0,
    }
