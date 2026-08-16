"""Qwen3.5 semantic verification and resumable response caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import REFERENCE_CONFIG


def semantic_cache_key(
    *,
    image_path: str | Path,
    prompt: str,
    view: str,
    model_id: str,
) -> str:
    """Hash all inputs that can change a semantic judgment."""

    path = Path(image_path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    payload = {
        "image_sha256": digest.hexdigest(),
        "prompt": prompt,
        "view": str(view),
        "model_id": str(model_id),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class JsonSemanticCache:
    """Small atomic JSON cache used to resume long semantic runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("semantic cache root must be a JSON object")
            self.entries: dict[str, str] = {str(key): str(value) for key, value in payload.items()}
        else:
            self.entries = {}

    def get(self, key: str) -> str | None:
        return self.entries.get(str(key))

    def put(self, key: str, raw_response: str) -> None:
        self.entries[str(key)] = str(raw_response)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.entries, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.replace(self.path)


class QwenSemanticVerifier:
    """Lazy local/Hugging Face Qwen3.5-4B image-text verifier."""

    def __init__(
        self,
        model_id_or_path: str = REFERENCE_CONFIG.semantic_model,
        *,
        device_map: str = "auto",
        local_files_only: bool = False,
        max_new_tokens: int = REFERENCE_CONFIG.semantic_max_new_tokens,
    ) -> None:
        self.model_id_or_path = str(model_id_or_path)
        self.device_map = str(device_map)
        self.local_files_only = bool(local_files_only)
        self.max_new_tokens = int(max_new_tokens)
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForMultimodalLM as AutoMultimodalModel
            except ImportError:
                from transformers import (
                    AutoModelForImageTextToText as AutoMultimodalModel,
                )
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Qwen verification requires the 'models' optional dependencies") from exc
        source = self.model_id_or_path
        local_only = self.local_files_only or Path(source).expanduser().exists()
        self.processor = AutoProcessor.from_pretrained(
            source,
            local_files_only=local_only,
            trust_remote_code=True,
        )
        load_args = {
            "device_map": self.device_map,
            "local_files_only": local_only,
            "trust_remote_code": True,
        }
        try:
            self.model = AutoMultimodalModel.from_pretrained(source, dtype="auto", **load_args).eval()
        except TypeError:
            self.model = AutoMultimodalModel.from_pretrained(source, torch_dtype="auto", **load_args).eval()

    def generate(self, image_path: str | Path, prompt: str, *, view: str) -> str:
        """Generate one deterministic, candidate-constrained response."""

        self.load()
        assert self.model is not None and self.processor is not None
        system = (
            "You are a fine-grained visual recognition verifier. Choose only "
            "from the supplied candidates and output strict JSON without explanation."
            if view == "finedefics"
            else "You are a strict JSON generator. Output only valid JSON without explanation."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(Path(image_path).expanduser().resolve())},
                    {"type": "text", "text": str(prompt)},
                ],
            },
        ]
        template_args = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            inputs = self.processor.apply_chat_template(messages, enable_thinking=False, **template_args)
        except TypeError:
            inputs = self.processor.apply_chat_template(messages, **template_args)
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_length:]
        return str(self.processor.decode(generated, skip_special_tokens=True)).strip()
