"""Reference DINOv2 and CLIP-LPP visual branches.

The array-level functions depend only on NumPy. Model loading is lazy so the
decision core and its tests remain usable without the heavyweight model stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .config import REFERENCE_CONFIG

EPS = 1e-8

CLIP_PROMPT_TEMPLATES: tuple[str, ...] = (
    "a photo of a {class_name}.",
    "a photo of the {class_name}.",
    "a natural image of a {class_name}.",
    "a close-up photo of a {class_name}.",
    "a cropped photo of a {class_name}.",
    "a bright photo of a {class_name}.",
    "a dark photo of a {class_name}.",
    "a photo of a small {class_name}.",
    "a photo of a large {class_name}.",
    "a photo of a {class_name} in the wild.",
    "a photo of one {class_name}.",
    "a good photo of a {class_name}.",
    "a blurry photo of a {class_name}.",
)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """L2-normalize the final dimension with a numerical floor."""

    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, EPS)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax."""

    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("logits must be a two-dimensional array")
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def class_prototypes(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Build normalized per-class support prototypes."""

    features = l2_normalize(np.asarray(support_features, dtype=np.float32))
    labels = np.asarray(support_labels, dtype=np.int64)
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("support features and labels have incompatible shapes")
    prototypes = np.empty((int(num_classes), features.shape[1]), dtype=np.float32)
    for class_index in range(int(num_classes)):
        mask = labels == class_index
        if not mask.any():
            raise ValueError(f"support set has no sample for class index {class_index}")
        prototypes[class_index] = features[mask].mean(axis=0)
    return l2_normalize(prototypes)


def prototype_probabilities(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    evaluation_features: np.ndarray,
    num_classes: int,
    *,
    temperature: float = REFERENCE_CONFIG.prototype_temperature,
) -> np.ndarray:
    """Cosine-prototype class probabilities."""

    prototypes = class_prototypes(support_features, support_labels, num_classes)
    evaluation = l2_normalize(np.asarray(evaluation_features, dtype=np.float32))
    return softmax(float(temperature) * evaluation @ prototypes.T)


def clip_lpp_probabilities(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    evaluation_features: np.ndarray,
    text_embeddings: np.ndarray,
    *,
    text_weight: float = REFERENCE_CONFIG.clip_text_weight,
    temperature: float = REFERENCE_CONFIG.clip_temperature,
) -> np.ndarray:
    """Compute the support-prototype-text fusion CLIP branch.

    ``text_embeddings`` has shape ``[classes, prompts, dimension]``. Each prompt
    vector is fused with its class support prototype before prompt logits are
    averaged, matching the locked CLIP-LPP definition.
    """

    texts = l2_normalize(np.asarray(text_embeddings, dtype=np.float32))
    if texts.ndim != 3:
        raise ValueError("text_embeddings must have shape [classes, prompts, dimension]")
    alpha = float(text_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("text_weight must be in [0, 1]")
    prototypes = class_prototypes(support_features, support_labels, texts.shape[0])
    evaluation = l2_normalize(np.asarray(evaluation_features, dtype=np.float32))
    if evaluation.shape[1] != texts.shape[2] or prototypes.shape[1] != texts.shape[2]:
        raise ValueError("CLIP image and text embedding dimensions must match")
    class_logits = []
    for class_index in range(texts.shape[0]):
        repeated = np.repeat(prototypes[class_index : class_index + 1], texts.shape[1], axis=0)
        weights = l2_normalize((1.0 - alpha) * repeated + alpha * texts[class_index])
        class_logits.append((float(temperature) * evaluation @ weights.T).mean(axis=1))
    return softmax(np.stack(class_logits, axis=1))


def dino_lp_proto_probabilities(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    evaluation_features: np.ndarray,
    num_classes: int,
    *,
    device: str = "cuda",
    epochs: int = 80,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-3,
    label_smoothing: float = 0.08,
    batch_size: int = 256,
    seed: int = 1,
    linear_probe_weight: float = REFERENCE_CONFIG.dino_linear_probe_weight,
) -> np.ndarray:
    """Train the linear probe and fuse it 0.5/0.5 with cosine prototypes."""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - exercised in full environments
        raise ImportError("DINO linear probing requires the 'models' optional dependencies") from exc

    alpha = float(linear_probe_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("linear_probe_weight must be in [0, 1]")
    x_train = l2_normalize(np.asarray(support_features, dtype=np.float32))
    x_eval = l2_normalize(np.asarray(evaluation_features, dtype=np.float32))
    labels = np.asarray(support_labels, dtype=np.int64)
    target_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = nn.Linear(x_train.shape[1], int(num_classes)).to(target_device)
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("every class must occur in the support set")
    inverse = (1.0 / counts) / (1.0 / counts).mean()
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(inverse, device=target_device),
        label_smoothing=float(label_smoothing),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=generator)
    model.train()
    for _ in range(int(epochs)):
        for feature_batch, label_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(feature_batch.to(target_device)), label_batch.to(target_device))
            loss.backward()
            optimizer.step()
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x_eval), 4096):
            tensor = torch.from_numpy(x_eval[start : start + 4096]).to(target_device)
            chunks.append(torch.softmax(model(tensor), dim=1).cpu().numpy())
    linear_probabilities = np.concatenate(chunks, axis=0).astype(np.float32)
    prototype = prototype_probabilities(support_features, labels, evaluation_features, num_classes)
    fused = alpha * linear_probabilities + (1.0 - alpha) * prototype
    return fused / np.maximum(fused.sum(axis=1, keepdims=True), EPS)


class DinoV2Encoder:
    """Frozen DINOv2 ViT-L/14 encoder using CLS + mean-patch concatenation."""

    def __init__(self, *, device: str = "cuda") -> None:
        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:  # pragma: no cover
            raise ImportError("DINOv2 encoding requires the 'models' optional dependencies") from exc
        self.torch = torch
        self.device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize((REFERENCE_CONFIG.dino_input_size, REFERENCE_CONFIG.dino_input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.model = torch.hub.load("facebookresearch/dinov2", REFERENCE_CONFIG.dino_backbone)
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, image_paths: Sequence[str | Path], *, batch_size: int = 16) -> np.ndarray:
        from PIL import Image

        result: list[np.ndarray] = []
        for start in range(0, len(image_paths), int(batch_size)):
            tensors = []
            for path in image_paths[start : start + int(batch_size)]:
                with Image.open(path) as image:
                    tensors.append(self.transform(image.convert("RGB")))
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.inference_mode():
                output = self.model.forward_features(batch)
                cls = output["x_norm_clstoken"]
                patch = output["x_norm_patchtokens"]
                if cls.ndim > 2:
                    cls = cls[:, 0]
                features = self.torch.cat([cls, patch.mean(dim=1)], dim=1)
            result.append(features.float().cpu().numpy())
        return np.concatenate(result, axis=0).astype(np.float32)


class CLIPEncoder:
    """Frozen Hugging Face CLIP ViT-B/16 image and text encoder."""

    def __init__(self, *, device: str = "cuda", local_files_only: bool = False) -> None:
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError("CLIP encoding requires the 'models' optional dependencies") from exc
        self.torch = torch
        self.device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        model_id = REFERENCE_CONFIG.clip_backbone
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = CLIPModel.from_pretrained(model_id, local_files_only=local_files_only).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_images(self, image_paths: Sequence[str | Path], *, batch_size: int = 32) -> np.ndarray:
        from PIL import Image

        result: list[np.ndarray] = []
        for start in range(0, len(image_paths), int(batch_size)):
            images = []
            for path in image_paths[start : start + int(batch_size)]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with self.torch.inference_mode():
                features = self.model.get_image_features(pixel_values=pixel_values)
            result.append(l2_normalize(features.float().cpu().numpy()))
        return np.concatenate(result, axis=0).astype(np.float32)

    def encode_class_prompts(
        self,
        class_ids: Sequence[str],
        names: Mapping[str, str],
        *,
        batch_size: int = 128,
    ) -> np.ndarray:
        prompts = [
            template.format(class_name=names.get(str(class_id), str(class_id)).replace("_", " "))
            for class_id in class_ids
            for template in CLIP_PROMPT_TEMPLATES
        ]
        result: list[np.ndarray] = []
        for start in range(0, len(prompts), int(batch_size)):
            inputs = self.processor(
                text=prompts[start : start + int(batch_size)],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                features = self.model.get_text_features(**inputs)
            result.append(l2_normalize(features.float().cpu().numpy()))
        flat = np.concatenate(result, axis=0).astype(np.float32)
        return flat.reshape(len(class_ids), len(CLIP_PROMPT_TEMPLATES), -1)
