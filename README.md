# RG-VSPD

RG-VSPD is a model-agnostic decision layer for few-shot image classification. It coordinates two heterogeneous visual predictors with a candidate-constrained semantic verifier.

This repository intentionally contains only the compact, runnable core. Dataset loaders, pretrained weights, generated caches, full experiment outputs, and manuscripts are not included.

## Decision rule

For each query image:

1. Fuse the DINO and CLIP class-probability vectors in the shared label space.
2. If their Top-1 predictions agree, return the fused visual Top-1 directly.
3. If they disagree, construct the fused Top-5 candidate set and collect three constrained semantic judgments:
   - **AttrSeek:** visible image evidence;
   - **AttrBank:** candidate-class attributes matched to visible evidence;
   - **FineDefics:** discriminative evidence and counter-evidence between candidates.
4. Accept a semantic prediction only when at least two views select the same candidate. Otherwise, fall back to the fused visual Top-1.

The fused visual prediction is a fallback, not a fourth vote. Semantic outputs outside the candidate set are rejected.

## Reference configuration

The reference DINO-CLIP-Qwen configuration is recorded in `rgvspd.config.REFERENCE_CONFIG`:

| Component | Setting |
| --- | --- |
| Visual representation branch | DINOv2 ViT-L/14 |
| Vision-language branch | CLIP ViT-B/16 |
| Semantic verifier | Qwen3.5-4B |
| DINO probe/prototype mixture | 0.5 / 0.5 |
| CLIP support/text mixture | 0.12 / 0.88 |
| CLIP temperature | 100 |
| CLIP prompt templates | 13 |
| Candidate count | 5 |
| DINO fusion weight | 0.1 (1-shot), 0.2 (2-shot), 0.5 (16-shot) |
| Support-set seeds | 1, 21, 93 |

These constants document the reference instantiation. The decision core accepts probability vectors from any pair of visual models.

## Quick start

No third-party runtime dependency is required.

```bash
python -m examples.minimal_demo
python -m unittest discover -s tests -v
```

Minimal use:

```python
from rgvspd import decide_for_shot

result = decide_for_shot(
    class_ids=["0", "1", "2", "3", "4", "5"],
    dino_probabilities=[0.45, 0.30, 0.10, 0.07, 0.05, 0.03],
    clip_probabilities=[0.20, 0.50, 0.12, 0.08, 0.06, 0.04],
    shot=2,
    semantic_votes={
        "attrseek": "0",
        "attrbank": "0",
        "finedefics": "1",
    },
)

print(result.prediction)  # 0
print(result.reason)      # semantic_majority
```

In a full system, call the semantic verifier only when `result.routed` would be true. The two-phase API `prepare_visual_decision` followed by `finalize_decision` supports that execution pattern without making unnecessary verifier calls.

## Integration contract

- `class_ids`, DINO probabilities, and CLIP probabilities must use the same class order.
- Inputs must be finite, non-negative scores with a positive sum; the core normalizes them before fusion.
- Semantic views must return one exact candidate ID as strict JSON: `{"best_class_id":"<candidate_id>"}`.
- Routing depends only on DINO/CLIP Top-1 disagreement, not on the fusion weight.
- Candidate ranking and all tie handling are deterministic.

## Repository scope

Included:

- score-level fusion and deterministic Top-5 construction;
- disagreement-based routing;
- strict candidate-response validation;
- three-view majority control and visual fallback;
- prompt-view templates and unit tests.

Excluded:

- datasets or few-shot splits;
- DINO, CLIP, or Qwen checkpoints;
- training and large-scale benchmark runners;
- generated predictions, caches, and manuscript files.
