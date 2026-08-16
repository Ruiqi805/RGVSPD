# RG-VSPD

[![tests](https://github.com/Ruiqi805/RGVSPD/actions/workflows/tests.yml/badge.svg)](https://github.com/Ruiqi805/RGVSPD/actions/workflows/tests.yml)

RG-VSPD is a reproducible reference implementation for few-shot image classification with heterogeneous visual experts and candidate-constrained semantic verification. The repository contains the complete method pipeline: deterministic support selection, frozen DINOv2 and CLIP feature extraction, the two visual classifiers, disagreement routing, three Qwen semantic views, majority voting, caching, and a command-line runner.

Datasets, pretrained weights, generated feature caches, experiment outputs, and manuscripts are intentionally not redistributed.

## Method

For every query image, RG-VSPD performs the following steps:

1. **DINO branch:** extract the concatenation of the normalized CLS token and mean patch token from frozen DINOv2 ViT-L/14 at 448 × 448 resolution. Fuse a linear probe and cosine-prototype classifier with equal weight.
2. **CLIP-LPP branch:** encode the image with frozen CLIP ViT-B/16. For each class, combine its few-shot support prototype with 13 prompt text embeddings using text weight `0.88`, then classify with temperature `100`.
3. **Visual fusion:** combine aligned class probabilities with the locked, test-independent DINO weights `0.1/0.2/0.3/0.4/0.5` for `1/2/4/8/16` shots.
4. **Routing:** return the fused visual Top-1 directly when DINO and CLIP Top-1 agree. Otherwise, send the fused Top-5 candidates to the semantic verifier.
5. **Semantic verification:** Qwen3.5-4B independently runs AttrSeek, AttrBank, and FineDefics. Each view must return one exact candidate ID as strict JSON.
6. **Decision:** use a two-of-three semantic majority. If no valid majority exists, return the fused visual Top-1. The visual prediction is a fallback, not a fourth vote.

## Repository layout

```text
rgvspd/
  cli.py          command-line entry point
  config.py       locked reference constants
  core.py         fusion, routing, majority, and fallback
  data.py         manifest validation and few-shot selection
  experiment.py   raw-image experiment runner and result export
  pipeline.py     auditable per-image pipeline
  prompts.py      AttrSeek, AttrBank, and FineDefics prompts
  semantic.py     Qwen inference and resumable cache
  visual.py       DINOv2, CLIP-LPP, linear probe, and prototypes
tests/             dependency-light unit and integration tests
examples/          synthetic demo and input schemas
```

## Installation

The decision layer and tests need only NumPy and Pillow:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the model stack for raw-image experiments:

```bash
python -m pip install -e ".[models]"
```

A CUDA GPU is strongly recommended for DINOv2 ViT-L/14 and Qwen3.5-4B. Model loading is lazy, and the unit tests do not download checkpoints.

## Dataset manifest

Prepare a UTF-8 CSV with one row per image:

```csv
image_path,class_id,class_name,split,image_id
images/train/class_a/0001.jpg,0,class a,train,train-a-0001
images/test/class_a/1001.jpg,0,class a,test,test-a-1001
```

Required columns are `image_path`, `class_id`, and `split`. Relative paths are resolved from the CSV location. Every class must have at least the requested number of images in the support split. Validate before a long run:

```bash
rgvspd validate-manifest path/to/manifest.csv
```

## Run the reference pipeline

Full DINO + CLIP-LPP + Qwen run:

```bash
rgvspd run \
  --manifest path/to/manifest.csv \
  --output-dir outputs/dataset-shot1-seed1 \
  --shot 1 \
  --seed 1 \
  --qwen-model Qwen/Qwen3.5-4B
```

Use a local Qwen checkpoint without network access:

```bash
rgvspd run \
  --manifest path/to/manifest.csv \
  --output-dir outputs/local-run \
  --shot 2 \
  --seed 21 \
  --qwen-model /path/to/Qwen3.5-4B \
  --local-files-only
```

Useful controls:

- `--visual-only` skips Qwen but keeps the locked visual branches and router.
- `--max-eval N` runs a small end-to-end smoke subset.
- `--attribute-bank path.json` adds dataset-specific visible, discriminative, and negative cues to AttrBank.
- `--eval-split val` evaluates another manifest split without changing support selection.

The runner writes `run_config.json`, `metrics.json`, `predictions.json`, model feature caches, and a resumable semantic-response cache under the selected output directory.

## Lightweight decision API

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

print(result.prediction)
print(result.reason)
```

Run the synthetic demo and all dependency-light tests:

```bash
python -m examples.minimal_demo
python -m unittest discover -s tests -v
```

## Reproducibility contract

- Class IDs and all probability arrays must share one class order.
- Support images are sampled only from the configured support split with seeds `1`, `21`, or `93`.
- Visual weights are fixed before test evaluation; test labels are never accepted by prompt-building APIs.
- Semantic choices outside the fused candidate set are invalid.
- Invalid or malformed semantic outputs are excluded from the majority.
- Ties and candidate ordering are deterministic.
- Feature and semantic caches are validated against their inputs before reuse.

See [docs/reproduction.md](docs/reproduction.md) for the full checklist and expected output files.

## Scope and limitations

This repository enables method-level reproduction, but it does not claim that benchmark tables can be regenerated without obtaining the corresponding public datasets and pretrained checkpoints. Dataset download terms and model licenses remain the responsibility of the reproducer. Citation metadata will be added after an archival publication is available.
