# Reproduction guide

## 1. Environment

Use Python 3.10 or newer. Create a clean environment and install the model extra:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[models]"
```

The runner uses the official Hugging Face identifier `openai/clip-vit-base-patch16`, the official DINOv2 hub entry `dinov2_vitl14`, and `Qwen/Qwen3.5-4B` by default. Pass a local Qwen path with `--local-files-only` when the checkpoint is already downloaded.

## 2. Data

Create a CSV manifest following `examples/manifest.example.csv`. A manifest is preferred over dataset-specific absolute paths because it makes the exact support and evaluation universe auditable.

The support split must contain at least `shot` images per class. Support selection sorts records deterministically and then samples per class from a SHA-256-derived seed. The evaluation split is never read during support selection.

## 3. Smoke run

Run a small visual-only subset first:

```bash
rgvspd run \
  --manifest path/to/manifest.csv \
  --output-dir outputs/smoke \
  --shot 1 \
  --seed 1 \
  --visual-only \
  --max-eval 8
```

Then repeat without `--visual-only` to check Qwen loading and strict-JSON parsing.

## 4. Full runs

Run each reported support seed independently:

```bash
rgvspd run --manifest path/to/manifest.csv --output-dir outputs/shot1-seed1  --shot 1 --seed 1
rgvspd run --manifest path/to/manifest.csv --output-dir outputs/shot1-seed21 --shot 1 --seed 21
rgvspd run --manifest path/to/manifest.csv --output-dir outputs/shot1-seed93 --shot 1 --seed 93
```

The same command structure applies to 2, 4, 8, and 16 shots. Do not choose fusion weights using evaluation labels.

## 5. Optional attribute bank

AttrBank works without a resource file by deriving generic visible-cue placeholders from class names. For a stronger dataset-specific instantiation, pass JSON in this form:

```json
{
  "classes": {
    "0": {
      "visible_attributes": ["compact body", "short muzzle"],
      "discriminative_cues": ["upright triangular ears"],
      "negative_cues": ["long drooping ears"]
    }
  }
}
```

The resource must be created without consulting evaluation labels or predictions.

## 6. Outputs and resume behavior

- `run_config.json`: complete runtime arguments.
- `support_selection.json`: exact support image IDs for the shot/seed run.
- `split_integrity.json`: path/content overlap check and support-set digest.
- `metrics.json`: aggregate accuracy and routing coverage.
- `predictions.json`: per-image visual and semantic decisions.
- `feature_cache/*.npz`: DINO, CLIP, and text embeddings keyed by exact inputs.
- `semantic_cache.json`: prompt responses keyed by image bytes, prompt text, view, and model identifier.

If a semantic run is interrupted, rerun the same command. Completed prompt calls are reused from the cache.

## 7. Integrity checks

Before reporting results:

1. Confirm that training/support image hashes do not overlap evaluation image hashes.
2. Preserve the manifest and command line for every run.
3. Verify that `run_config.json` records the intended shot and seed.
4. Confirm that every routed prediction has at most three semantic votes.
5. Confirm that every non-null semantic vote appears in its recorded candidate set.
6. Aggregate the three support seeds without selecting the best seed.
