"""Command-line interface for the reference pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import REFERENCE_CONFIG
from .data import class_names, load_manifest
from .experiment import ExperimentConfig, run_reference_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rgvspd", description="RG-VSPD reference pipeline")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate-manifest", help="validate a dataset CSV")
    validate.add_argument("manifest", type=Path)

    run = subcommands.add_parser("run", help="run the reference image-classification pipeline")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--shot", type=int, choices=[1, 2, 4, 8, 16], required=True)
    run.add_argument("--seed", type=int, choices=list(REFERENCE_CONFIG.support_set_seeds), default=1)
    run.add_argument("--support-split", default="train")
    run.add_argument("--eval-split", default="test")
    run.add_argument("--device", default="cuda")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--linear-epochs", type=int, default=80)
    run.add_argument("--qwen-model", default=REFERENCE_CONFIG.semantic_model)
    run.add_argument("--attribute-bank", type=Path)
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--visual-only", action="store_true")
    run.add_argument("--max-eval", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-manifest":
        records = load_manifest(args.manifest)
        splits: dict[str, int] = {}
        for record in records:
            splits[record.split] = splits.get(record.split, 0) + 1
        print(json.dumps({"images": len(records), "classes": len(class_names(records)), "splits": splits}, indent=2))
        return 0

    config = ExperimentConfig(
        manifest=args.manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        shot=args.shot,
        seed=args.seed,
        eval_split=args.eval_split,
        support_split=args.support_split,
        device=args.device,
        batch_size=args.batch_size,
        linear_epochs=args.linear_epochs,
        qwen_model=args.qwen_model,
        local_files_only=args.local_files_only,
        visual_only=args.visual_only,
        max_eval=args.max_eval,
        attribute_bank=args.attribute_bank.resolve() if args.attribute_bank else None,
    )
    metrics = run_reference_experiment(config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
