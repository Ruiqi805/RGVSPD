from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rgvspd.pipeline import run_pipeline_decision
from rgvspd.prompts import Candidate, build_reference_prompt
from rgvspd.semantic import JsonSemanticCache


class FakeVerifier:
    model_id_or_path = "fake-verifier"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, image_path: str | Path, prompt: str, *, view: str) -> str:
        self.calls.append(view)
        prediction = "0" if view in {"attrseek", "attrbank"} else "1"
        return f'{{"best_class_id":"{prediction}"}}'


class PipelineTests(unittest.TestCase):
    def test_disagreement_runs_three_views_and_caches_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"test-image")
            cache = JsonSemanticCache(root / "cache.json")
            verifier = FakeVerifier()
            kwargs = {
                "image_path": image,
                "class_ids": ["0", "1", "2", "3", "4", "5"],
                "class_names": {str(index): f"class {index}" for index in range(6)},
                "dino_probabilities": [0.45, 0.30, 0.10, 0.07, 0.05, 0.03],
                "clip_probabilities": [0.20, 0.50, 0.12, 0.08, 0.06, 0.04],
                "shot": 2,
                "verifier": verifier,
                "semantic_cache": cache,
            }
            first = run_pipeline_decision(**kwargs)
            self.assertEqual(first.decision.prediction, "0")
            self.assertEqual(first.decision.reason, "semantic_majority")
            self.assertEqual(verifier.calls, ["attrseek", "attrbank", "finedefics"])
            second = run_pipeline_decision(**kwargs)
            self.assertEqual(second.decision.prediction, "0")
            self.assertEqual(len(verifier.calls), 3)

    def test_prompt_is_candidate_constrained(self) -> None:
        candidates = (
            Candidate("7", "class seven", 1, 0.6, "dino"),
            Candidate("9", "class nine", 2, 0.4, "clip"),
        )
        prompt = build_reference_prompt(
            "attrbank",
            candidates,
            dino_prediction="7",
            clip_prediction="9",
        )
        self.assertIn("best_class_id must be exactly one of: 7, 9", prompt)
        self.assertNotIn("ground truth", prompt.lower())


if __name__ == "__main__":
    unittest.main()
