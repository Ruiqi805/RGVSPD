from __future__ import annotations

import unittest

from rgvspd import (
    build_semantic_prompt,
    decide,
    decide_for_shot,
    dino_weight_for_shot,
    fuse_probabilities,
    parse_semantic_response,
    prepare_visual_decision,
)

CLASS_IDS = ["0", "1", "2", "3", "4", "5"]


class ReferenceConfigTests(unittest.TestCase):
    def test_reported_shot_weights(self) -> None:
        self.assertEqual(dino_weight_for_shot(1), 0.1)
        self.assertEqual(dino_weight_for_shot(2), 0.2)
        self.assertEqual(dino_weight_for_shot(4), 0.3)
        self.assertEqual(dino_weight_for_shot(8), 0.4)
        self.assertEqual(dino_weight_for_shot(16), 0.5)

    def test_unreported_shot_is_explicit(self) -> None:
        with self.assertRaises(ValueError):
            dino_weight_for_shot(3)


class DecisionTests(unittest.TestCase):
    def test_score_level_fusion_normalizes_inputs(self) -> None:
        fused = fuse_probabilities([3, 1], [1, 3], dino_weight=0.25)
        self.assertAlmostEqual(fused[0], 0.375)
        self.assertAlmostEqual(fused[1], 0.625)
        self.assertAlmostEqual(sum(fused), 1.0)

    def test_agreement_returns_visual_without_semantic_override(self) -> None:
        result = decide(
            class_ids=CLASS_IDS,
            dino_probabilities=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02],
            clip_probabilities=[0.5, 0.3, 0.08, 0.05, 0.04, 0.03],
            dino_weight=0.2,
            semantic_votes={"attrseek": "1", "attrbank": "1", "finedefics": "1"},
        )
        self.assertFalse(result.routed)
        self.assertEqual(result.prediction, "0")
        self.assertEqual(result.reason, "visual_agreement")

    def test_disagreement_accepts_two_of_three_majority(self) -> None:
        result = decide_for_shot(
            class_ids=CLASS_IDS,
            dino_probabilities=[0.45, 0.30, 0.10, 0.07, 0.05, 0.03],
            clip_probabilities=[0.20, 0.50, 0.12, 0.08, 0.06, 0.04],
            shot=2,
            semantic_votes={"attrseek": "1", "attrbank": "1", "finedefics": "0"},
        )
        self.assertTrue(result.routed)
        self.assertEqual(result.prediction, "1")
        self.assertEqual(result.majority_count, 2)
        self.assertEqual(result.reason, "semantic_majority")

    def test_all_disagree_falls_back_to_fused_visual(self) -> None:
        result = decide_for_shot(
            class_ids=CLASS_IDS,
            dino_probabilities=[0.45, 0.30, 0.10, 0.07, 0.05, 0.03],
            clip_probabilities=[0.20, 0.50, 0.12, 0.08, 0.06, 0.04],
            shot=2,
            semantic_votes={"attrseek": "0", "attrbank": "1", "finedefics": "2"},
        )
        self.assertEqual(result.prediction, result.visual_prediction)
        self.assertEqual(result.majority_count, 1)
        self.assertEqual(result.reason, "visual_fallback_no_majority")

    def test_out_of_candidate_votes_are_rejected(self) -> None:
        visual = prepare_visual_decision(
            class_ids=CLASS_IDS,
            dino_probabilities=[0.45, 0.30, 0.10, 0.07, 0.05, 0.03],
            clip_probabilities=[0.20, 0.50, 0.12, 0.08, 0.06, 0.04],
            dino_weight=0.2,
            candidate_count=2,
        )
        self.assertNotIn("5", visual.candidates)
        result = decide(
            class_ids=CLASS_IDS,
            dino_probabilities=[0.45, 0.30, 0.10, 0.07, 0.05, 0.03],
            clip_probabilities=[0.20, 0.50, 0.12, 0.08, 0.06, 0.04],
            dino_weight=0.2,
            candidate_count=2,
            semantic_votes={"attrseek": "5", "attrbank": "5", "finedefics": None},
        )
        self.assertEqual(result.prediction, result.visual_prediction)
        self.assertEqual(result.majority_count, 0)

    def test_candidate_ties_follow_class_order(self) -> None:
        visual = prepare_visual_decision(
            class_ids=CLASS_IDS,
            dino_probabilities=[1, 0, 0, 0, 0, 0],
            clip_probabilities=[0, 1, 0, 0, 0, 0],
            dino_weight=0.5,
            candidate_count=5,
        )
        self.assertEqual(visual.candidates[:2], ("0", "1"))

    def test_router_is_independent_of_fusion_weight(self) -> None:
        kwargs = {
            "class_ids": CLASS_IDS,
            "dino_probabilities": [0.6, 0.3, 0.04, 0.03, 0.02, 0.01],
            "clip_probabilities": [0.2, 0.7, 0.04, 0.03, 0.02, 0.01],
        }
        clip_heavy = prepare_visual_decision(**kwargs, dino_weight=0.1)
        dino_heavy = prepare_visual_decision(**kwargs, dino_weight=0.9)
        self.assertTrue(clip_heavy.routed)
        self.assertTrue(dino_heavy.routed)
        self.assertNotEqual(clip_heavy.visual_prediction, dino_heavy.visual_prediction)


class PromptTests(unittest.TestCase):
    def test_prompt_contains_exact_candidates(self) -> None:
        prompt = build_semantic_prompt("attrseek", ["2", "4"], {"2": "red fox", "4": "gray fox"})
        self.assertIn("class_id=2", prompt)
        self.assertIn("class_name=gray fox", prompt)

    def test_strict_json_and_membership(self) -> None:
        self.assertEqual(parse_semantic_response('{"best_class_id":"2"}', ["2", "4"]), "2")
        self.assertIsNone(parse_semantic_response('```json\n{"best_class_id":"2"}\n```', ["2", "4"]))
        self.assertIsNone(parse_semantic_response('{"best_class_id":"9"}', ["2", "4"]))
        self.assertIsNone(parse_semantic_response('{"best_class_id":"2","note":"extra"}', ["2", "4"]))


if __name__ == "__main__":
    unittest.main()
