from __future__ import annotations

import unittest

import numpy as np

from rgvspd.visual import (
    CLIP_PROMPT_TEMPLATES,
    clip_lpp_probabilities,
    prototype_probabilities,
)


class VisualBranchTests(unittest.TestCase):
    def test_reference_prompt_count(self) -> None:
        self.assertEqual(len(CLIP_PROMPT_TEMPLATES), 13)
        self.assertEqual(len(set(CLIP_PROMPT_TEMPLATES)), 13)

    def test_prototype_probabilities_prefer_matching_class(self) -> None:
        support = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        labels = np.asarray([0, 1])
        evaluation = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        probabilities = prototype_probabilities(support, labels, evaluation, 2)
        self.assertEqual(np.argmax(probabilities, axis=1).tolist(), [0, 1])

    def test_clip_lpp_uses_support_and_text_in_same_space(self) -> None:
        support = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        labels = np.asarray([0, 1])
        evaluation = support.copy()
        text = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
        probabilities = clip_lpp_probabilities(support, labels, evaluation, text)
        self.assertEqual(np.argmax(probabilities, axis=1).tolist(), [0, 1])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
