from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from rgvspd.experiment import ExperimentConfig, run_reference_experiment


class FakeDinoEncoder:
    def __init__(self, **_: object) -> None:
        pass

    def encode(self, paths: list[str] | tuple[str, ...], **_: object) -> np.ndarray:
        return np.ones((len(paths), 4), dtype=np.float32)


class FakeClipEncoder:
    def __init__(self, **_: object) -> None:
        pass

    def encode_images(self, paths: list[str] | tuple[str, ...], **_: object) -> np.ndarray:
        return np.ones((len(paths), 2), dtype=np.float32)

    def encode_class_prompts(self, class_ids: tuple[str, ...], _: dict[str, str]) -> np.ndarray:
        return np.ones((len(class_ids), 13, 2), dtype=np.float32)


class ExperimentTests(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        rows = []
        for class_id, color in (("0", (255, 0, 0)), ("1", (0, 255, 0))):
            for split, suffix in (("train", "support"), ("test", "query")):
                path = root / f"{class_id}-{suffix}.png"
                query_color = tuple(min(255, channel + 1) for channel in color)
                Image.new("RGB", (2, 2), color=color if split == "train" else query_color).save(path)
                rows.append(
                    {
                        "image_path": path.name,
                        "class_id": class_id,
                        "class_name": f"class {class_id}",
                        "split": split,
                        "image_id": f"{class_id}-{suffix}",
                    }
                )
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_visual_only_runner_writes_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            config = ExperimentConfig(
                manifest=self._manifest(root),
                output_dir=output,
                shot=1,
                seed=1,
                visual_only=True,
            )
            probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
            with (
                patch("rgvspd.experiment.DinoV2Encoder", FakeDinoEncoder),
                patch("rgvspd.experiment.CLIPEncoder", FakeClipEncoder),
                patch("rgvspd.experiment.dino_lp_proto_probabilities", return_value=probabilities),
                patch("rgvspd.experiment.clip_lpp_probabilities", return_value=probabilities),
            ):
                metrics = run_reference_experiment(config)
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["routed_count"], 0)
            self.assertTrue((output / "run_config.json").is_file())
            self.assertTrue((output / "support_selection.json").is_file())
            with (output / "split_integrity.json").open("r", encoding="utf-8") as handle:
                integrity = json.load(handle)
            self.assertEqual(integrity["content_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
