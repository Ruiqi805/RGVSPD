from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from rgvspd.data import (
    assert_disjoint_images,
    class_names,
    load_manifest,
    records_for_split,
    select_fewshot_support,
)


class ManifestTests(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        rows = []
        for class_id in ("0", "1"):
            for index in range(3):
                image = root / f"train-{class_id}-{index}.jpg"
                image.write_bytes(f"train-{class_id}-{index}".encode())
                rows.append(
                    {
                        "image_path": image.name,
                        "class_id": class_id,
                        "class_name": f"class {class_id}",
                        "split": "train",
                        "image_id": f"train-{class_id}-{index}",
                    }
                )
            image = root / f"test-{class_id}.jpg"
            image.write_bytes(f"test-{class_id}".encode())
            rows.append(
                {
                    "image_path": image.name,
                    "class_id": class_id,
                    "class_name": f"class {class_id}",
                    "split": "test",
                    "image_id": f"test-{class_id}",
                }
            )
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_manifest_and_fewshot_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = load_manifest(self._manifest(Path(directory)))
            self.assertEqual(class_names(records), {"0": "class 0", "1": "class 1"})
            self.assertEqual(len(records_for_split(records, "test")), 2)
            first = select_fewshot_support(records, shot=2, seed=21)
            repeated = select_fewshot_support(records, shot=2, seed=21)
            self.assertEqual(first, repeated)
            self.assertEqual(len(first), 4)
            self.assertTrue(all(item.split == "train" for item in first))
            integrity = assert_disjoint_images(first, records_for_split(records, "test"))
            self.assertEqual(integrity["content_overlap"], 0)

    def test_insufficient_class_support_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = load_manifest(self._manifest(Path(directory)))
            with self.assertRaises(ValueError):
                select_fewshot_support(records, shot=4, seed=1)

    def test_content_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = load_manifest(self._manifest(root))
            support = select_fewshot_support(records, shot=1, seed=1)
            evaluation = list(records_for_split(records, "test"))
            evaluation[0].image_path.write_bytes(support[0].image_path.read_bytes())
            with self.assertRaises(ValueError):
                assert_disjoint_images(support, evaluation)


if __name__ == "__main__":
    unittest.main()
