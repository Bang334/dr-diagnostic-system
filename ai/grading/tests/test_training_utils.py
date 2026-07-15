import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from ai.grading.train import load_predefined_splits, resize_pos_embed_for_model


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid_size = (16, 16)


class _TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_prefix_tokens = 1
        self.patch_embed = _PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, 257, 8))


class PositionEmbeddingTests(unittest.TestCase):
    def test_resizes_retfound_37_grid_to_model_16_grid(self):
        source = torch.randn(1, 1370, 8)
        state = {"pos_embed": source.clone()}
        model = _TinyViT()

        resize_pos_embed_for_model(state, model)
        model.load_state_dict(state, strict=False)

        self.assertEqual(state["pos_embed"].shape, (1, 257, 8))
        torch.testing.assert_close(state["pos_embed"][:, :1], source[:, :1])


class PredefinedSplitTests(unittest.TestCase):
    def test_loads_kaggle_folder_layout_without_resplitting(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_dir = root / "download"
            for split_name in ("train", "validation", "test"):
                for grade in range(5):
                    class_dir = dataset_dir / "split_dataset" / split_name / str(grade)
                    class_dir.mkdir(parents=True, exist_ok=True)
                    sample_count = 3
                    for index in range(sample_count):
                        (class_dir / f"sample-{split_name}-{grade}-{index}.jpg").write_bytes(
                            b"image"
                        )

            args = SimpleNamespace(
                dataset_dir=dataset_dir,
                output_dir=root / "run",
                split_dir=None,
                label_column="diagnosis",
                max_train_images_per_grade=2,
                max_eval_images_per_grade=2,
                seed=17,
            )
            splits = load_predefined_splits(args)

            self.assertEqual(set(splits), {"train", "val", "test"})
            self.assertEqual(len(splits["train"]), 10)
            self.assertEqual(
                splits["train"]["diagnosis"].value_counts().sort_index().to_dict(),
                {grade: 2 for grade in range(5)},
            )
            for split in (splits["val"], splits["test"]):
                self.assertEqual(len(split), 10)
                self.assertEqual(
                    split["diagnosis"].value_counts().sort_index().to_dict(),
                    {grade: 2 for grade in range(5)},
                )
                self.assertTrue(split["image_id"].str.endswith(".jpg").all())
            self.assertTrue((root / "run" / "splits" / "val.csv").is_file())
            saved_train = (root / "run" / "splits" / "train.csv").read_text()

            repeated = load_predefined_splits(args)
            self.assertEqual(saved_train, (root / "run" / "splits" / "train.csv").read_text())
            self.assertEqual(
                splits["train"]["image_id"].tolist(),
                repeated["train"]["image_id"].tolist(),
            )


if __name__ == "__main__":
    unittest.main()
