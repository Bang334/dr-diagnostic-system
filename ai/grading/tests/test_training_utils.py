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
                    (class_dir / f"sample-{split_name}-{grade}.jpg").write_bytes(b"image")

            args = SimpleNamespace(
                dataset_dir=dataset_dir,
                output_dir=root / "run",
                split_dir=None,
                label_column="diagnosis",
            )
            splits = load_predefined_splits(args)

            self.assertEqual(set(splits), {"train", "val", "test"})
            for split in splits.values():
                self.assertEqual(len(split), 5)
                self.assertEqual(sorted(split["diagnosis"].tolist()), list(range(5)))
                self.assertTrue(split["image_id"].str.endswith(".jpg").all())
            self.assertTrue((root / "run" / "splits" / "val.csv").is_file())


if __name__ == "__main__":
    unittest.main()
