import errno
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from ai.train_fewshot.runtime import prepare_deepdrid_target


class DeepDRiDPreparationTests(unittest.TestCase):
    def test_copies_images_when_links_are_not_supported(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_root = root / "DeepDRiD-v1.1"
            regular_root = source_root / "regular_fundus_images"
            output_dir = root / "prepared"

            split_specs = {
                "regular-fundus-training": ("1_l1", b"train"),
                "regular-fundus-validation": ("2_r1", b"validation"),
            }
            for folder_name, (image_id, contents) in split_specs.items():
                folder = regular_root / folder_name
                image_dir = folder / "Images" / image_id.split("_", 1)[0]
                image_dir.mkdir(parents=True)
                (image_dir / f"{image_id}.jpg").write_bytes(contents)
                pd.DataFrame(
                    {
                        "image_id": [image_id],
                        "left_eye_DR_Level": [0],
                        "right_eye_DR_Level": [None],
                    }
                ).to_csv(folder / f"{folder_name}.csv", index=False)

            evaluation = regular_root / "Online-Challenge1&2-Evaluation"
            test_image_dir = evaluation / "Images" / "3"
            test_image_dir.mkdir(parents=True)
            (test_image_dir / "3_l1.jpg").write_bytes(b"test")
            pd.DataFrame(
                {"image_id": ["3_l1"], "DR_Levels": [0]}
            ).to_excel(evaluation / "Challenge1_labels.xlsx", index=False)

            link_error = PermissionError(errno.EPERM, "Operation not permitted")
            symlink_error = OSError(errno.ENOTSUP, "Operation not supported")
            with mock.patch(
                "ai.train_fewshot.runtime.os.link", side_effect=link_error
            ), mock.patch.object(Path, "symlink_to", side_effect=symlink_error):
                result = prepare_deepdrid_target(source_root, output_dir)

            self.assertEqual(result, output_dir.resolve())
            self.assertEqual((output_dir / "train/0/1_l1.jpg").read_bytes(), b"train")
            self.assertEqual(
                (output_dir / "validation/0/2_r1.jpg").read_bytes(), b"validation"
            )
            self.assertEqual((output_dir / "test/0/3_l1.jpg").read_bytes(), b"test")


if __name__ == "__main__":
    unittest.main()
