import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from ai.grading.evaluate_fewshot import (
    FewShotProtoNetEvaluator,
    load_fewshot_checkpoint,
)


class _TinyEncoder(nn.Module):
    num_features = 4

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 5)
        self.batch_sizes: list[int] = []

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(images.size(0))
        return images

    def forward_head(
        self, features: torch.Tensor, pre_logits: bool = False
    ) -> torch.Tensor:
        return features if pre_logits else self.head(features)


class FewShotCheckpointEvaluationTests(unittest.TestCase):
    def test_loads_few_shot_schema_without_supervised_args_key(self) -> None:
        source = _TinyEncoder()
        state = {
            "method": "fixed_support_target_domain_protonet",
            "encoder_model": source.state_dict(),
            "projection": {},
            "embedding_dim": 4,
            "prototypes": torch.randn(5, 4),
            "class_ids": torch.arange(5),
            "temperature": 0.1,
            "base_model_args": {
                "model_source": "timm",
                "model_name": "tiny-test-model",
                "image_size": 224,
                "loss": "ce",
            },
            "few_shot_args": {"forward_batch_size": 2},
            "best_support_loss": 0.25,
        }
        self.assertNotIn("args", state)

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "best.pth"
            torch.save(state, checkpoint_path)
            with patch(
                "ai.grading.evaluate_fewshot.timm.create_model",
                return_value=_TinyEncoder(),
            ):
                evaluator, saved_args = load_fewshot_checkpoint(
                    checkpoint_path, torch.device("cpu")
                )

        self.assertEqual(saved_args.model_name, "tiny-test-model")
        self.assertEqual(evaluator.forward_batch_size, 2)

    def test_encoder_uses_checkpoint_forward_batch_size(self) -> None:
        encoder = _TinyEncoder()
        evaluator = FewShotProtoNetEvaluator(
            encoder=encoder,
            projection=nn.Identity(),
            prototypes=torch.randn(5, 4),
            class_ids=torch.arange(5),
            forward_batch_size=2,
        )

        _, predictions = evaluator(torch.randn(5, 4))

        self.assertEqual(encoder.batch_sizes, [2, 2, 1])
        self.assertEqual(predictions.shape, (5,))


if __name__ == "__main__":
    unittest.main()
