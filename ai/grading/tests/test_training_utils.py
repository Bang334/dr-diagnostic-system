import unittest

import torch
import torch.nn as nn

from ai.grading.train import resize_pos_embed_for_model


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


if __name__ == "__main__":
    unittest.main()
