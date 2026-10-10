"""PatchTST-style Transformer encoder for 252-day equity windows.

Library module (no CLI). Each 21-day patch of the [WINDOW, n_channels] window is linearly embedded; a
learned CLS token and learned positional embeddings are added; masked patches are replaced by a learned
mask token before the pre-norm Transformer. ``emb`` is the CLS output after a final LayerNorm.
"""
from __future__ import annotations

import torch
from torch import nn

from .windows import CHANNELS, PATCH, WINDOW


class FingerprintEncoder(nn.Module):
    def __init__(self, n_channels: int = len(CHANNELS) + 1, window: int = WINDOW, patch: int = PATCH,
                 d_model: int = 128, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert window % patch == 0
        self.config = dict(n_channels=n_channels, window=window, patch=patch, d_model=d_model,
                           n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.n_patches = window // patch
        self.embed = nn.Linear(patch * n_channels, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, patch_mask=None):
        """x [B, WINDOW, C], patch_mask bool [B, n_patches] (True = masked) -> (emb [B, d], tokens [B, P, d])."""
        b = x.shape[0]
        p = self.embed(x.reshape(b, self.n_patches, -1))
        if patch_mask is not None:
            p = torch.where(patch_mask[..., None], self.mask_token.to(p.dtype).expand_as(p), p)
        h = torch.cat([self.cls.to(p.dtype).expand(b, -1, -1), p], dim=1) + self.pos.to(p.dtype)
        h = self.norm(self.blocks(h))
        return h[:, 0], h[:, 1:]


def load_encoder(path, device):
    """Load a pretraining checkpoint -> (encoder in eval mode on ``device``, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    enc = FingerprintEncoder(**ckpt["config"])
    enc.load_state_dict(ckpt["state_dict"])
    return enc.to(device).eval(), ckpt
