"""Spatially-aware MeanFlow network for the fine latent.

z_fine is an 8x8x8 tensor, not a vector. A flat MLP prior over the 512 flattened
dims can match every per-dimension moment and still miss the spatial covariance
the decoder depends on -- measured: per-dim dstd 0.38 and RBF-MMD 0.012 (both
look healthy) while FID was 218 against 7.1 for true latents.

This is a conv ResNet over the 8x8 grid. It accepts and returns FLAT [B, D]
tensors and reshapes internally, so it drops straight into the existing MeanFlow
class -- normalization, the JVP target and z_global conditioning all work
unchanged.

Everything here is forward-mode-AD safe (Conv2d / GroupNorm / SiLU), which the
MeanFlow identity's JVP requires.
"""
import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=x.device) * (-math.log(10000.0) / (half - 1))
        )
        a = x[:, None] * freqs[None, :]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class ConvResBlock(nn.Module):
    """Standard diffusion-style residual block with FiLM-ish additive conditioning."""

    def __init__(self, ch, emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, ch)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class ConvMeanFlowNet(nn.Module):
    """u_theta(z, r, t | cond) over a [C, S, S] latent, exposed as flat [B, C*S*S]."""

    def __init__(self, channels=8, spatial=8, width=128, n_blocks=6,
                 time_dim=64, cond_dim=0, groups=8,
                 time_scale=1.0):
        super().__init__()
        self.channels, self.spatial = channels, spatial
        self.cond_dim = cond_dim
        self.time_scale = time_scale
        self.D = channels * spatial * spatial

        emb_dim = width * 2
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(2 * time_dim + cond_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.conv_in = nn.Conv2d(channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ConvResBlock(width, emb_dim, groups) for _ in range(n_blocks)])
        self.out = nn.Sequential(
            nn.GroupNorm(groups, width), nn.SiLU(),
            nn.Conv2d(width, channels, 3, padding=1),
        )

    def forward(self, z, r, t, cond=None):
        B = z.shape[0]
        x = z.reshape(B, self.channels, self.spatial, self.spatial)
        parts = [self.time_emb(r.reshape(-1) * self.time_scale),
                 self.time_emb(t.reshape(-1) * self.time_scale)]
        if self.cond_dim:
            assert cond is not None, "conditional net called without cond"
            parts.append(cond)
        emb = self.emb_mlp(torch.cat(parts, dim=-1))

        h = self.conv_in(x)
        for blk in self.blocks:
            h = blk(h, emb)
        return self.out(h).reshape(B, self.D)
