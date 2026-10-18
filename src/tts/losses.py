"""GAN losses shared by HiFi-GAN and VITS (least-squares GAN, Mao et al. 2017).

- discriminator: (D(real) - 1)^2 + D(fake)^2        -> real towards 1, fake towards 0
- generator:     (D(fake) - 1)^2                      -> fool the discriminator
- feature matching: L1 between every hidden activation of D on real vs. fake; a smooth,
  perceptual reconstruction loss that keeps GAN training stable
- mel loss: L1 between log-mels of real and generated waveforms (weight 45 in HiFi-GAN)
- KL (VITS): KL( q(z | spectrogram) || p(z | text) ) between two diagonal Gaussians,
  computed on the flow-transformed posterior sample z_p
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def discriminator_loss(real_outs: list[torch.Tensor], fake_outs: list[torch.Tensor]) -> torch.Tensor:
    return sum(((r - 1) ** 2).mean() + (f**2).mean() for r, f in zip(real_outs, fake_outs))


def generator_loss(fake_outs: list[torch.Tensor]) -> torch.Tensor:
    return sum(((f - 1) ** 2).mean() for f in fake_outs)


def feature_matching_loss(real_feats: list[list[torch.Tensor]], fake_feats: list[list[torch.Tensor]]) -> torch.Tensor:
    total = 0.0
    for rf, ff in zip(real_feats, fake_feats):
        for r, f in zip(rf, ff):
            total = total + F.l1_loss(f, r.detach())
    return total


def kl_loss(z_p: torch.Tensor, logs_q: torch.Tensor, mu_p: torch.Tensor, logs_p: torch.Tensor,
            mask: torch.Tensor) -> torch.Tensor:
    """All (B, C, T); mask (B, 1, T). KL between N(z; mu_q, s_q) pushed through the flow and N(mu_p, s_p)."""
    kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - mu_p) ** 2 * torch.exp(-2.0 * logs_p)
    return (kl * mask).sum() / mask.sum() / z_p.size(1)


__all__ = ["discriminator_loss", "feature_matching_loss", "generator_loss", "kl_loss"]
