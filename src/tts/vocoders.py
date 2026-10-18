"""Vocoders: mel-spectrogram -> waveform.

- **Griffin-Lim** (`audio.griffin_lim`): no parameters, iterative phase recovery. Always
  available, robotic quality. The baseline and the vocoder of the unit tests.
- **HiFi-GAN** (Kong et al. 2020): a fully convolutional *generator* upsamples the mel
  (hop_length x) with transposed convolutions, each followed by a *multi-receptive-field
  fusion* block (sum of residual stacks with different kernel sizes / dilations).
  It is trained as a GAN against two families of *discriminators*:
  - MPD (multi-period): reshapes the waveform into 2-D (period x time) for periods
    2, 3, 5, 7, 11, so each sub-discriminator looks at a different periodic structure;
  - MSD (multi-scale): 1-D convolutions on the waveform, its 2x and 4x downsampled versions.
  Losses (`losses.py`): least-squares GAN, *feature matching* (L1 between the discriminators'
  hidden activations on real and fake), and a mel L1 reconstruction loss (weight 45).
  The same generator and discriminators are the waveform part of VITS.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from tts.config import Config

LRELU = 0.1


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilations: tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(nn.Conv1d(channels, channels, kernel, dilation=d, padding=(kernel - 1) // 2 * d))
            for d in dilations)
        self.convs2 = nn.ModuleList(
            weight_norm(nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2)) for _ in dilations)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            h = c1(F.leaky_relu(x, LRELU))
            x = x + c2(F.leaky_relu(h, LRELU))
        return x


class HiFiGANGenerator(nn.Module):
    """(B, cin, T) -> (B, prod(upsample_rates) * T) waveform in [-1, 1]."""

    def __init__(self, cfg: Config, cin: int | None = None):
        super().__init__()
        m = cfg.model
        cin = cin or cfg.audio.n_mels
        assert int(torch.tensor(m.upsample_rates).prod()) == cfg.audio.hop_length, \
            "product of upsample_rates must equal hop_length"
        ch = m.upsample_initial
        self.pre = weight_norm(nn.Conv1d(cin, ch, 7, padding=3))
        self.ups, self.mrfs = nn.ModuleList(), nn.ModuleList()
        for r in m.upsample_rates:
            self.ups.append(weight_norm(nn.ConvTranspose1d(ch, ch // 2, 2 * r, stride=r, padding=r // 2)))
            ch //= 2
            self.mrfs.append(nn.ModuleList(ResBlock(ch, k, m.resblock_dilations) for k in m.resblock_kernels))
        self.post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))
        self.smooth_init()

    def smooth_init(self) -> None:
        """Start every transposed convolution as a linear interpolator (channel i -> channel i).

        A randomly initialised ConvTranspose1d repeats its kernel every `stride` samples and
        produces a buzz at the frame rate (sample_rate / hop_length, 62.5 Hz at 8 kHz / 128);
        a triangular kernel interpolates smoothly between frames instead, so early training
        sounds like a low-pass version of the target rather than a buzz. The other channels keep
        a small random kernel (weight-norm needs a non-zero direction per channel)."""
        for up in self.ups:
            v, g = up.parametrizations.weight.original1, up.parametrizations.weight.original0
            cin, cout, k = v.shape
            r = up.stride[0]
            tri = torch.clamp(1 - torch.abs(torch.arange(k, dtype=v.dtype) - (k - 1) / 2) / r, min=0)
            with torch.no_grad():
                v.mul_(0.05)
                for i in range(min(cin, cout)):
                    v[i, i] = tri
                g.copy_(v.flatten(1).norm(dim=1).view(-1, 1, 1))  # weight == v

    def forward(self, x):
        x = self.pre(x)
        for up, mrf in zip(self.ups, self.mrfs):
            x = up(F.leaky_relu(x, LRELU))
            x = sum(block(x) for block in mrf) / len(mrf)
        return torch.tanh(self.post(F.leaky_relu(x, LRELU))).squeeze(1)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, channels: int = 32):
        super().__init__()
        self.period = period
        chs = [1, channels, channels * 2, channels * 4]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chs[i], chs[i + 1], (5, 1), stride=(3, 1), padding=(2, 0))) for i in range(3))
        self.post = weight_norm(nn.Conv2d(chs[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, wav):  # (B, samples)
        B, T = wav.shape
        pad = (-T) % self.period
        x = F.pad(wav, (0, pad), mode="reflect").view(B, 1, (T + pad) // self.period, self.period)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class ScaleDiscriminator(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, channels, 15, padding=7)),
            weight_norm(nn.Conv1d(channels, channels * 2, 41, stride=4, groups=4, padding=20)),
            weight_norm(nn.Conv1d(channels * 2, channels * 4, 41, stride=4, groups=8, padding=20)),
            weight_norm(nn.Conv1d(channels * 4, channels * 4, 5, padding=2)),
        ])
        self.post = weight_norm(nn.Conv1d(channels * 4, 1, 3, padding=1))

    def forward(self, wav):
        x = wav.unsqueeze(1)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class Discriminators(nn.Module):
    """MPD (periods 2, 3, 5, 7, 11) + MSD (scales 1, 2, 4). Returns lists of (logits, features)."""

    def __init__(self, periods=(2, 3, 5, 7, 11), scales: int = 3, channels: int = 16):
        super().__init__()
        self.periods = nn.ModuleList(PeriodDiscriminator(p, channels) for p in periods)
        self.scales = nn.ModuleList(ScaleDiscriminator(channels) for _ in range(scales))
        self.pool = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, wav):
        outs, feats = [], []
        for d in self.periods:
            o, f = d(wav)
            outs.append(o)
            feats.append(f)
        x = wav
        for i, d in enumerate(self.scales):
            if i > 0:
                x = self.pool(x.unsqueeze(1)).squeeze(1)
            o, f = d(x)
            outs.append(o)
            feats.append(f)
        return outs, feats


__all__ = ["Discriminators", "HiFiGANGenerator", "PeriodDiscriminator", "ResBlock", "ScaleDiscriminator"]
