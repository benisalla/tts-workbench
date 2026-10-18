"""Alignment between text and frames, without an external tool.

A non-autoregressive model (FastSpeech 2) needs to know, for each phoneme, how many
frames it lasts. Three ways to get these *durations* live here:

1. **Monotonic Alignment Search (MAS)**, from Glow-TTS. Given a log-likelihood matrix
   `logp[n, t]` (how well phoneme n explains frame t), dynamic programming finds the
   monotonic, non-skipping path N->T that maximises the total likelihood. It runs in
   O(N*T) with Python loops, which is fine at our sizes (`mas`).
2. **An aligner network** (as in "One TTS Alignment To Rule Them All"): a small text
   encoder and mel encoder give a soft alignment `softmax_n(-||mel_t - text_n||^2)`,
   trained with the *forward-sum* loss (the CTC forward algorithm over text positions:
   every monotonic path is summed, no blank). MAS over its log-probabilities yields hard
   durations. VITS uses the same MAS on the likelihood of its latent under the text prior.
3. **Teacher durations** from the attention matrix of a trained Tacotron 2
   (`durations_from_attention`): count how many frames each phoneme received the argmax.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tts.layers import ConvNorm


def mas(logp: np.ndarray, text_len: int, mel_len: int) -> np.ndarray:
    """Monotonic alignment search. logp (N, T) -> hard alignment (N, T) in {0, 1}.

    Each frame t is assigned to exactly one text position n, n never decreases and every
    text position gets at least one frame (needs T >= N).
    """
    neg = -1e9
    N, T = text_len, mel_len
    q = np.full((N, T), neg, dtype=np.float64)
    q[0, 0] = logp[0, 0]
    for t in range(1, T):
        lo, hi = max(0, N - T + t), min(N, t + 1)  # positions reachable at frame t
        for n in range(lo, hi):
            stay = q[n, t - 1]
            move = q[n - 1, t - 1] if n > 0 else neg
            q[n, t] = max(stay, move) + logp[n, t]
    path = np.zeros((N, T), dtype=np.int64)
    n = N - 1
    for t in range(T - 1, -1, -1):
        path[n, t] = 1
        if n > 0 and t > 0 and (q[n - 1, t - 1] > q[n, t - 1] or n == t):
            n -= 1
    return path


def mas_batch(logp: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
    """logp (B, N, T) -> (B, N, T) hard alignment (float, zeros outside each item's lengths)."""
    out = torch.zeros_like(logp)
    lp = logp.detach().cpu().numpy()
    for b in range(logp.size(0)):
        n, t = int(text_lens[b]), int(mel_lens[b])
        out[b, :n, :t] = torch.from_numpy(mas(lp[b, :n, :t], n, t)).to(logp.device, logp.dtype)
    return out


def forward_sum_loss(log_probs: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
    """log_probs (B, N, T) log-softmax over N per frame -> mean negative log of the sum of all monotonic
    paths. Implemented with CTC: text = [1..N] with no repeats and a blank that can never be used."""
    B, N, T = log_probs.shape
    # CTC wants (T, B, C) with class 0 = blank; shift text ids by one and give the blank -inf
    blank = torch.full((B, 1, T), -1e4, device=log_probs.device, dtype=log_probs.dtype)
    lp = torch.cat([blank, log_probs], dim=1).permute(2, 0, 1).log_softmax(-1)  # renormalise with blank
    targets = torch.arange(1, N + 1, device=log_probs.device)[None].expand(B, N)
    return F.ctc_loss(lp, targets, mel_lens, text_lens, blank=0, reduction="mean", zero_infinity=True)


def diagonal_log_prior(text_lens: torch.Tensor, mel_lens: torch.Tensor, N: int, T: int, sigma: float = 0.15
                       ) -> torch.Tensor:
    """log of a Gaussian band around the diagonal, (B, N, T): -((n+.5)/N_b - (t+.5)/T_b)^2 / (2 sigma^2).

    Added to the aligner's logits early in training (RAD-TTS uses a beta-binomial prior for the
    same purpose): it tells the aligner roughly where frames belong before it has learnt anything,
    so the forward-sum loss has a gradient to follow instead of a flat alignment."""
    n = (torch.arange(N, device=text_lens.device)[None, :, None] + 0.5) / text_lens[:, None, None]
    t = (torch.arange(T, device=text_lens.device)[None, None, :] + 0.5) / mel_lens[:, None, None]
    return -((n - t) ** 2) / (2 * sigma**2)


class Aligner(nn.Module):
    """Text (B, N, D) and mel (B, n_mels, T) -> log-probabilities (B, N, T), log-softmax over text positions.

    Logits are the negative squared distance between projected text and mel features, scaled by a
    small `temperature` (RAD-TTS uses 0.0005): large distances must not make the soft alignment
    one-hot before the features mean anything.

    How it is trained (see `FastSpeech2.forward`): the forward-sum loss alone has a degenerate
    optimum where one symbol absorbs every frame, and at our scale gradient descent finds it.
    So for `bin_warmup_steps` the aligner is first taught the *diagonal prior* (MAS on the prior
    alone, i.e. roughly uniform durations) as hard labels; then MAS on its own log-probabilities
    gives the labels (hard EM, the Glow-TTS recipe) and the forward-sum loss is ramped in.
    """

    def __init__(self, text_dim: int, n_mels: int, dim: int = 64, temperature: float = 0.005):
        super().__init__()
        self.text = nn.Sequential(ConvNorm(text_dim, dim, 3, act=nn.ReLU(), norm=False), nn.Conv1d(dim, dim, 1))
        self.mel = nn.Sequential(ConvNorm(n_mels, dim, 3, act=nn.ReLU(), norm=False),
                                 ConvNorm(dim, dim, 3, act=nn.ReLU(), norm=False), nn.Conv1d(dim, dim, 1))
        self.temperature = temperature

    def forward(self, text: torch.Tensor, mel: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        h_text = self.text(text.transpose(1, 2))  # (B, D, N)
        h_mel = self.mel(mel)  # (B, D, T)
        dist = ((h_mel[:, :, None, :] - h_text[:, :, :, None]) ** 2).sum(1)  # (B, N, T)
        logits = (-dist * self.temperature).masked_fill(~text_mask[:, :, None], -1e4)
        return logits.log_softmax(dim=1)  # over text positions

    def align(self, log_probs: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor, text_mask: torch.Tensor,
              bin_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Hard alignment (B, N, T) and the aligner losses for the current training phase.

        bin_weight 0 = warm-up (labels from the diagonal prior), ramping to 1 = labels from MAS on
        the aligner's own log-probabilities (with a fading prior) plus the forward-sum loss."""
        _, N, T = log_probs.shape
        with torch.no_grad():
            prior = diagonal_log_prior(text_lens, mel_lens, N, T).masked_fill(~text_mask[:, :, None], -1e4)
            guide = prior if bin_weight <= 0 else log_probs + (1.0 - bin_weight) * prior
            hard = mas_batch(guide, text_lens, mel_lens)
        losses = {"bin": -(hard * log_probs).sum() / hard.sum()}
        if bin_weight > 0:
            losses["forward_sum"] = bin_weight * forward_sum_loss(log_probs, text_lens, mel_lens)
        return hard, losses


def durations_from_alignment(hard: torch.Tensor) -> torch.Tensor:
    """(B, N, T) hard alignment -> integer durations (B, N)."""
    return hard.sum(dim=2).long()


def durations_from_attention(attn: torch.Tensor, text_len: int, mel_len: int) -> torch.Tensor:
    """Teacher durations from an AR attention matrix (T, N): frames whose argmax is n, forced monotonic
    by MAS on the log-attention so every phoneme has >= 1 frame."""
    logp = torch.log(attn[:mel_len, :text_len].clamp_min(1e-8)).T.detach().cpu().numpy()
    return torch.from_numpy(mas(logp, text_len, mel_len).sum(axis=1))


__all__ = [
    "Aligner",
    "durations_from_alignment",
    "durations_from_attention",
    "forward_sum_loss",
    "mas",
    "mas_batch",
]
