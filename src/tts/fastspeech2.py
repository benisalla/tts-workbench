"""FastSpeech 2 (Ren et al. 2020): non-autoregressive text -> mel, all frames at once.

```
text -> Embedding + positional encoding -> L x FFTBlock                    (encoder, (B, N, H))
     -> variance adaptor:
          duration predictor  d_n  (log frames per phoneme)      target: aligner / teacher
          pitch predictor     p_n  -> + pitch embedding            target: mean F0 of the phoneme's frames
          energy predictor    e_n  -> + energy embedding           target: mean energy of those frames
     -> LengthRegulator: repeat state n  d_n times                 (B, T, H)
     -> + positional encoding -> L x FFTBlock -> Linear -> mel     (decoder)
     -> + PostNet residual
```

There is no attention between text and frames, so the model *must* be told the durations.
By default a small **aligner** (see `align.py`) is trained jointly (diagonal-prior warm-up,
then hard-EM with monotonic alignment search and the forward-sum loss) and gives hard durations; `--duration-source
teacher` uses durations cached from a Tacotron 2 attention instead. Inference is one
forward pass (hundreds of times faster than an AR decoder), fully controllable (scale the
durations to speak slower, shift the pitch), and never loops or repeats words.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tts.align import Aligner, durations_from_alignment
from tts.config import Config
from tts.layers import (
    FFTBlock,
    LengthRegulator,
    PositionalEncoding,
    PostNet,
    SpeakerEmbedding,
    VariancePredictor,
    sequence_mask,
)


def average_by_duration(values: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """Frame-level (B, T) -> phoneme-level (B, N) mean over each phoneme's frames."""
    B, N = durations.shape
    out = values.new_zeros(B, N)
    for b in range(B):
        ends = durations[b].cumsum(0)
        starts = ends - durations[b]
        for n in range(N):
            if durations[b, n] > 0:
                seg = values[b, starts[n]:ends[n]]
                voiced = seg[seg > 0]
                out[b, n] = voiced.mean() if voiced.numel() else 0.0
    return out


class FastSpeech2(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        m, a = cfg.model, cfg.audio
        self.cfg = cfg
        H = m.hidden
        self.embed = nn.Embedding(vocab_size, H, padding_idx=0)
        self.enc_pos, self.dec_pos = PositionalEncoding(H), PositionalEncoding(H)
        self.encoder = nn.ModuleList(FFTBlock(H, m.heads, m.ffn_kernel, m.dropout) for _ in range(m.layers))
        self.decoder = nn.ModuleList(FFTBlock(H, m.heads, m.ffn_kernel, m.dropout) for _ in range(m.layers))
        self.speaker = SpeakerEmbedding(m.n_speakers, m.speaker_dim, H)
        self.duration_pred = VariancePredictor(H, dropout=m.dropout)
        self.pitch_pred = VariancePredictor(H, dropout=m.dropout)
        self.energy_pred = VariancePredictor(H, dropout=m.dropout)
        self.pitch_embed = nn.Conv1d(1, H, 9, padding=4)
        self.energy_embed = nn.Conv1d(1, H, 9, padding=4)
        self.length_regulator = LengthRegulator()
        self.mel_out = nn.Linear(H, a.n_mels)
        self.postnet = PostNet(a.n_mels, m.postnet_channels, m.postnet_layers, dropout=m.dropout)
        self.aligner = Aligner(H, a.n_mels) if cfg.train.duration_source == "aligner" else None
        self.bin_weight = 0.0  # set by the trainer: 0 during the aligner warm-up, then ramped to 1

    def encode(self, text, text_lens, speaker):
        mask = sequence_mask(text_lens, text.size(1))
        x = self.enc_pos(self.embed(text))
        for block in self.encoder:
            x = block(x, mask)
        return self.speaker(x, speaker), mask

    def decode(self, x, durations):
        x, mel_lens = self.length_regulator(x, durations)
        mask = sequence_mask(mel_lens, x.size(1))
        x = self.dec_pos(x)
        for block in self.decoder:
            x = block(x, mask)
        mel = self.mel_out(x).transpose(1, 2)  # (B, n_mels, T)
        return mel, mel + self.postnet(mel), mel_lens

    def variance(self, x, mask, pitch=None, energy=None):
        """Predict log-duration, pitch, energy; add pitch/energy embeddings (targets if given)."""
        log_dur = self.duration_pred(x, mask)
        p = self.pitch_pred(x, mask)
        e = self.energy_pred(x, mask)
        x = x + self.pitch_embed((pitch if pitch is not None else p).unsqueeze(1)).transpose(1, 2)
        x = x + self.energy_embed((energy if energy is not None else e).unsqueeze(1)).transpose(1, 2)
        return x, log_dur, p, e

    def forward(self, batch: dict) -> dict:
        text, text_lens, mel, mel_lens = batch["text"], batch["text_lens"], batch["mel"], batch["mel_lens"]
        x, mask = self.encode(text, text_lens, batch.get("speaker"))
        out = {}
        if self.aligner is not None:
            log_probs = self.aligner(self.embed(text), mel, mask)  # (B, N, T)
            hard, aligner_losses = self.aligner.align(log_probs, text_lens, mel_lens, mask, self.bin_weight)
            out.update(aligner_losses)
            durations = durations_from_alignment(hard)
        else:
            durations = batch["durations"]
        pitch_t = average_by_duration(torch.log1p(batch["f0"]), durations)  # log-F0 per phoneme (0 = unvoiced)
        energy_t = average_by_duration(batch["energy"], durations)
        x, log_dur, p, e = self.variance(x, mask, pitch_t, energy_t)
        before, after, _ = self.decode(x, durations)
        T = mel.size(2)
        before, after = before[:, :, :T], after[:, :, :T]
        if before.size(2) < T:
            before, after = F.pad(before, (0, T - before.size(2))), F.pad(after, (0, T - after.size(2)))
        fmask = sequence_mask(mel_lens, T)[:, None, :]
        n = fmask.sum() * mel.size(1)
        out["mel_before"] = (F.l1_loss(before, mel, reduction="none") * fmask).sum() / n
        out["mel_after"] = (F.l1_loss(after, mel, reduction="none") * fmask).sum() / n
        out["duration"] = (F.mse_loss(log_dur, torch.log(durations.float() + 1), reduction="none") * mask).sum() / mask.sum()
        out["pitch"] = (F.mse_loss(p, pitch_t, reduction="none") * mask).sum() / mask.sum()
        out["energy"] = (F.mse_loss(e, energy_t, reduction="none") * mask).sum() / mask.sum()
        out["loss"] = sum(v for k, v in out.items())
        out["mel"], out["durations"] = after, durations
        if self.aligner is not None:
            out["attention"] = hard.transpose(1, 2)  # (B, T, N) like the AR models
        return out

    @torch.no_grad()
    def infer(self, text, speaker=None, duration_scale: float = 1.0):
        text_lens = torch.tensor([text.size(1)], device=text.device)
        x, mask = self.encode(text, text_lens, speaker)
        x, log_dur, _p, _e = self.variance(x, mask)
        durations = torch.clamp(torch.round((torch.exp(log_dur) - 1) * duration_scale), min=1).long() * mask
        _, mel, _ = self.decode(x, durations)
        hard = torch.repeat_interleave(torch.eye(text.size(1), device=text.device), durations[0], dim=0)
        return mel, hard  # (1, n_mels, T), (T, N)


__all__ = ["FastSpeech2", "average_by_duration"]
