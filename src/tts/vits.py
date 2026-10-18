"""VITS (Kim et al. 2021), compact: end-to-end text -> waveform in one model.

VITS is a conditional VAE whose decoder is a HiFi-GAN generator, with a normalising flow
that makes the prior expressive and MAS that aligns text to frames, all trained together:

```
training
  linear spectrogram --PosteriorEncoder--> (mu_q, logs_q) --sample--> z      (B, C, T)
  z --Flow f--> z_p                                                       (B, C, T)
  text --TextEncoder--> (mu_p, logs_p)                                    (B, C, N)
  alignment = MAS( log N(z_p ; mu_p, logs_p) )          hard, monotonic   (B, N, T)
  expand mu_p, logs_p with the alignment to T frames
  KL loss   = KL( N(z_p; ., logs_q) || N(mu_p, logs_p) )
  duration  = MSE( DurationPredictor(text), log(alignment durations) )    # deterministic (see below)
  random slice of z --HiFiGANGenerator--> wav_hat;  mel L1 + GAN + feature matching vs the real slice

inference
  text -> (mu_p, logs_p), durations -> expand -> z_p = mu_p + noise * exp(logs_p) * temperature
       -> inverse flow -> z -> generator -> waveform
```

Simplification kept on purpose: the *stochastic* duration predictor of the paper (a flow
over durations) is replaced by the deterministic FastSpeech-style predictor; the rest of
the model (posterior encoder, coupling flow, MAS, GAN decoder) is the real thing.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tts.align import mas_batch
from tts.config import Config
from tts.layers import FFTBlock, PositionalEncoding, SpeakerEmbedding, VariancePredictor, sequence_mask
from tts.vocoders import HiFiGANGenerator


class WaveNetStack(nn.Module):
    """Dilated gated convolutions with residual + skip connections (the WN of WaveGlow/VITS)."""

    def __init__(self, channels: int, kernel: int = 5, layers: int = 4, dilation_rate: int = 2):
        super().__init__()
        self.in_layers = nn.ModuleList()
        self.res_skip = nn.ModuleList()
        for i in range(layers):
            d = dilation_rate**i
            self.in_layers.append(nn.Conv1d(channels, 2 * channels, kernel, dilation=d, padding=(kernel - 1) // 2 * d))
            self.res_skip.append(nn.Conv1d(channels, 2 * channels if i < layers - 1 else channels, 1))
        self.layers = layers

    def forward(self, x, mask):
        out = torch.zeros_like(x)
        for i in range(self.layers):
            a, b = self.in_layers[i](x).chunk(2, dim=1)
            h = torch.tanh(a) * torch.sigmoid(b)
            rs = self.res_skip[i](h)
            if i < self.layers - 1:
                res, skip = rs.chunk(2, dim=1)
                x = (x + res) * mask
                out = out + skip
            else:
                out = out + rs
        return out * mask


class PosteriorEncoder(nn.Module):
    """linear spectrogram (B, n_fft/2+1, T) -> z, mu_q, logs_q (B, C, T)."""

    def __init__(self, cin: int, hidden: int, latent: int):
        super().__init__()
        self.pre = nn.Conv1d(cin, hidden, 1)
        self.wn = WaveNetStack(hidden)
        self.proj = nn.Conv1d(hidden, 2 * latent, 1)

    def forward(self, spec, mask):
        h = self.wn(self.pre(spec) * mask, mask)
        mu, logs = self.proj(h).chunk(2, dim=1)
        z = (mu + torch.randn_like(mu) * torch.exp(logs)) * mask
        return z, mu * mask, logs * mask


class CouplingLayer(nn.Module):
    """Affine coupling with mean-only transform (as in VITS): x = [x0, x1]; x1' = x1 + m(x0)."""

    def __init__(self, latent: int, hidden: int):
        super().__init__()
        self.half = latent // 2
        self.pre = nn.Conv1d(self.half, hidden, 1)
        self.wn = WaveNetStack(hidden, layers=3)
        self.post = nn.Conv1d(hidden, self.half, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x, mask, reverse=False):
        x0, x1 = x[:, :self.half], x[:, self.half:]
        m = self.post(self.wn(self.pre(x0) * mask, mask)) * mask
        x1 = x1 - m if reverse else x1 + m
        return torch.cat([x0, x1], dim=1)


class Flow(nn.Module):
    """Stack of coupling layers, flipping channel halves between them (volume preserving)."""

    def __init__(self, latent: int, hidden: int, blocks: int):
        super().__init__()
        self.layers = nn.ModuleList(CouplingLayer(latent, hidden) for _ in range(blocks))

    def forward(self, x, mask, reverse=False):
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers:
            if reverse:
                x = torch.flip(x, [1])
                x = layer(x, mask, reverse=True)
            else:
                x = layer(x, mask)
                x = torch.flip(x, [1])
        return x


class VITS(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        m, a = cfg.model, cfg.audio
        self.cfg = cfg
        H, C = m.hidden, m.latent_dim
        self.embed = nn.Embedding(vocab_size, H, padding_idx=0)
        self.pos = PositionalEncoding(H)
        self.encoder = nn.ModuleList(FFTBlock(H, m.heads, m.ffn_kernel, m.dropout) for _ in range(m.layers))
        self.speaker = SpeakerEmbedding(m.n_speakers, m.speaker_dim, H)
        self.proj = nn.Conv1d(H, 2 * C, 1)
        self.duration_pred = VariancePredictor(H, dropout=m.dropout)
        self.posterior = PosteriorEncoder(a.n_fft // 2 + 1, H, C)
        self.flow = Flow(C, H, m.flow_blocks)
        self.decoder = HiFiGANGenerator(cfg, cin=C)
        self.segment_frames = m.segment_frames

    def encode_text(self, text, text_lens, speaker):
        mask = sequence_mask(text_lens, text.size(1))
        x = self.pos(self.embed(text))
        for block in self.encoder:
            x = block(x, mask)
        x = self.speaker(x, speaker)
        mu_p, logs_p = self.proj(x.transpose(1, 2)).chunk(2, dim=1)  # (B, C, N)
        return x, mu_p * mask[:, None], logs_p * mask[:, None], mask

    @staticmethod
    def random_slice(z, lengths, seg):
        """Pick one segment of `seg` frames per item (within its length). Returns slice and start indices."""
        B = z.size(0)
        starts = [int(torch.randint(0, max(int(lengths[b]) - seg, 0) + 1, (1,))) for b in range(B)]
        return torch.stack([z[b, :, s:s + seg] for b, s in enumerate(starts)]), starts

    def forward(self, batch: dict) -> dict:
        """Returns z-slice decoded waveform, KL/duration losses and the slice starts (train.py adds the GAN)."""
        text, text_lens, spec, mel_lens = batch["text"], batch["text_lens"], batch["linear"], batch["mel_lens"]
        x, mu_p, logs_p, text_mask = self.encode_text(text, text_lens, batch.get("speaker"))
        spec_mask = sequence_mask(mel_lens, spec.size(2))[:, None]  # (B, 1, T)
        z, _mu_q, logs_q = self.posterior(spec, spec_mask)
        z_p = self.flow(z, spec_mask)
        with torch.no_grad():  # log N(z_p ; mu_p, logs_p) for every (text n, frame t): (B, N, T)
            s_p_sq_r = torch.exp(-2 * logs_p)  # (B, C, N)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, 1, keepdim=True).transpose(1, 2)  # (B,N,1)
            neg_cent2 = torch.matmul((-0.5 * s_p_sq_r).transpose(1, 2), z_p**2)  # (B, N, T)
            neg_cent3 = torch.matmul((mu_p * s_p_sq_r).transpose(1, 2), z_p)
            neg_cent4 = torch.sum(-0.5 * mu_p**2 * s_p_sq_r, 1, keepdim=True).transpose(1, 2)
            logp = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            logp = logp.masked_fill(~(text_mask[:, :, None] & spec_mask), -1e9)
            attn = mas_batch(logp, text_lens, mel_lens)  # (B, N, T)
        durations = attn.sum(2)
        log_dur = self.duration_pred(x, text_mask)
        l_dur = (F.mse_loss(log_dur, torch.log(durations + 1), reduction="none") * text_mask).sum() / text_mask.sum()
        mu_p_t = torch.matmul(mu_p, attn)  # (B, C, T): expand to frames
        logs_p_t = torch.matmul(logs_p, attn)
        kl = logs_p_t - logs_q - 0.5 + 0.5 * (z_p - mu_p_t) ** 2 * torch.exp(-2.0 * logs_p_t)
        l_kl = (kl * spec_mask).sum() / spec_mask.sum() / z.size(1)
        z_slice, starts = self.random_slice(z, mel_lens, self.segment_frames)
        wav_hat = self.decoder(z_slice)
        return {"wav_hat": wav_hat, "starts": starts, "kl": l_kl, "duration": l_dur, "attention": attn.transpose(1, 2),
                "durations": durations.long()}

    @torch.no_grad()
    def infer(self, text, speaker=None, duration_scale: float = 1.0, temperature: float = 0.667):
        text_lens = torch.tensor([text.size(1)], device=text.device)
        x, mu_p, logs_p, mask = self.encode_text(text, text_lens, speaker)
        log_dur = self.duration_pred(x, mask)
        durations = torch.clamp(torch.round((torch.exp(log_dur) - 1) * duration_scale), min=1).long() * mask
        attn = torch.repeat_interleave(torch.eye(text.size(1), device=text.device), durations[0], dim=0)  # (T, N)
        mu_p_t, logs_p_t = mu_p[0] @ attn.T, logs_p[0] @ attn.T  # (C, T)
        z_p = (mu_p_t + torch.randn_like(mu_p_t) * torch.exp(logs_p_t) * temperature)[None]
        ones = torch.ones(1, 1, z_p.size(2), device=text.device)
        z = self.flow(z_p, ones, reverse=True)
        return self.decoder(z)[0], attn  # (samples,), (T, N)


__all__ = ["VITS", "CouplingLayer", "Flow", "PosteriorEncoder", "WaveNetStack"]
