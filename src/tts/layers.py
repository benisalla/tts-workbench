"""Building blocks shared by the models.

- `PreNet`: two dropout-ed linear layers on the previous mel frame (Tacotron 2). The dropout
  stays on at inference: it is what stops the decoder from simply copying its last frame.
- `PostNet`: a stack of 1-D convolutions that predicts a residual to sharpen the coarse mel.
- `LocationSensitiveAttention`: Tacotron 2's attention. The score for each text position
  depends on the query, the encoder output *and the attention of the previous step*
  (convolved), which pushes the alignment to move forward monotonically.
- `FFTBlock`: the FastSpeech "feed-forward transformer" block: multi-head self-attention
  + a 1-D convolution feed-forward (convolutions see neighbouring phonemes, which linear
  layers cannot).
- `LengthRegulator`: repeats each phoneme's hidden state `duration` times (the non-
  autoregressive way to go from N phonemes to T frames).
- `VariancePredictor`: 2 conv layers + linear that predict duration / pitch / energy per phoneme.
- `SpeakerEmbedding`: a learned vector per speaker, added to the encoder output.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """(batch, max_len) bool, True where t < length."""
    max_len = max_len or int(lengths.max())
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class ConvNorm(nn.Module):
    """Conv1d with 'same' padding, optional activation and dropout."""

    def __init__(self, cin: int, cout: int, kernel: int = 1, dilation: int = 1, act: nn.Module | None = None,
                 dropout: float = 0.0, norm: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel, padding=(kernel - 1) // 2 * dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(cout) if norm else nn.Identity()
        self.act = act or nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class PreNet(nn.Module):
    def __init__(self, cin: int, dim: int, dropout: float = 0.5):
        super().__init__()
        self.fc1, self.fc2, self.p = nn.Linear(cin, dim), nn.Linear(dim, dim), dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dropout is applied even in eval mode (Tacotron 2): it keeps the decoder from copying
        x = F.dropout(F.relu(self.fc1(x)), self.p, training=True)
        return F.dropout(F.relu(self.fc2(x)), self.p, training=True)


class PostNet(nn.Module):
    """n_mels -> channels -> ... -> n_mels residual, tanh except the last layer."""

    def __init__(self, n_mels: int, channels: int, layers: int, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        dims = [n_mels] + [channels] * (layers - 1) + [n_mels]
        self.convs = nn.ModuleList(
            ConvNorm(dims[i], dims[i + 1], kernel, act=nn.Tanh() if i < layers - 1 else None, dropout=dropout)
            for i in range(layers))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # (batch, n_mels, frames)
        x = mel
        for conv in self.convs:
            x = conv(x)
        return x


class LocationSensitiveAttention(nn.Module):
    """e_{t,i} = v^T tanh(W q_t + V h_i + U f_{t,i}),   f_t = conv(alpha_{t-1} ++ sum_{<t} alpha)."""

    def __init__(self, query_dim: int, memory_dim: int, attn_dim: int, filters: int, kernel: int):
        super().__init__()
        self.query = nn.Linear(query_dim, attn_dim, bias=False)
        self.memory = nn.Linear(memory_dim, attn_dim, bias=False)
        self.location_conv = nn.Conv1d(2, filters, kernel, padding=(kernel - 1) // 2, bias=False)
        self.location = nn.Linear(filters, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, processed_memory: torch.Tensor,
                prev_attn: torch.Tensor, cum_attn: torch.Tensor, mask: torch.Tensor):
        """query (B, Q); memory (B, N, M); prev/cum attn (B, N); mask (B, N) True = valid.
        Returns context (B, M) and alignment (B, N)."""
        loc = self.location_conv(torch.stack([prev_attn, cum_attn], dim=1)).transpose(1, 2)  # (B, N, filters)
        energies = self.v(torch.tanh(self.query(query)[:, None] + processed_memory + self.location(loc))).squeeze(-1)
        energies = energies.masked_fill(~mask, -1e4)
        alignment = torch.softmax(energies, dim=-1)
        context = torch.bmm(alignment.unsqueeze(1), memory).squeeze(1)
        return context, alignment


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len)[:, None].float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
        self.alpha = nn.Parameter(torch.ones(1))  # Transformer-TTS: learned scale of the encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, dim)
        return x + self.alpha * self.pe[: x.size(1)]


class FFTBlock(nn.Module):
    """Self-attention + conv feed-forward, pre-norm residuals, key padding mask."""

    def __init__(self, dim: int, heads: int, kernel: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Conv1d(dim, dim * 4, kernel, padding=(kernel - 1) // 2), nn.ReLU(),
                                nn.Dropout(dropout), nn.Conv1d(dim * 4, dim, kernel, padding=(kernel - 1) // 2))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # mask (B, T) True = valid
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + self.drop(h)
        h = self.ff(self.ln2(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.drop(h)
        return x * mask[..., None]


class LengthRegulator(nn.Module):
    """(B, N, D) + integer durations (B, N) -> (B, T, D) with T = max(sum durations)."""

    def forward(self, x: torch.Tensor, durations: torch.Tensor):
        out, lengths = [], durations.sum(dim=1)
        max_len = int(lengths.max())
        for b in range(x.size(0)):
            expanded = torch.repeat_interleave(x[b], durations[b], dim=0)
            out.append(F.pad(expanded, (0, 0, 0, max_len - expanded.size(0))))
        return torch.stack(out), lengths


class VariancePredictor(nn.Module):
    def __init__(self, dim: int, filters: int = 128, kernel: int = 3, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, filters, kernel, padding=(kernel - 1) // 2), nn.ReLU(), nn.GroupNorm(1, filters),
            nn.Dropout(dropout), nn.Conv1d(filters, filters, kernel, padding=(kernel - 1) // 2), nn.ReLU(),
            nn.GroupNorm(1, filters), nn.Dropout(dropout))
        self.out = nn.Linear(filters, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # (B, N, D) -> (B, N)
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.out(h).squeeze(-1) * mask


class SpeakerEmbedding(nn.Module):
    """Learned speaker vectors, projected and added to encoder states (no-op for n_speakers == 1)."""

    def __init__(self, n_speakers: int, speaker_dim: int, dim: int):
        super().__init__()
        self.enabled = n_speakers > 1
        if self.enabled:
            self.table = nn.Embedding(n_speakers, speaker_dim)
            self.proj = nn.Linear(speaker_dim, dim)

    def forward(self, x: torch.Tensor, speaker: torch.Tensor | None) -> torch.Tensor:  # x (B, N, dim)
        if not self.enabled or speaker is None:
            return x
        return x + self.proj(self.table(speaker))[:, None, :]


def guided_attention_loss(attn: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor,
                          sigma: float = 0.2) -> torch.Tensor:
    """Penalise attention far from the diagonal: W[n, t] = 1 - exp(-(n/N - t/T)^2 / 2 sigma^2).
    attn (B, T, N). Used in Transformer-TTS / Tacotron 2 training to make alignment learning fast."""
    _B, T, N = attn.shape
    n = torch.arange(N, device=attn.device)[None, None, :].float()
    t = torch.arange(T, device=attn.device)[None, :, None].float()
    w = 1.0 - torch.exp(-((n / text_lens[:, None, None] - t / mel_lens[:, None, None]) ** 2) / (2 * sigma**2))
    mask = sequence_mask(mel_lens, T)[:, :, None] & sequence_mask(text_lens, N)[:, None, :]
    return (attn * w * mask).sum() / mask.sum()


__all__ = [
    "ConvNorm",
    "FFTBlock",
    "LengthRegulator",
    "LocationSensitiveAttention",
    "PositionalEncoding",
    "PostNet",
    "PreNet",
    "SpeakerEmbedding",
    "VariancePredictor",
    "guided_attention_loss",
    "sequence_mask",
]
