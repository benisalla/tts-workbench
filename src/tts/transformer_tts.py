"""Transformer-TTS (Li et al. 2019): Tacotron 2 with the RNNs and the location attention
replaced by a Transformer encoder-decoder.

```
text -> Embedding -> conv prenet -> + scaled positional encoding -> L x [self-attn, FFN]  (encoder)
mel frames (shifted right) -> PreNet -> + positional encoding
   -> L x [causal self-attn, cross-attn to the encoder, FFN]                           (decoder)
   -> Linear: r mel frames + stop logit ;  PostNet residual
```

Pros: every decoder step sees every earlier frame (no LSTM bottleneck) and training is
parallel over time (one causal mask instead of a Python loop). Cons: nothing forces the
cross-attention to be monotonic, so a diagonal *guided-attention loss* is important, and
inference is still frame-by-frame (slow for long sentences), which FastSpeech 2 fixes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tts.config import Config
from tts.layers import (
    ConvNorm,
    FFTBlock,
    PositionalEncoding,
    PostNet,
    PreNet,
    SpeakerEmbedding,
    guided_attention_loss,
    sequence_mask,
)


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask, mem_mask):
        h = self.ln1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.drop(h)
        h = self.ln2(x)
        h, attn = self.cross_attn(h, memory, memory, key_padding_mask=~mem_mask, need_weights=True)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.ln3(x)))
        return x, attn  # attn (B, T, N) averaged over heads


class TransformerTTS(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        m, a = cfg.model, cfg.audio
        self.cfg, self.n_mels, self.r = cfg, a.n_mels, m.reduction_factor
        H = m.hidden
        self.embed = nn.Embedding(vocab_size, H, padding_idx=0)
        self.enc_prenet = nn.Sequential(*[ConvNorm(H, H, 5, act=nn.ReLU(), dropout=m.dropout) for _ in range(3)])
        self.enc_pos, self.dec_pos = PositionalEncoding(H), PositionalEncoding(H)
        self.encoder = nn.ModuleList(FFTBlock(H, m.heads, 1, m.dropout) for _ in range(m.layers))
        self.speaker = SpeakerEmbedding(m.n_speakers, m.speaker_dim, H)
        self.dec_prenet = PreNet(a.n_mels, m.prenet_dim)
        self.dec_in = nn.Linear(m.prenet_dim, H)
        self.decoder = nn.ModuleList(DecoderLayer(H, m.heads, m.dropout) for _ in range(m.layers))
        self.ln = nn.LayerNorm(H)
        self.mel_out = nn.Linear(H, a.n_mels * self.r)
        self.stop_out = nn.Linear(H, self.r)
        self.postnet = PostNet(a.n_mels, m.postnet_channels, m.postnet_layers, dropout=m.dropout)

    def encode(self, text, text_lens, speaker):
        mask = sequence_mask(text_lens, text.size(1))
        x = self.enc_prenet(self.embed(text).transpose(1, 2)).transpose(1, 2)
        x = self.enc_pos(x)
        for block in self.encoder:
            x = block(x, mask)
        return self.speaker(x, speaker), mask

    def decode(self, frames_in, memory, mem_mask):
        """frames_in (B, S, n_mels): the frame that precedes each decoder step (zeros first)."""
        S = frames_in.size(1)
        x = self.dec_pos(self.dec_in(self.dec_prenet(frames_in)))
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), 1)
        attn = None
        for layer in self.decoder:
            x, attn = layer(x, memory, causal, mem_mask)
        x = self.ln(x)
        return self.mel_out(x), self.stop_out(x), attn

    def forward(self, batch: dict) -> dict:
        mel, _mel_lens, text_lens = batch["mel"], batch["mel_lens"], batch["text_lens"]
        B, _, T = mel.shape
        pad = (-T) % self.r
        frames = F.pad(mel, (0, pad)).transpose(1, 2)  # (B, T_pad, n_mels)
        groups = frames.view(B, -1, self.r, self.n_mels)  # (B, S, r, n_mels)
        frames_in = torch.cat([frames.new_zeros(B, 1, self.n_mels), groups[:, :-1, -1]], 1)
        memory, mem_mask = self.encode(batch["text"], text_lens, batch.get("speaker"))
        m_out, s_out, attn = self.decode(frames_in, memory, mem_mask)
        mel_before = m_out.view(B, -1, self.n_mels).transpose(1, 2)[:, :, :T]
        stop_logits = s_out.reshape(B, -1)[:, :T]
        mel_after = mel_before + self.postnet(mel_before)
        return self.loss(mel_before, mel_after, stop_logits, attn, batch) | {"mel": mel_after, "attention": attn}

    def loss(self, before, after, stop_logits, attn, batch) -> dict:
        mel, mel_lens, text_lens = batch["mel"], batch["mel_lens"], batch["text_lens"]
        mask = sequence_mask(mel_lens, mel.size(2))[:, None, :]
        n = mask.sum() * mel.size(1)
        l_before = (F.mse_loss(before, mel, reduction="none") * mask).sum() / n
        l_after = (F.mse_loss(after, mel, reduction="none") * mask).sum() / n
        stop_target = (torch.arange(mel.size(2), device=mel.device)[None] >= (mel_lens - 1)[:, None]).float()
        l_stop = F.binary_cross_entropy_with_logits(stop_logits, stop_target, pos_weight=torch.tensor(5.0))
        total = l_before + l_after + l_stop
        out = {"mel_before": l_before, "mel_after": l_after, "stop": l_stop}
        if self.cfg.train.guided_attention > 0:
            steps = torch.div(mel_lens + self.r - 1, self.r, rounding_mode="floor")
            l_ga = guided_attention_loss(attn, text_lens, steps)
            total = total + self.cfg.train.guided_attention * l_ga
            out["guided_attention"] = l_ga
        out["loss"] = total
        return out

    @torch.no_grad()
    def infer(self, text, speaker=None, max_steps=None):
        """Autoregressive: re-run the decoder on all frames so far at each step (O(T^2), simple)."""
        text_lens = torch.tensor([text.size(1)], device=text.device)
        memory, mem_mask = self.encode(text, text_lens, speaker)
        frames_in = memory.new_zeros(1, 1, self.n_mels)
        attn = None
        for _ in range(max_steps or self.cfg.model.max_decoder_steps):
            m_out, s_out, attn = self.decode(frames_in, memory, mem_mask)
            last = m_out[:, -1].view(1, self.r, self.n_mels)
            frames_in = torch.cat([frames_in, last[:, -1:]], 1)
            if torch.sigmoid(s_out[:, -1]).max() > 0.5:
                break
        mel = m_out.view(1, -1, self.n_mels).transpose(1, 2)
        return mel + self.postnet(mel), attn[0]


__all__ = ["DecoderLayer", "TransformerTTS"]
