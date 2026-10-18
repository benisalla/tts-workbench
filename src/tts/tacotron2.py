"""Tacotron 2 (Shen et al. 2018): autoregressive attention-based text -> mel.

```
text ids -> Embedding -> 3 x Conv1d -> BiLSTM = encoder memory h (B, N, hidden)
frame 0 = zeros
for each decoder step t:
    p = PreNet(previous frame)                       # dropout always on
    attention LSTM: s_t = LSTM([p, context_{t-1}])
    alignment alpha_t = LocationSensitiveAttention(s_t, h, alpha_{t-1}, cumulative alpha)
    context_t = sum_i alpha_t[i] * h_i
    decoder LSTM: d_t = LSTM([s_t, context_t])
    mel_t (r frames) = Linear([d_t, context_t]);  stop_t = sigmoid(Linear([d_t, context_t]))
mel = mel + PostNet(mel)                            # residual refinement
```

Losses: MSE before and after the postnet, BCE on the stop token (positive at the last step),
and an optional guided-attention loss that pulls the alignment to the diagonal early on.
Training is *teacher-forced*: the previous frame comes from the recording, not from the
model; at inference the model feeds itself, which is why the prenet dropout matters.
`reduction_factor` r predicts r frames per step: fewer steps, easier attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tts.config import Config
from tts.layers import (
    ConvNorm,
    LocationSensitiveAttention,
    PostNet,
    PreNet,
    SpeakerEmbedding,
    guided_attention_loss,
    sequence_mask,
)


class Tacotron2Encoder(nn.Module):
    def __init__(self, vocab: int, hidden: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden, padding_idx=0)
        self.convs = nn.Sequential(*[ConvNorm(hidden, hidden, 5, act=nn.ReLU(), dropout=dropout) for _ in range(3)])
        self.lstm = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)

    def forward(self, text: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.convs(self.embed(text).transpose(1, 2)).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=text.size(1))
        return out


class Tacotron2(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        m, a = cfg.model, cfg.audio
        self.cfg, self.n_mels, self.r = cfg, a.n_mels, m.reduction_factor
        H = m.hidden
        self.encoder = Tacotron2Encoder(vocab_size, H, m.dropout)
        self.speaker = SpeakerEmbedding(m.n_speakers, m.speaker_dim, H)
        self.prenet = PreNet(a.n_mels, m.prenet_dim)
        self.attention_rnn = nn.LSTMCell(m.prenet_dim + H, H)
        self.attention = LocationSensitiveAttention(H, H, m.attention_dim, m.attention_filters, m.attention_kernel)
        self.decoder_rnn = nn.LSTMCell(2 * H, H)
        self.mel_out = nn.Linear(2 * H, a.n_mels * self.r)
        self.stop_out = nn.Linear(2 * H, self.r)
        self.postnet = PostNet(a.n_mels, m.postnet_channels, m.postnet_layers, dropout=m.dropout)
        self.dropout = m.dropout

    # -- one decoder step -------------------------------------------------
    def _init_state(self, memory: torch.Tensor):
        B, N, H = memory.shape
        z = memory.new_zeros
        return {"attn_h": z(B, H), "attn_c": z(B, H), "dec_h": z(B, H), "dec_c": z(B, H), "context": z(B, H),
                    "attn": z(B, N), "cum_attn": z(B, N)}

    def _step(self, prev_frame: torch.Tensor, state: dict, memory: torch.Tensor, processed: torch.Tensor,
              mask: torch.Tensor):
        p = self.prenet(prev_frame)
        state["attn_h"], state["attn_c"] = self.attention_rnn(torch.cat([p, state["context"]], -1),
                                                              (state["attn_h"], state["attn_c"]))
        query = F.dropout(state["attn_h"], self.dropout, self.training)
        context, attn = self.attention(query, memory, processed, state["attn"], state["cum_attn"], mask)
        state.update(context=context, attn=attn, cum_attn=state["cum_attn"] + attn)
        state["dec_h"], state["dec_c"] = self.decoder_rnn(torch.cat([query, context], -1),
                                                          (state["dec_h"], state["dec_c"]))
        out = torch.cat([F.dropout(state["dec_h"], self.dropout, self.training), context], -1)
        return self.mel_out(out), self.stop_out(out), attn

    def encode(self, text: torch.Tensor, text_lens: torch.Tensor, speaker: torch.Tensor | None):
        memory = self.speaker(self.encoder(text, text_lens), speaker)
        return memory, self.attention.memory(memory), sequence_mask(text_lens, text.size(1))

    # -- training (teacher forcing) --------------------------------------
    def forward(self, batch: dict) -> dict:
        mel, _mel_lens = batch["mel"], batch["mel_lens"]  # (B, n_mels, T)
        B, _, T = mel.shape
        pad = (-T) % self.r
        mel = F.pad(mel, (0, pad))
        T_pad = T + pad
        memory, processed, mask = self.encode(batch["text"], batch["text_lens"], batch.get("speaker"))
        state = self._init_state(memory)
        frames = mel.transpose(1, 2)  # (B, T_pad, n_mels)
        prev = frames.new_zeros(B, self.n_mels)
        mels, stops, attns = [], [], []
        for t in range(0, T_pad, self.r):
            m_out, s_out, attn = self._step(prev, state, memory, processed, mask)
            mels.append(m_out.view(B, self.r, self.n_mels))
            stops.append(s_out)
            attns.append(attn)
            prev = frames[:, t + self.r - 1]  # last frame of the group, from the recording
        mel_before = torch.cat(mels, 1).transpose(1, 2)[:, :, :T]
        stop_logits = torch.cat(stops, 1)[:, :T]
        mel_after = mel_before + self.postnet(mel_before)
        attn = torch.stack(attns, 1)  # (B, steps, N)
        return self.loss(mel_before, mel_after, stop_logits, attn, batch) | {
            "mel": mel_after, "attention": attn}

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

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def infer(self, text: torch.Tensor, speaker: torch.Tensor | None = None, max_steps: int | None = None):
        """text (1, N) -> mel (1, n_mels, T), attention (T/r, N)."""
        text_lens = torch.tensor([text.size(1)], device=text.device)
        memory, processed, mask = self.encode(text, text_lens, speaker)
        state = self._init_state(memory)
        prev = memory.new_zeros(1, self.n_mels)
        mels, attns = [], []
        for _ in range(max_steps or self.cfg.model.max_decoder_steps):
            m_out, s_out, attn = self._step(prev, state, memory, processed, mask)
            group = m_out.view(1, self.r, self.n_mels)
            mels.append(group)
            attns.append(attn[0])
            prev = group[:, -1]
            if torch.sigmoid(s_out).max() > 0.5:
                break
        mel = torch.cat(mels, 1).transpose(1, 2)
        return mel + self.postnet(mel), torch.stack(attns)


__all__ = ["Tacotron2", "Tacotron2Encoder"]
