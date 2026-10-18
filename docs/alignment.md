# Alignment: how the model knows which frame belongs to which letter

A sentence of N symbols has to become T mel frames (T is 20-50 x N). Every TTS model solves
this somewhere; this page is the map of the three solutions in `src/tts/align.py` and the
models that use them.

## 1. Attention (Tacotron 2, Transformer-TTS)

The decoder learns a soft alignment `alpha_t` over the N symbols at every step. Nothing
guarantees it is monotonic; the location-sensitive attention of Tacotron 2 and the
guided-attention loss (`layers.guided_attention_loss`) push it there. The matrix
`attention (T/r, N)` is saved as `alignment_<step>.png` during validation, and
`metrics.attention_diagonality` measures how much of its mass lies within 10 % of the diagonal.

## 2. Monotonic Alignment Search, MAS (`align.mas`)

Given a score `logp[n, t]` saying how well symbol n explains frame t, MAS finds the best path
from (0, 0) to (N-1, T-1) that moves right every frame and never goes back or skips a symbol:

```
q[n, t] = logp[n, t] + max(q[n, t-1], q[n-1, t-1])        (dynamic programming, O(N T))
backtrack from (N-1, T-1): stay on n or step to n-1
```

The result is a hard 0/1 matrix `(N, T)`; its row sums are the durations. It is plain Python
loops, fast enough for our sizes (a 20 x 200 problem takes ~ 5 ms).

Two models use it:

- **VITS**: `logp` is the log-likelihood of the flow-transformed latent `z_p[t]` under the
  Gaussian `N(mu_p[n], logs_p[n])` of each text position.
- **FastSpeech 2's aligner** (below).

## 3. The built-in aligner (`align.Aligner`, used by FastSpeech 2)

Following "One TTS Alignment To Rule Them All" (Badlani et al. 2022):

```
text embeddings -> 2 convs -> h_text (B, D, N)
mel             -> 3 convs -> h_mel  (B, D, T)
logits[n, t] = -||h_mel[t] - h_text[n]||^2 / sqrt(D)
log_probs    = log_softmax over n                   (soft alignment, every frame picks a symbol)
```

It is trained with the **forward-sum loss**: the log of the total probability of *all*
monotonic paths, computed with the CTC forward algorithm (`F.ctc_loss` with the symbols
1..N as targets and a blank that can never be emitted). Then **MAS** on `log_probs` gives
the hard alignment, and a small **binarisation loss** `-sum(hard * log_probs)` pulls the
soft alignment towards it. The durations are `hard.sum(dim=2)`.

This is why `tts-train --model fastspeech2` works on any dataset in the LJSpeech layout with no
alignment tool: the aligner trains together with the model (its loss shows as `forward_sum`
and `bin` in the logs) and the alignment plot of the validation set is the MAS path.

## 4. Teacher durations (`align.durations_from_attention`, `tts-eval --export-durations`)

The FastSpeech 1 recipe: train Tacotron 2, run it teacher-forced on every clip, and count the
frames each symbol wins in the attention matrix (MAS on the log-attention makes it monotonic
and gives every symbol at least one frame). `tts-eval --checkpoint tacotron2/best.pt
--data-dir data/x --export-durations` writes `data/x/durations/<id>.pt`, and
`tts-train --model fastspeech2 --duration-source teacher` reads them.

## Reading the plots

A healthy alignment is a thin diagonal from bottom-left to top-right. Horizontal streaks are
frames that cannot decide (early training); a vertical stripe is a symbol that attracts too
many frames (often a long silence); a line that stops before the top means the model ran out
of frames before reading the whole sentence.
