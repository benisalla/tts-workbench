"""Monotonic alignment search (MAS) on a toy similarity matrix, printed as text.

    python examples/06_monotonic_alignment_search.py

MAS is how Glow-TTS / VITS and the built-in aligner turn a soft text-to-frame similarity into
a hard, monotonic alignment: every frame reads exactly one symbol, the symbol index never
goes back, and every symbol gets at least one frame. The durations are the row sums.
"""

import numpy as np
import torch

from tts.align import durations_from_alignment, mas
from tts.metrics import attention_diagonality

text = "ba ko"
N, T = len(text), 24  # 5 symbols, 24 frames

# A soft similarity: each frame likes the symbol it "should" read, with noise and a decoy.
rng = np.random.default_rng(0)
true_boundaries = [0, 6, 9, 11, 17, 24]  # frames of b, a, space, k, o
logp = np.full((N, T), -3.0)
for n in range(N):
    logp[n, true_boundaries[n]:true_boundaries[n + 1]] = 0.0
logp += rng.normal(0, 0.8, size=logp.shape)
logp[1, 20:23] += 4.0  # a decoy: late frames that also look like "a"; MAS cannot go back to it

hard = mas(logp, N, T)
print("soft log-similarity (rows = symbols, columns = frames), argmax per frame:")
print("  " + "".join(text[i] for i in logp.argmax(0)))
print("MAS path (monotonic, every symbol >= 1 frame):")
print("  " + "".join(text[i] for i in hard.argmax(0)))
for n, ch in enumerate(text):
    print(f"  {ch!r}: " + "".join("#" if hard[n, t] else "." for t in range(T)))

durations = durations_from_alignment(torch.from_numpy(hard)[None].float())[0]
print(f"durations per symbol: {durations.tolist()}  (sum {int(durations.sum())} = {T} frames)")
print(f"attention diagonality of the path: {attention_diagonality(torch.from_numpy(hard).float().T):.2f}")
