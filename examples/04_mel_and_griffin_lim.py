"""Waveform -> log-mel -> waveform with Griffin-Lim, and what more iterations buy.

    python examples/04_mel_and_griffin_lim.py [some.wav]     # writes examples/output/griffin_lim_*.wav, mel.png

No checkpoint needed. `MelSpectrogram` is the analysis every model is trained on;
`griffin_lim` is the vocoder that needs no training; `estimate_f0` is the pitch tracker the
metrics use. The MCD between the resynthesis and the original is the "copy-synthesis floor".
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tts.audio import MelSpectrogram, estimate_f0, griffin_lim, load_wav, save_wav
from tts.config import AudioConfig
from tts.metrics import mcd

cfg = AudioConfig()  # 8 kHz, 512-point FFT, hop 128 (16 ms), 64 mel bins up to 4 kHz
wav_path = Path(sys.argv[1]) if len(sys.argv) > 1 else next(Path("data/examples/synthetic/wavs").glob("*.wav"))
wav = torch.from_numpy(load_wav(str(wav_path), cfg.sample_rate))
mel_fn = MelSpectrogram(cfg)
mel = mel_fn(wav[None])[0]  # (n_mels, frames)
f0 = estimate_f0(wav.numpy(), cfg)
print(f"{wav_path.name}: {wav.numel() / cfg.sample_rate:.2f} s -> mel {tuple(mel.shape)}, "
      f"F0 median {float(torch.tensor(f0[f0 > 0]).median()) if (f0 > 0).any() else 0:.0f} Hz "
      f"on {(f0 > 0).mean() * 100:.0f}% voiced frames")

out = Path("examples/output")
out.mkdir(parents=True, exist_ok=True)
print("Griffin-Lim iterations -> MCD vs the recording (lower is better):")
for n_iter in (4, 16, 64):
    copy = griffin_lim(mel, cfg, n_iter=n_iter)
    score, _ = mcd(mel_fn(copy[None])[0], mel)
    save_wav(str(out / f"griffin_lim_{n_iter}.wav"), copy, cfg.sample_rate)
    print(f"  {n_iter:3d} iterations: {score:.1f} dB")

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)
axes[0].plot(wav.numpy(), lw=0.5)
axes[0].set_title("waveform")
axes[1].imshow(mel.numpy(), origin="lower", aspect="auto", cmap="magma")
axes[1].set_title("log-mel (what the models predict)")
axes[2].plot(f0)
axes[2].set_title("F0 in Hz (0 = unvoiced), what the F0 metrics and FastSpeech 2's pitch predictor use")
axes[2].set_xlabel("frame (16 ms)")
fig.tight_layout()
fig.savefig(out / "mel.png", dpi=110)
print(f"wrote {out}/griffin_lim_*.wav and {out}/mel.png")
