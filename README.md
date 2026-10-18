# tts-workbench

**Every main text-to-speech approach in one small, readable PyTorch code base, built to
understand how each one works: Tacotron 2, Transformer-TTS, FastSpeech 2, VITS and a
HiFi-GAN vocoder, with a built-in aligner, MCD / F0 / WER / MOS evaluation and a Kaggle/Colab
notebook.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](docs/testing.md)

The models are deliberately small (about one million parameters each by default) so that a
run on a laptop CPU or a free GPU shows what the method does, and so that every line can be
read. Each model file starts with a docstring that explains the architecture and the
equations; [docs/architecture.md](docs/architecture.md) compares them.

```mermaid
flowchart LR
    T["text"] --> N["normalise + tokenise"]
    N --> T2["Tacotron 2<br/>AR, location attention"]
    N --> TT["Transformer-TTS<br/>AR, self-attention"]
    N --> FS["FastSpeech 2<br/>NAR, durations from<br/>the built-in aligner"]
    N --> VI["VITS<br/>VAE + flow + MAS + GAN"]
    T2 --> M["log-mel"]
    TT --> M
    FS --> M
    M --> GL["Griffin-Lim"] --> W["waveform"]
    M --> HG["HiFi-GAN"] --> W
    VI --> W
```

| approach | file | what you learn from it |
|---|---|---|
| **Tacotron 2** | `src/tts/tacotron2.py` | autoregressive decoding, teacher forcing, location-sensitive attention, prenet dropout, stop token, postnet |
| **Transformer-TTS** | `src/tts/transformer_tts.py` | the same with self-attention, causal masks, scaled positional encodings, why guided attention is needed |
| **FastSpeech 2** | `src/tts/fastspeech2.py` | non-autoregressive synthesis, length regulation, duration/pitch/energy predictors, variance information |
| **built-in aligner + MAS** | `src/tts/align.py` | monotonic alignment search, the forward-sum (CTC) loss, teacher durations from attention |
| **VITS** | `src/tts/vits.py` | conditional VAE, posterior encoder, normalising flow (coupling layers), MAS on a latent, end-to-end GAN training |
| **HiFi-GAN** | `src/tts/vocoders.py`, `src/tts/losses.py` | neural vocoding, multi-receptive-field fusion, multi-period and multi-scale discriminators, feature matching |
| **Griffin-Lim, mel, F0** | `src/tts/audio.py` | STFT, mel filterbanks, phase reconstruction, autocorrelation pitch tracking |
| **metrics** | `src/tts/metrics.py` | MCD with DTW, log-F0 RMSE, attention diagonality, WER through Whisper, predicted MOS (UTMOS) |

Every model has a multi-speaker option (a learned speaker embedding), every mel model works
with either vocoder, and all of them share the same data layout, trainer, checkpoint format
and evaluation.

> **No pretrained weights are published.** Train on the built-in datasets or your own
> ([docs/data.md](docs/data.md)). Real training (LJSpeech, hundreds of epochs) is meant for the
> Kaggle/Colab notebook; the runs below are short CPU checks.

## Quick start

```bash
git clone https://github.com/benisalla/tts-workbench.git && cd tts-workbench
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"      # or: pip install -e ".[dev]"

tts-data synthetic --out data/synthetic                              # toy spoken language, 8 kHz, 2 MB
tts-train --model fastspeech2 --data-dir data/synthetic --output-dir checkpoints/fs2 --epochs 40
tts-synthesize --checkpoint checkpoints/fs2/best.pt --text "ba ko mi tu" --out synthesized/
tts-eval --checkpoint checkpoints/fs2/best.pt --data-dir data/synthetic --n 15
```

Swap `--model` for `tacotron2`, `transformer_tts`, `vits` or `hifigan`; add
`--vocoder hifigan --vocoder-checkpoint checkpoints/hifigan/best.pt` to `tts-synthesize` once a
vocoder is trained. Every field of `src/tts/config.py` is a flag (`--hidden 256 --layers 4 ...`).

Real speech to try on this scale: `tts-data fsdd` (3,000 spoken digits, 8 kHz, 20 MB) and
`tts-data yesno` (60 clips, 4.5 MB). For LJSpeech use the notebook.

## Kaggle / Colab notebook

[`notebooks/train_colab_kaggle.ipynb`](notebooks/train_colab_kaggle.ipynb) is generated from
`src/` by `notebooks/build_notebook.py`, so it contains exactly this code. Choose `MODEL`,
`DOWNLOAD_DATASET` (`ljspeech`, `fsdd`, `yesno`) and the audio settings in one cell; it trains
with a time budget, draws the curves and the alignment, plays the synthesized sentences and
scores them (MCD, F0, WER with Whisper, predicted MOS with UTMOS). Its checkpoints work with
`tts-synthesize` and `tts-eval` here.

## Results

RESULTS_PLACEHOLDER

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | every model and vocoder: shapes, equations, why each piece exists, trade-offs |
| [docs/alignment.md](docs/alignment.md) | attention vs. MAS vs. the built-in aligner vs. teacher durations |
| [docs/data.md](docs/data.md) | the LJSpeech layout, built-in datasets, audio settings per sample rate |
| [docs/training.md](docs/training.md) | the training loops, what to watch, Kaggle/Colab, reproducing the results |
| [docs/inference.md](docs/inference.md) | `tts-synthesize`, the Python API, what inference does per model |
| [docs/evaluation.md](docs/evaluation.md) | MCD, F0, WER, UTMOS, how to run a human MOS test |
| [docs/testing.md](docs/testing.md) | the test suite |

## Layout

```
src/tts/        config, text, audio, layers, align, tacotron2, transformer_tts, fastspeech2,
                vocoders, losses, vits, data, datasets, models, train, synthesize, metrics
tests/          pytest suite (micro models, toy data, ~ 25 s)
docs/           the pages above, alignment plots and sample audio
notebooks/      build_notebook.py -> train_colab_kaggle.ipynb
```

## References

- Shen et al., *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions* (Tacotron 2), 2018
- Li et al., *Neural Speech Synthesis with Transformer Network* (Transformer-TTS), 2019
- Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*, 2020
- Kim et al., *Glow-TTS* (monotonic alignment search), 2020; Badlani et al., *One TTS Alignment To Rule Them All*, 2022
- Kim et al., *Conditional Variational Autoencoder with Adversarial Learning for End-to-End TTS* (VITS), 2021
- Kong et al., *HiFi-GAN*, 2020; Griffin & Lim, *Signal estimation from modified short-time Fourier transform*, 1984
- Saeki et al., *UTMOS*, 2022

## License

MIT.
