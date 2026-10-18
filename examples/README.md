# Examples

Short scripts that use the `tts` package from Python, in the order one would use it. All run
on a CPU in minutes; run them from the repository root after `pip install -e ".[dev]"`.

| script | what it shows | needs |
|---|---|---|
| [01_train_on_toy_data.py](01_train_on_toy_data.py) | `Config` and `Trainer`: FastSpeech 2 with the built-in aligner on 120 toy sentences, a few minutes | nothing (writes `data/examples/synthetic`) |
| [02_synthesize.py](02_synthesize.py) | `Synthesizer`: text to waveform (Griffin-Lim), slower speech with `duration_scale`, the mel and the alignment drawn | 01 |
| [03_evaluate.py](03_evaluate.py) | `evaluate`: MCD, F0, voicing, length and alignment metrics on validation sentences, plus the copy-synthesis floor | 01 |
| [04_mel_and_griffin_lim.py](04_mel_and_griffin_lim.py) | the audio front end: `MelSpectrogram`, `estimate_f0`, `griffin_lim` with 4 / 16 / 64 iterations and their MCD | 01 or any wav |
| [05_model_zoo.py](05_model_zoo.py) | all models (Tacotron 2, Transformer-TTS, FastSpeech 2, VITS, HiFi-GAN) on a random batch: one loss, one backward, one inference | nothing |
| [06_monotonic_alignment_search.py](06_monotonic_alignment_search.py) | MAS on a toy similarity matrix, printed as text, with the durations it yields | nothing |

```bash
python examples/01_train_on_toy_data.py
python examples/02_synthesize.py
python examples/03_evaluate.py
python examples/04_mel_and_griffin_lim.py
python examples/05_model_zoo.py
python examples/06_monotonic_alignment_search.py
```

The same steps as CLI commands are in the README's quick start (`tts-data`, `tts-train`,
`tts-synthesize`, `tts-eval`); the Kaggle/Colab notebook does them on real data with a GPU.
