# Data

## Layout

Every dataset is a folder in the **LJSpeech layout**:

```
<data_dir>/
  metadata.csv       <id>|<text>            or   <id>|<raw text>|<normalised text>
  wavs/<id>.wav      any sample rate (resampled on load), mono or stereo (averaged)
  speakers.csv       optional: <id>|<speaker>   -> multi-speaker models
  cache/             written by the loader: log-mel, F0, energy per clip
  durations/         optional: teacher durations (tts-eval --export-durations)
```

or a `manifest.jsonl` with one `{"audio": "...", "text": "...", "speaker": "..."}` per line.
Train/validation is a deterministic 95/5 split (`data.split_items`), the same in `tts-train`
and `tts-eval`.

## Built-in datasets (`tts-data`)

| name | what | size | why |
|---|---|---|---|
| `synthetic` | toy spoken language: each letter is a harmonic tone with its own pitch and spectral shape, spaces are silences, 25 dB noise, 8 kHz | 300 clips, 2 MB | known, learnable text -> audio map; tests and smoke runs |
| `fsdd` | Free Spoken Digit Dataset, "zero".."nine", 6 speakers, 8 kHz | 3000 clips, 20 MB (500 with `--speaker jackson`) | real low-quality speech with a 10-word vocabulary |
| `yesno` | 60 eight-word yes/no sequences (Hebrew "ken"/"lo", transcribed as yes/no), 8 kHz | 4.5 MB | the smallest real dataset there is |
| `ljspeech` | 13,100 clips, 24 h, one English speaker, 22.05 kHz | 2.6 GB | the standard single-speaker benchmark; Kaggle/Colab |

```bash
tts-data synthetic --out data/synthetic --n 300          # add --n-speakers 3 for a toy multi-speaker set
tts-data fsdd --out data/fsdd                             # --speaker all keeps the six speakers
tts-data yesno --out data/yesno
tts-data ljspeech --out data/ljspeech                     # 2.6 GB download + extraction
```

## Audio settings must match the data

| data | flags |
|---|---|
| 8 kHz sets (default) | `--sample-rate 8000 --n-fft 512 --hop-length 128 --win-length 512 --n-mels 64 --fmax 4000` and `--upsample-rates 8,4,4` |
| LJSpeech / 22.05 kHz | `--sample-rate 22050 --n-fft 1024 --hop-length 256 --win-length 1024 --n-mels 80 --fmax 8000` and `--upsample-rates 8,8,4` |
| 16 kHz (VCTK, LibriTTS) | `--sample-rate 16000 --n-fft 1024 --hop-length 200 --win-length 800 --n-mels 80 --fmax 8000` and `--upsample-rates 10,5,4` |

The product of `upsample_rates` must equal `hop_length` (HiFi-GAN and VITS upsample a frame to
`hop_length` samples). The cache file name includes the audio settings, so changing them
recomputes the features.

## Your own data

Put `metadata.csv` and `wavs/` in a folder; keep clips under ~ 10 s (`--max-frames` skips
longer ones) and trim leading/trailing silence (long silences give the aligner and the
attention nothing to hold on to). Numbers and abbreviations are spelled out by `text.normalize`
only for small integers; expand the rest in the metadata.
