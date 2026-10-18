# Evaluation

```bash
tts-eval --checkpoint checkpoints/fs2/best.pt --data-dir data/fsdd --n 20 --out results/fs2
tts-eval --checkpoint ... --vocoder hifigan --vocoder-checkpoint checkpoints/hifigan/best.pt
tts-eval --checkpoint ... --whisper --utmos          # intelligibility WER and predicted MOS (downloads models)
```

`tts-eval` synthesizes `--n` validation sentences, compares each with its recording and prints
the mean of every metric (`metrics.json` has the per-sentence values; `--out` also keeps the wavs).

## Objective metrics (`src/tts/metrics.py`)

| metric | what it measures | how to read it |
|---|---|---|
| `mcd_db` | mel-cepstral distortion: Euclidean distance between the mel-cepstra (DCT of the log-mel, coefficients 1-12) of synthesis and recording, along a DTW path, in dB | lower is better; ~ 5 dB is a good neural TTS on LJSpeech, > 8 dB is clearly different, > 15 dB is noise |
| `f0_rmse_semitones` | RMSE of the log-F0 over frames voiced in both clips, along the same DTW path | < 1 semitone: same melody; 3+: a different intonation |
| `voicing_error` | share of frames where one clip is voiced and the other not | 0.1 is typical, 0.5 is random |
| `duration_error` | relative length difference of the two clips | 0.1 = 10 % faster/slower |
| `attention_diagonality` | share of attention mass within 10 % of the diagonal (AR models; the MAS path for FastSpeech 2 / VITS, which is always ~1) | > 0.8 is a healthy alignment |
| `wer` / `wer_recording` | word error rate of a Whisper transcript against the text, for the synthesis and for the recording (the ceiling) | needs `pip install "tts-workbench[metrics]"` and real words (not the toy language) |
| `utmos` / `utmos_recording` | predicted MOS (1-5) from UTMOS22-strong, a model trained to predict listening-test scores | > 3.5 natural; the recording's score is the ceiling; 8 kHz audio is penalised by design |

The validation loss printed during training (`val_mel`) is the *teacher-forced* mel error: it
says how well the model predicts the next frame when given the truth so far, which is not
the same as synthesizing a whole sentence. MCD is measured on free-running synthesis.

## Why DTW

The synthesis is never exactly as long as the recording, so frames cannot be compared one to
one. `metrics.dtw_path` aligns the two cepstral sequences with dynamic time warping (monotonic
path of minimal total distance) and MCD / F0 are averaged along that path.

## Human MOS

Predicted MOS is a proxy. For a real **mean opinion score**:

1. Pick 10-20 validation sentences, synthesize them with each system (and keep the recordings as
   a hidden reference).
2. Shuffle, hide the system names, and ask >= 10 listeners to rate naturalness from 1 (bad) to
   5 (excellent) on headphones; a 1-5 *intelligibility* question is a cheap add-on.
3. Report mean +/- 95 % confidence interval per system, and the recordings' score.

`tts-synthesize --text-file sentences.txt --out <system>/` produces the files; a spreadsheet
does the rest.

## Results on the built-in datasets

Short CPU runs on 8 kHz data meant to show the pipeline learns; the numbers are not
comparable with LJSpeech benchmarks. Every row is `tts-eval --checkpoint
checkpoints/<run>/best.pt --data-dir data/<set> --n 15 [--vocoder hifigan --vocoder-checkpoint
checkpoints/<set>_hifigan/best.pt]`; `results/summary.json` collects them and
`python scripts/results_table.py synthetic|fsdd` prints the tables. Training commands:
[training.md](training.md).

### Synthetic toy language (285 train / 15 val sentences)

| model | vocoder | params | steps | val mel L1 (first → last) | MCD (dB) | log-F0 RMSE (st) | voicing err. | duration err. | alignment diag. |
|---|---|---|---|---|---|---|---|---|---|
| `tacotron2` | Griffin-Lim | 0.9 M | 2880 | 2.68 → 0.12 | 18.8 | 6.1 | 0.14 | 0.04 | 0.17 |
| `tacotron2` | HiFi-GAN | 0.9 M | 2880 | 2.68 → 0.12 | 18.6 | 16.4 | 0.14 | 0.03 | 0.17 |
| `transformer-tts` | Griffin-Lim | 1.4 M | 720 | 0.88 → 0.19 | 13.0 | 4.6 | 0.06 | 0.04 | 0.91 |
| `transformer-tts` | HiFi-GAN | 1.4 M | 720 | 0.88 → 0.19 | 15.4 | 19.9 | 0.13 | 0.03 | 0.91 |
| `fastspeech2` | Griffin-Lim | 2.4 M | 720 | 0.44 → 0.27 | 12.3 | 3.0 | 0.04 | 0.05 | 0.97 |
| `fastspeech2` | HiFi-GAN | 2.4 M | 720 | 0.44 → 0.27 | 14.6 | 19.5 | 0.13 | 0.03 | 0.97 |
| `vits` | (end-to-end) | 3.1 M | 720 | 0.56 → 0.38 | 12.9 | 9.7 | 0.17 | 0.02 | 0.92 |
| `hifigan` (vocoder alone) | | | 540 | 0.52 → 0.44 | | | | | |

Reference points for MCD on this data (same metric, same 8 kHz mel settings):
- Griffin-Lim copy-synthesis of the *true* mel vs. the recording: **10.5 dB** (the floor a mel model + Griffin-Lim can reach).

### FSDD spoken digits (one speaker, 475 train / 25 val clips)

| model | vocoder | params | steps | val mel L1 (first → last) | MCD (dB) | log-F0 RMSE (st) | voicing err. | duration err. | alignment diag. |
|---|---|---|---|---|---|---|---|---|---|
| `fastspeech2` | Griffin-Lim | 2.4 M | 4200 | 0.72 → 0.51 | 31.9 | 9.1 | 0.10 | 0.21 | 0.40 |
| `fastspeech2` | HiFi-GAN | 2.4 M | 4200 | 0.72 → 0.51 | 33.5 | 13.3 | 0.15 | 0.19 | 0.40 |
| `tacotron2` | Griffin-Lim | 0.9 M | 7200 | 1.11 → 0.34 | 29.0 | 7.7 | 0.13 | 0.10 | 0.29 |
| `tacotron2` | HiFi-GAN | 0.9 M | 7200 | 1.11 → 0.34 | 34.0 | 13.4 | 0.14 | 0.09 | 0.29 |
| `hifigan` (vocoder alone) | | | 1200 | 1.01 → 0.83 | | | | | |

Reference points for MCD on this data (same metric, same 8 kHz mel settings):
- Griffin-Lim copy-synthesis of the *true* mel vs. the recording: **18.8 dB** (the floor a mel model + Griffin-Lim can reach).
- Another recording of the same text by the same speaker vs. the recording: **26.4 dB**, log-F0 RMSE 5.4 semitones (natural variation).

How to read them (details and figures in the README):

- Transformer-TTS, FastSpeech 2 and VITS reach 12–13 dB on the toy language, 2–3 dB above the
  copy-synthesis floor, with near-diagonal alignments. Tacotron 2 has the lowest teacher-forced
  error but no diagonal attention, so its free-running synthesis is 6 dB worse: `val_mel` alone
  never tells you whether an attention model can synthesize.
- On FSDD both models land just above the natural variation between two takes of the same
  digit (26.4 dB, 5.4 semitones). That is the ceiling one can see with 15 one-word sentences.
- The HiFi-GAN vocoders (540 / 1 200 steps) are not trained enough: they add about 2 dB MCD and
  make the F0 error explode, which is what an unconverged GAN vocoder sounds like (buzz, no
  pitch). Griffin-Lim is the reference vocoder at this scale.
- Alignment diagonality is ~1 for FastSpeech 2 / VITS by construction when the aligner has
  settled (toy language) and drops to 0.4 when the MAS path piles frames on one symbol (FSDD).

The figures `images/synthetic_curves.png`, `images/fsdd_curves.png` (`scripts/plot_curves.py`)
and `images/synthetic_alignments.png`, `images/fsdd_alignments.png` (`scripts/alignment_grid.py`)
are built from the `curves.csv` and `alignment_<step>.png` files that `tts-train` writes.
