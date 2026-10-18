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

See the README's results table. They are short CPU runs on 8 kHz data meant to show the
pipeline learns; the numbers are not comparable with LJSpeech benchmarks.
