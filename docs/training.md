# Training

```bash
tts-train --model tacotron2 --data-dir data/synthetic --output-dir checkpoints/tacotron2 --epochs 40
tts-train --model fastspeech2 --data-dir data/fsdd --output-dir checkpoints/fs2 --epochs 60 --batch-size 16
tts-train --model hifigan --data-dir data/fsdd --output-dir checkpoints/hifigan --epochs 30
tts-train --model vits --data-dir data/fsdd --output-dir checkpoints/vits --epochs 60
tts-train --config checkpoints/fs2/config.json --resume --epochs 120     # continue a run
```

Every field of `config.py` is a flag (`--hidden 256`, `--lr 5e-4`, `--amp true`, ...). The run
folder gets `config.json`, `last.pt`, `best.pt` (lowest validation mel loss), `curves.csv/json`
and `alignment_<step>.png`.

## What happens in a step (`src/tts/train.py`)

- **Mel models** (tacotron2, transformer_tts, fastspeech2): `model(batch)` returns the loss
  terms; AdamW with betas (0.9, 0.98), linear warm-up over `--warmup-steps` then
  `1/sqrt(step)` decay, gradient clipping at `--grad-clip`, optional autocast (`--amp`).
- **GAN models** (hifigan, vits): a random `segment_frames` slice of the mel / latent and the
  matching waveform slice; first the discriminators (real -> 1, fake -> 0), then the generator
  with `LS-GAN + 2 x feature matching + 45 x mel L1` (+ KL and duration for VITS). Two AdamW
  optimisers with betas (0.8, 0.99). No autocast (GAN losses are fragile in half precision).

## What to watch

| model | signs of learning | typical problem |
|---|---|---|
| tacotron2 / transformer_tts | `mel_after` falling, `stop` -> 0, alignment plot turning into a diagonal within a few hundred steps | no diagonal: raise `--guided-attention`, lower the learning rate, trim silences |
| fastspeech2 | `forward_sum` falling, `bin` -> ~0.1, `duration` < 0.1, `mel_after` falling | `pitch` stuck: data with little voicing (toy data is fine, whispered speech is not) |
| hifigan | `mel` falling below ~ 0.5 while `gen` and `disc` hover around 1-3 | `disc` -> 0 and `gen` exploding: the discriminator won; lower `--lr` |
| vits | `kl` 0.5-2, `duration` < 0.2, `mel` falling | same as hifigan |

Validation uses the same losses; `best.pt` follows `val_mel` (the post-net mel loss, or the mel
L1 of the GAN models). Note that BatchNorm in the convolution stacks uses running statistics
in eval mode, so the first validation numbers of Tacotron 2 can be much worse than the training
loss until the statistics settle (a few epochs).

## Kaggle / Colab

`notebooks/train_colab_kaggle.ipynb` is generated from `src/` by `notebooks/build_notebook.py`
and contains the same code. Pick `MODEL`, `DOWNLOAD_DATASET`, batch size and the audio settings
in its section 2.2; `TIME_BUDGET_HOURS` makes the trainer save and stop before the session
limit and `RESUME = True` continues from `last.pt`. Real training (LJSpeech, hundreds of
epochs) belongs there; this machine is for the tests and the short runs of the README.

Recommended LJSpeech settings for a first real model: `--hidden 256 --layers 4 --batch-size 32
--lr 1e-3 --amp true` with the 22.05 kHz audio flags of [data.md](data.md), 100-200 epochs for
Tacotron 2 / FastSpeech 2, plus a HiFi-GAN run of similar length.

## Reproducing the README numbers

```bash
tts-data synthetic --out data/synthetic --n 300
tts-train --model tacotron2       --data-dir data/synthetic --output-dir checkpoints/synthetic_tacotron2 --epochs 40 --eval-every 5
tts-train --model transformer_tts --data-dir data/synthetic --output-dir checkpoints/synthetic_transformer_tts --epochs 40 --eval-every 5
tts-train --model fastspeech2     --data-dir data/synthetic --output-dir checkpoints/synthetic_fastspeech2 --epochs 40 --eval-every 5
tts-train --model hifigan         --data-dir data/synthetic --output-dir checkpoints/synthetic_hifigan --epochs 30 --eval-every 5
tts-train --model vits            --data-dir data/synthetic --output-dir checkpoints/synthetic_vits --epochs 40 --eval-every 5
tts-eval --checkpoint checkpoints/synthetic_<model>/best.pt --data-dir data/synthetic --n 15
```
