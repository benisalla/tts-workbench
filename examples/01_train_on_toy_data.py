"""Train a small FastSpeech 2 on the synthetic toy language, on a CPU, in a few minutes.

    python examples/01_train_on_toy_data.py

What happens: `write_synthetic` makes 120 sentences of a toy language (each letter a tone
pattern, 8 kHz), `Trainer` runs the normal loop with the built-in aligner providing the
durations, and the checkpoint lands in checkpoints/examples-fastspeech2/ (best.pt, last.pt,
config.json, curves.csv and an alignment_<step>.png per evaluation).
Examples 02 and 03 load that checkpoint.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path

import torch

from tts.config import Config
from tts.datasets import write_synthetic
from tts.train import Trainer

torch.set_num_threads(2)

DATA = Path("data/examples/synthetic")
OUT = "checkpoints/examples-fastspeech2"

if not (DATA / "metadata.csv").exists():
    write_synthetic(DATA, n=120, seed=0)
    print(f"wrote {DATA}")

# One Config object = audio + text + model + train sections; every field is also a tts-train flag.
cfg = Config()
cfg.model.model = "fastspeech2"  # or tacotron2, transformer_tts, vits, hifigan
cfg.model.hidden = 64  # default 128; smaller is enough for 120 sentences
cfg.train.data_dir = str(DATA)
cfg.train.output_dir = OUT
cfg.train.epochs = 40  # 120 sentences / batch 16 = 8 steps per epoch
cfg.train.batch_size = 16
cfg.train.warmup_steps = 50
cfg.train.bin_warmup_steps = 100  # fastspeech2: when the aligner's soft attention starts being pulled to its hard path
cfg.train.eval_every = 5  # epochs; each evaluation writes an alignment plot
cfg.train.log_every = 20
cfg.train.device = "cpu"

summary = Trainer(cfg).fit()
print(f"\nfinal: {summary['step']} steps, best teacher-forced val mel L1 {summary['best_val_mel']:.3f}, "
      f"{summary['seconds'] / 60:.1f} min")
print(f"checkpoint: {OUT}/best.pt   alignment plots: {OUT}/alignment_*.png")
