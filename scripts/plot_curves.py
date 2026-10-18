"""Training curves of several runs on one figure, from the curves.csv that tts-train writes.

    python scripts/plot_curves.py docs/images/synthetic_curves.png checkpoints/synthetic_*

Left: training loss (log scale, every model has a different mix of terms, so only the trend
is comparable). Right: validation teacher-forced mel L1 (`val_mel`), the one term every model
shares, so the models can be compared.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
runs = [Path(p) for p in sys.argv[2:]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for run in runs:
    rows = list(csv.DictReader((run / "curves.csv").open()))
    train = [(int(r["step"]), float(r["loss"])) for r in rows if r.get("loss")]
    val = [(int(r["step"]), float(r["val_mel"])) for r in rows if r.get("val_mel")]
    label = run.name.split("_", 1)[-1]
    if train:
        axes[0].plot(*zip(*train), label=label)
    if val:
        axes[1].plot(*zip(*val), "o-", label=label)
axes[0].set_yscale("log")
for ax, title in zip(axes, ("train loss (log)", "val mel L1 (teacher-forced)")):
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
