"""One figure with the first and the last alignment plot of several runs.

    python scripts/alignment_grid.py docs/images/synthetic_alignments.png checkpoints/synthetic_*

Each run folder holds alignment_<step>.png files written by tts-train; the first and the last
are shown side by side, so the reader sees the alignment appear.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
runs = [Path(p) for p in sys.argv[2:] if sorted(Path(p).glob("alignment_*.png"))]
fig, axes = plt.subplots(len(runs), 2, figsize=(11, 3.2 * len(runs)))
for row, run in zip(axes, runs):
    plots = sorted(run.glob("alignment_*.png"))
    for ax, plot in zip(row, (plots[0], plots[-1])):
        ax.imshow(mpimg.imread(plot))
        ax.set_axis_off()
    row[0].set_title(f"{run.name}: first evaluation", fontsize=10)
    row[1].set_title(f"{run.name}: last evaluation", fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=100)
print("wrote", out)
