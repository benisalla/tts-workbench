"""Load the checkpoint of example 01 and turn text into a waveform, a mel and an alignment.

    python examples/02_synthesize.py          # writes examples/output/*.wav and synthesis.png

`Synthesizer` rebuilds model + tokenizer from the checkpoint; calling it gives the waveform
(Griffin-Lim here, or a HiFi-GAN checkpoint through `vocoder="hifigan"`), `.mel()` gives
the log-mel and the alignment. FastSpeech 2 and VITS take `duration_scale` to speak slower.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tts.audio import load_wav, save_wav
from tts.data import read_manifest, split_items
from tts.synthesize import Synthesizer

syn = Synthesizer("checkpoints/examples-fastspeech2/best.pt")  # run examples/01_train_on_toy_data.py first
_, val = split_items(read_manifest("data/examples/synthetic"), seed=syn.cfg.train.seed)
text = val[0].text  # a validation sentence of the toy language, never seen in training
print(f"{syn.cfg.model.model}, {syn.sample_rate} Hz, text: {text!r}")

out = Path("examples/output")
out.mkdir(parents=True, exist_ok=True)
for name, kw in (("normal", {}), ("slow", {"duration_scale": 1.5})):
    wav, alignment = syn(text, **kw)
    save_wav(str(out / f"{val[0].id}_{name}.wav"), wav, syn.sample_rate)
    print(f"  {name:6s}: {wav.numel() / syn.sample_rate:.2f} s, alignment {tuple(alignment.shape)} (frames x symbols)")
save_wav(str(out / f"{val[0].id}_recording.wav"), load_wav(str(val[0].audio), syn.sample_rate), syn.sample_rate)

mel, alignment = syn.mel(text)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].imshow(mel.numpy(), origin="lower", aspect="auto", cmap="magma")
axes[0].set_title(f"log-mel {tuple(mel.shape)}")
axes[0].set_xlabel("frame")
axes[1].imshow(alignment.numpy().T, origin="lower", aspect="auto", cmap="viridis")
axes[1].set_title("alignment: which symbol each frame reads")
axes[1].set_xlabel("frame")
axes[1].set_yticks(range(len(text)), list(text))
fig.tight_layout()
fig.savefig(out / "synthesis.png", dpi=110)
print(f"wrote {out}/*.wav and {out}/synthesis.png")
