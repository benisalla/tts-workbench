"""Score the checkpoint of example 01 against the recordings, and compute the floor it cannot beat.

    python examples/03_evaluate.py

`evaluate` synthesizes validation sentences and compares them with the recordings along a
DTW path: MCD (mel-cepstral distortion, dB), log-F0 RMSE (semitones), voicing and length
errors, and how diagonal the alignment is. Copy-synthesis (the *true* mel through the same
Griffin-Lim) gives the floor a mel-predicting model can reach with this vocoder.
"""

import torch

from tts.audio import MelSpectrogram, griffin_lim, load_wav
from tts.data import read_manifest, split_items
from tts.metrics import evaluate, mcd
from tts.models import load_checkpoint

CKPT = "checkpoints/examples-fastspeech2/best.pt"  # run examples/01_train_on_toy_data.py first
DATA = "data/examples/synthetic"

result = evaluate(CKPT, DATA, vocoder="griffinlim", n=5, out_dir="examples/output/eval")
s = result["summary"]
print(f"{s['model']} + {s['vocoder']} on {s['n']} validation sentences:")
print(f"  MCD {s['mcd_db']:.1f} dB   log-F0 RMSE {s['f0_rmse_semitones']:.1f} st   voicing error {s['voicing_error']:.2f}"
      f"   length error {s['duration_error']:.2f}   alignment diagonality {s['attention_diagonality']:.2f}")
print("  per sentence:")
for item in result["items"]:
    print(f"    {item['id']}: MCD {item['mcd_db']:.1f} dB  {item['text']!r}")

# The floor: analyse a recording to a mel and resynthesize it with the same Griffin-Lim.
_, cfg, _, _ = load_checkpoint(CKPT)
mel_fn = MelSpectrogram(cfg.audio)
_, val = split_items(read_manifest(DATA), seed=cfg.train.seed)
floors = []
for item in val[:5]:
    ref = torch.from_numpy(load_wav(str(item.audio), cfg.audio.sample_rate))
    copy = griffin_lim(mel_fn(ref[None])[0], cfg.audio, n_iter=32)
    floors.append(mcd(mel_fn(copy[None])[0], mel_fn(ref[None])[0])[0])
print(f"\ncopy-synthesis floor (true mel -> Griffin-Lim -> MCD vs recording): {sum(floors) / len(floors):.1f} dB")
print("a model is as good as this vocoder allows when its MCD approaches that number")
print("wrote examples/output/eval/*_syn.wav and metrics.json")
