"""Markdown results tables from results/summary.json and results/ceilings.json.

    python scripts/results_table.py synthetic      # or fsdd

summary.json is written by the evaluation of every checkpoint (see docs/evaluation.md); the
table has one row per (model, vocoder). Values are means over the evaluated validation sentences.
"""

import json
import sys
from pathlib import Path

which = sys.argv[1]
rows = json.loads(Path("results/summary.json").read_text())
ceil = json.loads(Path("results/ceilings.json").read_text())[which]

print("| model | vocoder | params | steps | val mel L1 (first → last) | MCD (dB) | log-F0 RMSE (st) | voicing err. | duration err. | alignment diag. |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if not r["run"].startswith(which + "_") or "mcd_db" not in r:
        continue
    model = r["run"].split("_", 1)[1].replace("_", "-")
    voc = r.get("vocoder", "")
    voc = "(end-to-end)" if voc.startswith("none") else voc.replace("griffinlim", "Griffin-Lim").replace("hifigan", "HiFi-GAN")
    print(f"| `{model}` | {voc} | {r['params'] / 1e6:.1f} M | {r['steps']} | {r['val_mel_first']:.2f} → {r['val_mel']:.2f} | "
          f"{r['mcd_db']:.1f} | {r['f0_rmse_semitones']:.1f} | {r['voicing_error']:.2f} | {r['duration_error']:.2f} | "
          f"{r['attention_diagonality']:.2f} |")
voc = [r for r in rows if r["run"] == f"{which}_hifigan" and "mcd_db" not in r]
if voc:
    v = voc[0]
    print(f"| `hifigan` (vocoder alone) | | | {v['steps']} | {v['val_mel_first']:.2f} → {v['val_mel']:.2f} | | | | | |")
print()
print("Reference points for MCD on this data (same metric, same 8 kHz mel settings):")
print(f"- Griffin-Lim copy-synthesis of the *true* mel vs. the recording: **{ceil['copy_synthesis_mcd']:.1f} dB** "
      "(the floor a mel model + Griffin-Lim can reach).")
if ceil.get("same_text_other_recording_mcd"):
    print(f"- Another recording of the same text by the same speaker vs. the recording: **{ceil['same_text_other_recording_mcd']:.1f} dB**, "
          f"log-F0 RMSE {ceil['same_text_other_recording_f0_rmse']:.1f} semitones (natural variation).")
