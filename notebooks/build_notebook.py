"""Build notebooks/train_colab_kaggle.ipynb: a self-contained TTS training notebook.

The code of src/tts/*.py is copied into the notebook at build time (imports between those
modules are removed, as everything lives in one namespace), so the notebook is identical
to the package and needs no GitHub access. Regenerate after changing src/tts:

    python notebooks/build_notebook.py . notebooks/train_colab_kaggle.ipynb
"""

import ast
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])

MODULES = ["config", "text", "audio", "layers", "align", "tacotron2", "transformer_tts", "fastspeech2", "vocoders",
           "losses", "vits", "data", "datasets", "models", "train", "synthesize", "metrics"]


def module_source(module: str) -> str:
    """A module's code without `from tts...` imports, __all__, __main__ and the docstring."""
    text = (ROOT / "src" / "tts" / f"{module}.py").read_text()
    lines = text.splitlines()
    drop: set[int] = set()
    tree = ast.parse(text)
    for node in tree.body:
        is_local_import = isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tts")
        is_all = isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "__all__"
        is_main = isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
        is_docstring = node is tree.body[0] and isinstance(node, ast.Expr)
        if is_local_import or is_all or is_main or is_docstring:
            drop.update(range(node.lineno - 1, node.end_lineno))
    kept = [line for i, line in enumerate(lines) if i not in drop]
    src = "\n".join(kept).strip() + "\n"
    # local imports inside functions (lazy imports of sibling modules) are no-ops in one namespace
    src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("from tts."))
    return src


def module_doc(module: str) -> str:
    text = (ROOT / "src" / "tts" / f"{module}.py").read_text()
    return ast.get_docstring(ast.parse(text)) or ""


cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


def module_cells(module: str, title: str) -> None:
    md(f"### {title} (`src/tts/{module}.py`)\n\n{module_doc(module)}")
    code(module_source(module))


# ---------------------------------------------------------------------------
md("""
# tts-workbench: training notebook (Kaggle & Colab)

Train any of the text-to-speech models of the `tts-workbench` repo on a **free Kaggle or Colab GPU**:

| `MODEL` | family | text -> | needs a vocoder |
|---|---|---|---|
| `tacotron2` | autoregressive, location-sensitive attention | mel | yes |
| `transformer_tts` | autoregressive, transformer encoder-decoder | mel | yes |
| `fastspeech2` | non-autoregressive, built-in aligner, duration/pitch/energy | mel | yes |
| `vits` | end-to-end VAE + flow + GAN | waveform | no |
| `hifigan` | neural vocoder (train it on the same data, then chain it with a mel model) | waveform | - |

This notebook is **self-contained**: all the code is copied in from the repo with the same
defaults, and its checkpoints work with the repo's `tts-synthesize` and `tts-eval`.

| | Kaggle | Colab |
|---|---|---|
| GPU | *Settings -> Accelerator -> GPU T4 x2* (or P100) | *Runtime -> Change runtime type -> T4 GPU* |
| Data | your dataset as input, or `DOWNLOAD_DATASET` | Google Drive, upload, or `DOWNLOAD_DATASET` |
| Session limit | 12 h per session, 30 h GPU / week | ~12 h, can disconnect when idle |
| Checkpoints kept in | `/kaggle/working` (saved with "Save Version") | Google Drive if `USE_GOOGLE_DRIVE = True` |

**Just trying it?** Keep `SMOKE_TEST = True`: the notebook writes the toy language dataset and
runs every cell in a few minutes. For real speech set `DOWNLOAD_DATASET = "ljspeech"` (2.6 GB, 24 h
of one English speaker) and the 22.05 kHz audio settings, or `"fsdd"` / `"yesno"` for small 8 kHz tests.

**Sections:** 1 Install - 2 Config - 3 Text & audio - 4 Building blocks & alignment - 5 Models -
6 Vocoders & losses - 7 Data - 8 Training code - 9 Train - 10 Results (listen, MCD/F0/WER/UTMOS) -
11 Download the checkpoint
""")

md("""
## 1. Install

Kaggle and Colab already have PyTorch, NumPy, SciPy and matplotlib. This cell adds `soundfile`
(audio I/O) and `transformers` (Whisper, for the intelligibility WER in section 10).
""")
code("""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "soundfile>=0.12", "transformers>=4.40"])

import torch

print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
""")

md(f"""
## 2. Config

### 2.1 Config classes (`src/tts/config.py`)

{module_doc("config")}
""")
code(module_source("config"))

md("""
### 2.2 Settings for this run

**Where is your data?** A folder in the LJSpeech layout (`metadata.csv` + `wavs/`, optional
`speakers.csv`), see the repo's `docs/data.md`.

- **Kaggle:** add the dataset as input; the cell below finds it.
- **Colab:** put it on Google Drive (`USE_GOOGLE_DRIVE = True`) and set `DATA_DIR`.
- **Either:** set `DOWNLOAD_DATASET` to `"ljspeech"`, `"fsdd"` or `"yesno"`.

`AUDIO` must match the data: 8 kHz for the small sets, 22.05 kHz / 80 mels for LJSpeech.
""")
code("""
import os
from pathlib import Path

SMOKE_TEST = True            # True = toy language data, small model, a few minutes
MODEL = "tacotron2"          # "tacotron2" | "transformer_tts" | "fastspeech2" | "vits" | "hifigan"
VOCODER = "griffinlim"       # "griffinlim" | "hifigan" (needs HIFIGAN_CHECKPOINT) - ignored by vits
HIFIGAN_CHECKPOINT = None    # path of a hifigan best.pt trained with this notebook (MODEL = "hifigan")
DOWNLOAD_DATASET = None      # None | "ljspeech" (2.6 GB) | "fsdd" (20 MB) | "yesno" (4.5 MB)
USE_GOOGLE_DRIVE = False     # Colab only: data + checkpoints on Drive (survive disconnects)
DATA_DIR = None              # None = find it automatically (Kaggle input, /content/data, ./data)
RESUME = True                # continue from last.pt in OUTPUT_DIR if it exists
TIME_BUDGET_HOURS = 11.0     # stop cleanly (after saving) before the 12 h session limit
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
AUDIO = dict(sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmax=8000.0)
UPSAMPLE_RATES = (8, 8, 4)   # hifigan / vits: product must equal hop_length (8*8*4 = 256)
if SMOKE_TEST or DOWNLOAD_DATASET in ("fsdd", "yesno"):
    AUDIO = dict(sample_rate=8000, n_fft=512, hop_length=128, win_length=512, n_mels=64, fmax=4000.0)
    UPSAMPLE_RATES = (8, 4, 4)

ON_KAGGLE = Path("/kaggle/working").exists()
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ
if ON_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_DIR = Path(f"/content/drive/MyDrive/tts-workbench/{MODEL}")
elif ON_KAGGLE:
    OUTPUT_DIR = Path(f"/kaggle/working/checkpoints/{MODEL}")
elif ON_COLAB:
    OUTPUT_DIR = Path(f"/content/checkpoints/{MODEL}")
else:
    OUTPUT_DIR = Path(f"checkpoints/notebook-{MODEL}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_data_dir():
    roots = [Path("/kaggle/input"), Path("/content/data"), Path("/content/drive/MyDrive"), Path("data")]
    for root in roots:
        if not root.exists():
            continue
        for path in [root, *sorted(p for p in root.rglob("*") if p.is_dir())][:500]:
            if (path / "metadata.csv").exists() and (path / "wavs").is_dir():
                return path
    return None


print("output:", OUTPUT_DIR)
""")

md("## 3. Text & audio")
module_cells("text", "3.1 Text front-end")
module_cells("audio", "3.2 Mel-spectrograms, Griffin-Lim, F0")

md("## 4. Building blocks & alignment")
module_cells("layers", "4.1 Shared layers")
module_cells("align", "4.2 Monotonic alignment search and the aligner")

md("## 5. Models")
module_cells("tacotron2", "5.1 Tacotron 2")
module_cells("transformer_tts", "5.2 Transformer-TTS")
module_cells("fastspeech2", "5.3 FastSpeech 2")

md("## 6. Vocoders, GAN losses and VITS")
module_cells("vocoders", "6.1 HiFi-GAN generator and discriminators")
module_cells("losses", "6.2 GAN / KL losses")
module_cells("vits", "6.3 VITS")

md("## 7. Data")
module_cells("data", "7.1 Dataset and batching")
module_cells("datasets", "7.2 Built-in datasets")
code("""
if SMOKE_TEST:
    DATA_DIR = write_synthetic(Path("data/synthetic"), n=200, sample_rate=AUDIO["sample_rate"])
elif DOWNLOAD_DATASET == "ljspeech":
    DATA_DIR = write_ljspeech(Path("data/ljspeech"))
elif DOWNLOAD_DATASET == "fsdd":
    DATA_DIR = write_fsdd(Path("data/fsdd"))
elif DOWNLOAD_DATASET == "yesno":
    DATA_DIR = write_yesno(Path("data/yesno"))
else:
    DATA_DIR = Path(DATA_DIR) if DATA_DIR else find_data_dir()
assert DATA_DIR is not None, "no dataset found: set DATA_DIR or DOWNLOAD_DATASET"
items = read_manifest(DATA_DIR)
print(f"{len(items)} clips in {DATA_DIR}; e.g. {items[0].id}: {items[0].text!r}")
""")

md("## 8. Training, synthesis and metrics code")
module_cells("models", "8.1 Model registry and checkpoints")
module_cells("train", "8.2 Trainer")
module_cells("synthesize", "8.3 Synthesizer")
module_cells("metrics", "8.4 Metrics")

md("""
## 9. Train

The config below is the repo default (a few million parameters at most) scaled by the flags of
section 2.2. Curves and alignment plots land in `OUTPUT_DIR`; the cell after the training draws them.
""")
code("""
import dataclasses
import json

cfg = Config()
for k, v in AUDIO.items():
    setattr(cfg.audio, k, v)
cfg.model.model = MODEL
cfg.model.upsample_rates = UPSAMPLE_RATES
cfg.train.data_dir = str(DATA_DIR)
cfg.train.output_dir = str(OUTPUT_DIR)
cfg.train.epochs = 8 if SMOKE_TEST else EPOCHS
cfg.train.batch_size = 16 if SMOKE_TEST else BATCH_SIZE
cfg.train.lr = LR
cfg.train.resume = RESUME
cfg.train.time_budget_hours = 0.2 if SMOKE_TEST else TIME_BUDGET_HOURS
cfg.train.amp = torch.cuda.is_available() and MODEL not in ("hifigan", "vits")
cfg.train.eval_every = 1 if SMOKE_TEST else 5
cfg.train.log_every = 10
if SMOKE_TEST:
    cfg.model.hidden = 64
    cfg.model.upsample_initial = 64
print(json.dumps(dataclasses.asdict(cfg.model), indent=1))

trainer = Trainer(cfg)
summary = trainer.fit()
print(summary)
""")
code("""
import matplotlib.pyplot as plt
import pandas as pd

curves = pd.read_csv(OUTPUT_DIR / "curves.csv")
train_rows = curves[curves["loss"].notna()] if "loss" in curves else curves
val_cols = [c for c in curves.columns if c.startswith("val_")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train_rows.plot(x="step", y=[c for c in ("loss", "mel", "mel_after", "kl", "duration") if c in train_rows], ax=axes[0], title="training")
curves[curves[val_cols[0]].notna()].plot(x="step", y=val_cols[:4], ax=axes[1], title="validation")
plt.show()
plots = sorted(OUTPUT_DIR.glob("alignment_*.png"))
if plots:
    from IPython.display import Image, display

    display(Image(filename=str(plots[-1])))
""")

md("""
## 10. Results

Synthesize a few sentences, listen, and score the validation set: **MCD** (lower is better),
**log-F0 RMSE**, **duration error**, **attention diagonality** and, with Whisper and UTMOS, the
**WER** (can the words be understood?) and a **predicted MOS** (1..5, how natural does it sound?).
UTMOS is downloaded through `torch.hub` (needs internet in the session); for a human MOS see the
repo's `docs/evaluation.md`.
""")
code("""
from IPython.display import Audio, display

ckpt = OUTPUT_DIR / "best.pt"
if MODEL == "hifigan":
    print("hifigan is a vocoder: copy-synthesis of a validation clip")
    syn_v = Synthesizer(ckpt)
    item = items[0]
    ref = load_wav(str(item.audio), cfg.audio.sample_rate)
    mel = MelSpectrogram(cfg.audio)(torch.from_numpy(ref)[None])[0]
    wav = syn_v.model(mel[None])[0]
    display(Audio(ref, rate=cfg.audio.sample_rate))
    display(Audio(wav.numpy(), rate=cfg.audio.sample_rate))
else:
    syn = Synthesizer(ckpt, VOCODER, HIFIGAN_CHECKPOINT, "cuda" if torch.cuda.is_available() else "cpu")
    sentences = [it.text for it in items[:3]] if SMOKE_TEST else [
        "The quick brown fox jumps over the lazy dog.",
        "Text to speech turns written words into sound.",
        items[0].text,
    ]
    for text in sentences:
        wav, attn = syn(text)
        print(text)
        display(Audio(wav.cpu().numpy(), rate=cfg.audio.sample_rate))
    if attn is not None:
        plt.imshow(attn.cpu().numpy().T, aspect="auto", origin="lower")
        plt.xlabel("frames")
        plt.ylabel("symbols")
        plt.title("alignment of the last sentence")
        plt.show()
""")
code("""
USE_WHISPER = not SMOKE_TEST   # the toy language has no words Whisper could recognise
USE_UTMOS = not SMOKE_TEST
if MODEL != "hifigan":
    result = evaluate(str(ckpt), str(DATA_DIR), VOCODER, HIFIGAN_CHECKPOINT, n=10, out_dir=str(OUTPUT_DIR / "eval"),
                      whisper=USE_WHISPER, utmos=USE_UTMOS, device="cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(result["summary"], indent=2))
    pd.DataFrame(result["items"]).drop(columns=["text"]).head(10)
""")

md("""
## 11. Download the trained model

`best.pt` holds the config, tokenizer kind, speaker map and weights, so it is all `tts-synthesize`
needs: `tts-synthesize --checkpoint best.pt --text "..."`. On Kaggle, click *Save Version* to keep
`/kaggle/working`; on Colab, the cell below downloads the file.
""")
code("""
if ON_COLAB:
    from google.colab import files

    files.download(str(OUTPUT_DIR / "best.pt"))
else:
    print("checkpoint:", OUTPUT_DIR / "best.pt")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.validate(nb)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
