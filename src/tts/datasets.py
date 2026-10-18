"""Built-in datasets, written in the LJSpeech layout by `tts-data <name> --out <dir>`.

- `synthetic`: a deterministic **toy spoken language** at 8 kHz. Every letter is a short
  harmonic tone with its own fundamental frequency and spectral envelope (a crude vowel),
  spaces are silences, and 25 dB of noise is added. The text -> audio mapping is simple and
  fully known, so a model that learns it proves the pipeline (alignment, durations, pitch,
  vocoder) works before any real data is touched. Used by the tests and the smoke runs.
- `fsdd`: Free Spoken Digit Dataset (3,000 recordings of "zero".."nine" by 6 speakers at
  8 kHz, ~ 20 MB). Real, low-quality speech; `--speaker jackson` (default) keeps one voice.
- `yesno`: 60 recordings of eight-word yes/no sequences at 8 kHz (openslr 1, 4.5 MB). The
  words are the Hebrew "ken"/"lo"; the transcripts use "yes"/"no".
- `ljspeech`: the standard 24 h single-speaker English corpus at 22.05 kHz (2.6 GB); for
  Kaggle/Colab. Use `--sample-rate 22050 --n-fft 1024 --hop-length 256 --win-length 1024
  --n-mels 80 --fmax 8000` with it.
- VCTK and any other corpus: write `metadata.csv` + `wavs/` (+ `speakers.csv`) yourself,
  or a `manifest.jsonl`.
"""

from __future__ import annotations

import argparse
import math
import random
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from tts.audio import save_wav

FSDD_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
YESNO_URL = "https://www.openslr.org/resources/1/waves_yesno.tar.gz"
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

TOY_WORDS = ["ba", "ko", "mi", "tu", "sel", "ran", "dop", "lia", "nef", "goz", "we", "pyx", "hu", "qim", "jav"]


def _letter_profile(ch: str) -> tuple[float, list[float], float]:
    """(fundamental Hz, harmonic amplitudes, duration s) for one letter, deterministic."""
    i = ord(ch) - ord("a")
    f0 = 110.0 * 2 ** (i / 12.0)  # one semitone per letter: 110 Hz (a) .. 467 Hz (z)
    rng = random.Random(1000 + i)
    harmonics = [1.0] + [rng.uniform(0.1, 0.8) / (k + 1) for k in range(1, 5)]
    duration = 0.06 + 0.04 * ((i * 7) % 5) / 4  # 60..100 ms
    return f0, harmonics, duration


def synth_toy_wave(text: str, sample_rate: int = 8000, noise_db: float = -25.0, seed: int = 0) -> np.ndarray:
    """Render text in the toy language: letters -> harmonic tones, spaces -> 80 ms silence."""
    rng = np.random.default_rng(seed)
    pieces = [np.zeros(int(0.05 * sample_rate), dtype=np.float32)]
    for ch in text.lower():
        if ch == " ":
            pieces.append(np.zeros(int(0.08 * sample_rate), dtype=np.float32))
            continue
        if not ch.isalpha():
            continue
        f0, harmonics, dur = _letter_profile(ch)
        n = int(dur * sample_rate)
        t = np.arange(n) / sample_rate
        glide = 1.0 + 0.03 * np.sin(2 * math.pi * 3 * t)  # slight vibrato so F0 tracking is not trivial
        wave = sum(a * np.sin(2 * math.pi * f0 * (k + 1) * np.cumsum(glide) / sample_rate)
                   for k, a in enumerate(harmonics))
        env = np.minimum(1.0, np.minimum(t / 0.01, (dur - t) / 0.015))  # attack / release
        pieces.append((0.4 * wave * np.clip(env, 0, 1)).astype(np.float32))
    pieces.append(np.zeros(int(0.05 * sample_rate), dtype=np.float32))
    wav = np.concatenate(pieces)
    wav = wav + rng.normal(0, 10 ** (noise_db / 20), wav.shape).astype(np.float32)
    return np.clip(wav, -1, 1)


def toy_sentence(rng: random.Random, min_words: int = 2, max_words: int = 5) -> str:
    return " ".join(rng.choice(TOY_WORDS) for _ in range(rng.randint(min_words, max_words)))


def write_synthetic(out: Path, n: int = 300, sample_rate: int = 8000, seed: int = 0, n_speakers: int = 1) -> Path:
    """`n` toy sentences. With n_speakers > 1 every speaker shifts all pitches (a crude voice)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "wavs").mkdir(exist_ok=True)
    rng = random.Random(seed)
    lines, speakers = [], []
    for i in range(n):
        text = toy_sentence(rng)
        spk = i % n_speakers
        wav = synth_toy_wave(text, sample_rate, seed=seed + i)
        if spk:
            wav = _pitch_shift(wav, 2 ** (spk / 12.0 * 3))
        save_wav(str(out / "wavs" / f"toy{i:04d}.wav"), wav, sample_rate)
        lines.append(f"toy{i:04d}|{text}")
        speakers.append(f"toy{i:04d}|spk{spk}")
    (out / "metadata.csv").write_text("\n".join(lines) + "\n")
    if n_speakers > 1:
        (out / "speakers.csv").write_text("\n".join(speakers) + "\n")
    return out


def _pitch_shift(wav: np.ndarray, factor: float) -> np.ndarray:
    """Crude resampling pitch shift (changes duration too; fine for a toy)."""
    idx = np.arange(0, len(wav), factor)
    return np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    return dest


def write_fsdd(out: Path, speaker: str | None = "jackson", cache: Path | None = None) -> Path:
    """Free Spoken Digit Dataset; `speaker=None` keeps all six speakers (writes speakers.csv)."""
    zip_path = _download(FSDD_URL, (cache or out) / "fsdd-master.zip")
    out.mkdir(parents=True, exist_ok=True)
    (out / "wavs").mkdir(exist_ok=True)
    lines, speakers = [], []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if "/recordings/" not in name or not name.endswith(".wav"):
                continue
            stem = Path(name).stem  # <digit>_<speaker>_<index>
            digit, spk, _ = stem.split("_", 2)
            if speaker and spk != speaker:
                continue
            (out / "wavs" / f"{stem}.wav").write_bytes(zf.read(name))
            lines.append(f"{stem}|{DIGITS[int(digit)]}")
            speakers.append(f"{stem}|{spk}")
    (out / "metadata.csv").write_text("\n".join(lines) + "\n")
    if not speaker:
        (out / "speakers.csv").write_text("\n".join(speakers) + "\n")
    print(f"fsdd: {len(lines)} clips -> {out}")
    return out


def write_yesno(out: Path, cache: Path | None = None) -> Path:
    tar_path = _download(YESNO_URL, (cache or out) / "waves_yesno.tar.gz")
    out.mkdir(parents=True, exist_ok=True)
    (out / "wavs").mkdir(exist_ok=True)
    lines = []
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".wav"):
                continue
            stem = Path(member.name).stem
            data = tf.extractfile(member).read()
            (out / "wavs" / f"{stem}.wav").write_bytes(data)
            lines.append(f"{stem}|" + " ".join("yes" if c == "1" else "no" for c in stem.split("_")))
    (out / "metadata.csv").write_text("\n".join(sorted(lines)) + "\n")
    print(f"yesno: {len(lines)} clips -> {out}")
    return out


def write_ljspeech(out: Path, cache: Path | None = None) -> Path:
    """Downloads 2.6 GB and extracts LJSpeech-1.1 (already in the right layout) into `out`."""
    tar_path = _download(LJSPEECH_URL, (cache or out.parent) / "LJSpeech-1.1.tar.bz2")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:bz2") as tf:
        tf.extractall(out.parent)
    src = out.parent / "LJSpeech-1.1"
    if src != out:
        src.rename(out)
    print(f"ljspeech -> {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Write a built-in dataset in the LJSpeech layout.")
    p.add_argument("name", choices=["synthetic", "fsdd", "yesno", "ljspeech"])
    p.add_argument("--out", default=None, help="default: data/<name>")
    p.add_argument("--n", type=int, default=300, help="synthetic: number of sentences")
    p.add_argument("--n-speakers", type=int, default=1, help="synthetic: number of toy speakers")
    p.add_argument("--sample-rate", type=int, default=8000, help="synthetic only")
    p.add_argument("--speaker", default="jackson", help="fsdd: speaker to keep, 'all' for every speaker")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out = Path(a.out or f"data/{a.name}")
    if a.name == "synthetic":
        write_synthetic(out, a.n, a.sample_rate, a.seed, a.n_speakers)
        print(f"synthetic: {a.n} clips -> {out}")
    elif a.name == "fsdd":
        write_fsdd(out, None if a.speaker == "all" else a.speaker)
    elif a.name == "yesno":
        write_yesno(out)
    else:
        write_ljspeech(out)


if __name__ == "__main__":
    main()

__all__ = [
    "DIGITS",
    "TOY_WORDS",
    "main",
    "synth_toy_wave",
    "toy_sentence",
    "write_fsdd",
    "write_ljspeech",
    "write_synthetic",
    "write_yesno",
]
