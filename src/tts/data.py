"""Datasets and batching.

Every dataset is a folder in the **LJSpeech layout**:

```
<data_dir>/metadata.csv      lines "<id>|<text>" or "<id>|<raw text>|<normalised text>"
<data_dir>/wavs/<id>.wav
<data_dir>/speakers.csv      optional: "<id>|<speaker name>" for multi-speaker data
```

or a `manifest.jsonl` with `{"audio": ..., "text": ..., "speaker": ...}` per line. `tts-data`
writes the built-in datasets in this layout.

`TTSDataset` computes the log-mel, F0 and energy of each clip once and caches them as
`.pt` files under `<data_dir>/cache/`, so epochs after the first only read tensors.
A batch is a dict of padded tensors:
`text (B, N)`, `text_lens`, `mel (B, n_mels, T)`, `mel_lens`, `f0 (B, T)`, `energy (B, T)`,
`wav (B, samples)`, `speaker (B,)`, plus `linear (B, n_fft/2+1, T)` for VITS and
`durations (B, N)` when teacher durations exist in `<data_dir>/durations/`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from tts.audio import MelSpectrogram, estimate_f0, frame_energy, load_wav
from tts.config import AudioConfig
from tts.text import Tokenizer


@dataclass
class Item:
    id: str
    audio: Path
    text: str
    speaker: str = "0"


def read_manifest(data_dir: str | Path) -> list[Item]:
    data_dir = Path(data_dir)
    items: list[Item] = []
    if (data_dir / "manifest.jsonl").exists():
        for line in (data_dir / "manifest.jsonl").read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                audio = Path(d["audio"])
                audio = audio if audio.is_absolute() else data_dir / audio
                items.append(Item(d.get("id", audio.stem), audio, d["text"], str(d.get("speaker", "0"))))
        return items
    meta = data_dir / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"no metadata.csv or manifest.jsonl in {data_dir}; run tts-data first")
    speakers: dict[str, str] = {}
    if (data_dir / "speakers.csv").exists():
        for line in (data_dir / "speakers.csv").read_text().splitlines():
            if "|" in line:
                k, v = line.split("|", 1)
                speakers[k] = v
    for line in meta.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        text = parts[2] if len(parts) > 2 and parts[2].strip() else parts[1]
        items.append(Item(parts[0], data_dir / "wavs" / f"{parts[0]}.wav", text.strip(), speakers.get(parts[0], "0")))
    return items


def split_items(items: list[Item], val_fraction: float = 0.05, seed: int = 0) -> tuple[list[Item], list[Item]]:
    """Deterministic train/val split (at least 2 validation items)."""
    rng = random.Random(seed)
    order = list(items)
    rng.shuffle(order)
    n_val = max(2, int(len(order) * val_fraction)) if len(order) > 4 else 1
    return order[n_val:], order[:n_val]


class TTSDataset(Dataset):
    def __init__(self, items: list[Item], audio: AudioConfig, tokenizer: Tokenizer, data_dir: str | Path,
                 speaker_ids: dict[str, int] | None = None, need_linear: bool = False, need_wav: bool = False,
                 max_frames: int = 0):
        self.items, self.audio, self.tok = items, audio, tokenizer
        self.mel = MelSpectrogram(audio)
        self.cache = Path(data_dir) / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.durations_dir = Path(data_dir) / "durations"
        self.speaker_ids = speaker_ids or {s: i for i, s in enumerate(sorted({it.speaker for it in items}))}
        self.need_linear, self.need_wav, self.max_frames = need_linear, need_wav, max_frames
        if max_frames:
            self.items = [it for it in self.items if self._n_frames(it) <= max_frames]

    def _n_frames(self, item: Item) -> int:
        return self._features(item)["mel"].size(1)

    def _features(self, item: Item) -> dict:
        tag = f"{self.audio.sample_rate}_{self.audio.n_fft}_{self.audio.hop_length}_{self.audio.n_mels}"
        path = self.cache / f"{item.id}_{tag}.pt"
        if path.exists():
            return torch.load(path)
        wav = load_wav(str(item.audio), self.audio.sample_rate)
        wav_t = torch.from_numpy(wav)
        mel = self.mel(wav_t[None])[0]
        f0 = torch.from_numpy(estimate_f0(wav, self.audio))[: mel.size(1)]
        f0 = torch.nn.functional.pad(f0, (0, mel.size(1) - f0.numel()))
        feats = {"mel": mel, "f0": f0, "energy": frame_energy(mel), "wav": wav_t}
        torch.save(feats, path)
        return feats

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        item = self.items[i]
        feats = self._features(item)
        out = {"id": item.id, "text": torch.tensor(self.tok.encode(item.text)), "mel": feats["mel"], "f0": feats["f0"],
               "energy": feats["energy"], "speaker": self.speaker_ids.get(item.speaker, 0)}
        if self.need_wav or self.need_linear:
            out["wav"] = feats["wav"]
        if self.need_linear:
            out["linear"] = self.mel.linear(feats["wav"][None])[0]
        dur = self.durations_dir / f"{item.id}.pt"
        if dur.exists():
            out["durations"] = torch.load(dur)
        return out


def collate(samples: list[dict]) -> dict:
    """Pad every sequence in the batch to the longest one."""
    B = len(samples)
    N = max(s["text"].numel() for s in samples)
    T = max(s["mel"].size(1) for s in samples)
    n_mels = samples[0]["mel"].size(0)
    batch = {"ids": [s["id"] for s in samples], "text": torch.zeros(B, N, dtype=torch.long),
             "text_lens": torch.tensor([s["text"].numel() for s in samples]),
             "mel": torch.full((B, n_mels, T), float(samples[0]["mel"].min())),
             "mel_lens": torch.tensor([s["mel"].size(1) for s in samples]),
             "f0": torch.zeros(B, T), "energy": torch.zeros(B, T),
             "speaker": torch.tensor([s["speaker"] for s in samples])}
    for b, s in enumerate(samples):
        n, t = s["text"].numel(), s["mel"].size(1)
        batch["text"][b, :n] = s["text"]
        batch["mel"][b, :, :t] = s["mel"]
        batch["f0"][b, :t], batch["energy"][b, :t] = s["f0"][:t], s["energy"][:t]
    if "wav" in samples[0]:
        S = max(s["wav"].numel() for s in samples)
        batch["wav"] = torch.zeros(B, S)
        for b, s in enumerate(samples):
            batch["wav"][b, : s["wav"].numel()] = s["wav"]
    if "linear" in samples[0]:
        batch["linear"] = torch.zeros(B, samples[0]["linear"].size(0), T)
        for b, s in enumerate(samples):
            batch["linear"][b, :, : s["linear"].size(1)] = s["linear"][:, :T]
    if all("durations" in s for s in samples):
        batch["durations"] = torch.zeros(B, N, dtype=torch.long)
        for b, s in enumerate(samples):
            d = s["durations"][:N]
            batch["durations"][b, : d.numel()] = d
    return batch


def make_loader(dataset: TTSDataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=num_workers,
                      drop_last=False)


__all__ = ["Item", "TTSDataset", "collate", "make_loader", "read_manifest", "split_items"]
