"""Configuration: small dataclasses, overridable from the command line.

Every field of every dataclass becomes a `--flag-name` option (`AudioConfig.sample_rate`
-> `--sample-rate`). Defaults are deliberately small: these models are built to be
understood and trained on a laptop CPU or a free Kaggle GPU, not to beat benchmarks.
Scale `hidden`, `layers` and `batch_size` up for real data.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

MODELS = ("tacotron2", "transformer_tts", "fastspeech2", "vits", "hifigan")
VOCODERS = ("griffinlim", "hifigan")


@dataclass
class AudioConfig:
    """How waveforms become (log-)mel-spectrograms and back."""

    sample_rate: int = 8000  # 8 kHz default matches the toy / FSDD / YESNO datasets; use 22050 for LJSpeech
    n_fft: int = 512
    hop_length: int = 128  # 16 ms at 8 kHz
    win_length: int = 512
    n_mels: int = 64
    fmin: float = 0.0
    fmax: float = 4000.0  # <= sample_rate / 2
    log_floor: float = 1e-5  # log(max(mel, floor))
    f0_min: float = 60.0
    f0_max: float = 400.0


@dataclass
class TextConfig:
    """Text front-end."""

    tokenizer: str = "char"  # "char" | "phoneme" (needs g2p-en)
    lowercase: bool = True


@dataclass
class ModelConfig:
    """Architecture sizes shared by all models (each model reads what it needs)."""

    model: str = "tacotron2"  # one of MODELS
    hidden: int = 128  # encoder/decoder width (Tacotron2 paper: 512)
    layers: int = 2  # FFT / transformer blocks per stack
    heads: int = 2
    ffn_kernel: int = 3
    dropout: float = 0.1
    prenet_dim: int = 64
    attention_dim: int = 64
    attention_filters: int = 16
    attention_kernel: int = 15
    reduction_factor: int = 2  # Tacotron2 / Transformer-TTS: mel frames per decoder step
    postnet_layers: int = 3
    postnet_channels: int = 128
    n_speakers: int = 1  # > 1 enables a speaker embedding (FiLM-style add) in every model
    speaker_dim: int = 32
    max_decoder_steps: int = 600  # inference safety stop for autoregressive models
    # VITS / HiFi-GAN
    latent_dim: int = 32
    flow_blocks: int = 2
    upsample_rates: tuple[int, ...] = (8, 4, 4)  # product must equal AudioConfig.hop_length
    upsample_initial: int = 64
    resblock_kernels: tuple[int, ...] = (3, 7)
    resblock_dilations: tuple[int, ...] = (1, 3)
    segment_frames: int = 16  # frames of random audio segment fed to the waveform decoder in GAN training


@dataclass
class TrainConfig:
    """Optimisation."""

    data_dir: str = "data/synthetic"
    output_dir: str = "checkpoints/run"
    epochs: int = 20
    max_steps: int = 0  # 0 = epochs decide
    batch_size: int = 16
    lr: float = 1e-3
    warmup_steps: int = 200
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    amp: bool = False  # bf16/fp16 autocast on GPU
    num_workers: int = 0
    seed: int = 0
    log_every: int = 10
    eval_every: int = 1  # epochs
    time_budget_hours: float = 0.0  # 0 = no limit
    resume: bool = False
    duration_source: str = "aligner"  # fastspeech2: "aligner" | "teacher" (durations cached by tts-eval --export-durations)
    bin_warmup_steps: int = 300  # fastspeech2: steps before the aligner's binarisation loss ramps in
    guided_attention: float = 1.0  # weight of the diagonal guided-attention loss (AR models), 0 = off
    max_frames: int = 600  # clips longer than this (in mel frames) are skipped
    device: str = "auto"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    text: TextConfig = field(default_factory=TextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        cfg = cls()
        for section in fields(cls):
            sub = getattr(cfg, section.name)
            for f in fields(sub):
                if f.name in d.get(section.name, {}):
                    value = d[section.name][f.name]
                    if isinstance(value, list):
                        value = tuple(value)
                    setattr(sub, f.name, value)
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _parse_value(text: str, default: Any) -> Any:
    if isinstance(default, bool):
        return text.lower() in ("1", "true", "yes", "on")
    if isinstance(default, tuple):
        return tuple(int(x) for x in text.split(","))
    if isinstance(default, int):
        return int(text)
    if isinstance(default, float):
        return float(text)
    return text


def add_config_args(parser: argparse.ArgumentParser, cfg: Config | None = None) -> None:
    """One `--flag` per dataclass field; the help shows the default."""
    cfg = cfg or Config()
    for section in fields(Config):
        group = parser.add_argument_group(section.name)
        sub = getattr(cfg, section.name)
        for f in fields(sub):
            default = getattr(sub, f.name)
            group.add_argument(f"--{f.name.replace('_', '-')}", dest=f"{section.name}.{f.name}", default=None,
                               metavar=type(default).__name__.upper(), help=f"default: {default}")


def config_from_args(args: argparse.Namespace, base: Config | None = None) -> Config:
    """Apply the given `--flags` on top of `base` (or the defaults)."""
    cfg = base or Config()
    for key, value in vars(args).items():
        if "." not in key or value is None:
            continue
        section, name = key.split(".", 1)
        sub = getattr(cfg, section)
        setattr(sub, name, _parse_value(value, getattr(sub, name)))
    return cfg


__all__ = [
    "MODELS",
    "VOCODERS",
    "AudioConfig",
    "Config",
    "ModelConfig",
    "TextConfig",
    "TrainConfig",
    "add_config_args",
    "config_from_args",
]
