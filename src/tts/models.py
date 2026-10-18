"""Model registry and checkpoints.

A checkpoint is one `.pt` file holding the config, the tokenizer kind, the speaker map and
the weights (plus the discriminators and optimiser states for resuming), so `tts-synthesize`
and `tts-eval` need nothing but the file.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from tts.config import Config
from tts.fastspeech2 import FastSpeech2
from tts.tacotron2 import Tacotron2
from tts.text import Tokenizer
from tts.transformer_tts import TransformerTTS
from tts.vits import VITS
from tts.vocoders import HiFiGANGenerator


def build_model(cfg: Config, vocab_size: int) -> nn.Module:
    name = cfg.model.model
    if name == "tacotron2":
        return Tacotron2(cfg, vocab_size)
    if name == "transformer_tts":
        return TransformerTTS(cfg, vocab_size)
    if name == "fastspeech2":
        return FastSpeech2(cfg, vocab_size)
    if name == "vits":
        return VITS(cfg, vocab_size)
    if name == "hifigan":
        return HiFiGANGenerator(cfg)
    raise ValueError(f"unknown model {name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path: str | Path, model: nn.Module, cfg: Config, speaker_ids: dict[str, int], step: int,
                    epoch: int, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {"cfg": cfg.to_dict(), "model": model.state_dict(), "speaker_ids": speaker_ids, "step": step,
             "epoch": epoch}
    state.update(extra or {})
    torch.save(state, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, Config, Tokenizer, dict]:
    """-> (model in eval mode, config, tokenizer, raw checkpoint dict)."""
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(state["cfg"])
    tok = Tokenizer(cfg.text.tokenizer, cfg.text.lowercase)
    model = build_model(cfg, tok.vocab_size)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, cfg, tok, state


__all__ = ["build_model", "count_parameters", "load_checkpoint", "save_checkpoint"]
