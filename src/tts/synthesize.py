"""Text -> waveform with any acoustic model and any vocoder.

```
tts-synthesize --checkpoint checkpoints/run/best.pt --text "hello world" --out synthesized/
tts-synthesize --checkpoint ... --text-file sentences.txt --vocoder hifigan --vocoder-checkpoint checkpoints/hifigan/best.pt
```

Mel models (Tacotron 2, Transformer-TTS, FastSpeech 2) produce a log-mel that the vocoder
turns into audio: Griffin-Lim by default, or a trained HiFi-GAN. VITS produces the
waveform directly and ignores the vocoder flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tts.audio import griffin_lim, save_wav
from tts.models import load_checkpoint


class Synthesizer:
    def __init__(self, checkpoint: str | Path, vocoder: str = "griffinlim", vocoder_checkpoint: str | Path | None = None,
                 device: str = "cpu", griffin_lim_iters: int = 32):
        self.model, self.cfg, self.tok, state = load_checkpoint(checkpoint, device)
        self.speaker_ids = state.get("speaker_ids", {})
        self.device = device
        self.vocoder_name = vocoder
        self.gl_iters = griffin_lim_iters
        self.vocoder = None
        if vocoder == "hifigan" and self.cfg.model.model != "vits":
            if vocoder_checkpoint is None:
                raise ValueError("--vocoder hifigan needs --vocoder-checkpoint")
            self.vocoder, vcfg, _, _ = load_checkpoint(vocoder_checkpoint, device)
            assert vcfg.audio.n_mels == self.cfg.audio.n_mels, "vocoder and acoustic model must share the mel setup"

    @property
    def sample_rate(self) -> int:
        return self.cfg.audio.sample_rate

    def speaker_tensor(self, speaker: str | int | None) -> torch.Tensor | None:
        if speaker is None or self.cfg.model.n_speakers <= 1:
            return None
        idx = self.speaker_ids.get(str(speaker), speaker if isinstance(speaker, int) else 0)
        return torch.tensor([int(idx)], device=self.device)

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """log-mel (n_mels, T) -> waveform."""
        if self.vocoder is not None:
            with torch.no_grad():
                return self.vocoder(mel[None])[0]
        return griffin_lim(mel, self.cfg.audio, n_iter=self.gl_iters)

    @torch.no_grad()
    def mel(self, text: str, speaker=None, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        """-> log-mel (n_mels, T) and alignment (T or T/r, N). Not for VITS."""
        ids = torch.tensor([self.tok.encode(text)], device=self.device)
        mel, attn = self.model.infer(ids, self.speaker_tensor(speaker), **kw)
        return mel[0], attn

    @torch.no_grad()
    def __call__(self, text: str, speaker=None, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        """text -> (waveform, alignment)."""
        ids = torch.tensor([self.tok.encode(text)], device=self.device)
        if self.cfg.model.model == "vits":
            wav, attn = self.model.infer(ids, self.speaker_tensor(speaker), **kw)
            return wav, attn
        mel, attn = self.model.infer(ids, self.speaker_tensor(speaker), **kw)
        return self.vocode(mel[0]), attn


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Synthesize speech from text.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", action="append", default=[], help="sentence (repeatable)")
    p.add_argument("--text-file", default=None, help="one sentence per line")
    p.add_argument("--vocoder", default="griffinlim", choices=["griffinlim", "hifigan"])
    p.add_argument("--vocoder-checkpoint", default=None)
    p.add_argument("--speaker", default=None)
    p.add_argument("--out", default="synthesized")
    p.add_argument("--device", default="cpu")
    p.add_argument("--duration-scale", type=float, default=1.0, help="fastspeech2 / vits: > 1 speaks slower")
    a = p.parse_args(argv)
    texts = list(a.text)
    if a.text_file:
        texts += [line.strip() for line in Path(a.text_file).read_text().splitlines() if line.strip()]
    if not texts:
        p.error("give --text or --text-file")
    syn = Synthesizer(a.checkpoint, a.vocoder, a.vocoder_checkpoint, a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = {"duration_scale": a.duration_scale} if syn.cfg.model.model in ("fastspeech2", "vits") else {}
    manifest = []
    for i, text in enumerate(texts):
        wav, _ = syn(text, a.speaker, **kw)
        path = out / f"{i:03d}.wav"
        save_wav(str(path), wav, syn.sample_rate)
        manifest.append({"audio": str(path), "text": text})
        print(f"{path}: {len(wav) / syn.sample_rate:.2f} s | {text}")
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")


if __name__ == "__main__":
    main()

__all__ = ["Synthesizer", "main"]


