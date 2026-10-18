"""Training: one loop per family of models.

- **mel models** (tacotron2, transformer_tts, fastspeech2): one AdamW optimiser, the model's
  own `forward(batch)` returns every loss term; linear warm-up then inverse-sqrt decay.
- **hifigan**: random `segment_frames`-frame slices of the mel and the matching waveform;
  generator and discriminators alternate (D step on the detached fake, then G step with
  LS-GAN + feature matching + 45 x mel L1).
- **vits**: the model returns the decoded slice of z; the GAN part is the same as HiFi-GAN,
  plus the KL and duration losses.

Every run writes `last.pt`, `best.pt` (lowest validation mel loss), `curves.csv`, `curves.json`,
`config.json` and, when matplotlib is installed, `alignment_<step>.png` for the first
validation item, to `--output-dir`. `--resume` continues from `last.pt`; `--time-budget-hours`
saves and stops before a Kaggle/Colab session ends.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tts.audio import MelSpectrogram
from tts.config import Config, add_config_args, config_from_args
from tts.data import TTSDataset, make_loader, read_manifest, split_items
from tts.losses import discriminator_loss, feature_matching_loss, generator_loss
from tts.models import build_model, count_parameters, save_checkpoint
from tts.text import Tokenizer
from tts.vocoders import Discriminators

GAN_MODELS = ("hifigan", "vits")
MEL_WEIGHT = 45.0


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(step: int, base: float, warmup: int) -> float:
    """Linear warm-up, then 1/sqrt(step) decay (the Transformer schedule), floor at 5% of base."""
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    return max(base * math.sqrt(warmup / (step + 1)), 0.05 * base)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def plot_alignment(attn: torch.Tensor, path: Path, title: str = "") -> None:
    """attn (T, N) -> png; silently skipped without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(attn.detach().cpu().numpy().T, aspect="auto", origin="lower", interpolation="none")
    ax.set_xlabel("decoder frames")
    ax.set_ylabel("text symbols")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = pick_device(cfg.train.device)
        set_seed(cfg.train.seed)
        self.out = Path(cfg.train.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.tok = Tokenizer(cfg.text.tokenizer, cfg.text.lowercase)
        self.mel_fn = MelSpectrogram(cfg.audio)
        self.gan = cfg.model.model in GAN_MODELS
        items = read_manifest(cfg.train.data_dir)
        train_items, val_items = split_items(items, seed=cfg.train.seed)
        speaker_ids = {s: i for i, s in enumerate(sorted({it.speaker for it in items}))}
        if len(speaker_ids) > 1 and cfg.model.n_speakers < len(speaker_ids):
            cfg.model.n_speakers = len(speaker_ids)
        kw = {"need_linear": cfg.model.model == "vits", "need_wav": self.gan, "max_frames": cfg.train.max_frames}
        self.train_set = TTSDataset(train_items, cfg.audio, self.tok, cfg.train.data_dir, speaker_ids, **kw)
        self.val_set = TTSDataset(val_items, cfg.audio, self.tok, cfg.train.data_dir, speaker_ids, **kw)
        self.speaker_ids = speaker_ids
        self.train_loader = make_loader(self.train_set, cfg.train.batch_size, True, cfg.train.num_workers)
        self.val_loader = make_loader(self.val_set, cfg.train.batch_size, False, cfg.train.num_workers)
        self.model = build_model(cfg, self.tok.vocab_size).to(self.device)
        self.disc = Discriminators().to(self.device) if self.gan else None
        t = cfg.train
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=t.lr, betas=(0.8, 0.99) if self.gan else (0.9, 0.98),
                                     weight_decay=t.weight_decay)
        self.opt_d = torch.optim.AdamW(self.disc.parameters(), lr=t.lr, betas=(0.8, 0.99),
                                       weight_decay=t.weight_decay) if self.gan else None
        self.scaler = torch.amp.GradScaler(enabled=t.amp and self.device.type == "cuda")
        self.step, self.epoch, self.best = 0, 0, float("inf")
        self.curves: list[dict] = []
        if t.resume and (self.out / "last.pt").exists():
            self.load(self.out / "last.pt")
        cfg.save(self.out / "config.json")
        print(f"{cfg.model.model}: {count_parameters(self.model) / 1e6:.2f} M parameters | "
              f"{len(self.train_set)} train / {len(self.val_set)} val clips | device {self.device}")

    # -- checkpoints --------------------------------------------------------
    def save(self, name: str) -> None:
        extra = {"opt": self.opt.state_dict(), "best": self.best, "curves": self.curves}
        if self.gan:
            extra["disc"], extra["opt_d"] = self.disc.state_dict(), self.opt_d.state_dict()
        save_checkpoint(self.out / name, self.model, self.cfg, self.speaker_ids, self.step, self.epoch, extra)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        if self.gan and "disc" in state:
            self.disc.load_state_dict(state["disc"])
            self.opt_d.load_state_dict(state["opt_d"])
        self.step, self.epoch, self.best = state["step"], state["epoch"], state.get("best", float("inf"))
        self.curves = state.get("curves", [])
        print(f"resumed from {path} at step {self.step}")

    # -- one optimisation step ---------------------------------------------
    def gan_step(self, batch: dict) -> dict:
        seg = self.cfg.model.segment_frames
        hop = self.cfg.audio.hop_length
        if self.cfg.model.model == "vits":
            out = self.model(batch)
            wav_hat, starts = out["wav_hat"], out["starts"]
            losses = {"kl": out["kl"], "duration": out["duration"]}
        else:
            mel, lens = batch["mel"], batch["mel_lens"]
            starts = [int(torch.randint(0, max(int(lens[b]) - seg, 0) + 1, (1,))) for b in range(mel.size(0))]
            mel_slice = torch.stack([mel[b, :, s:s + seg] for b, s in enumerate(starts)])
            wav_hat = self.model(mel_slice)
            losses = {}
        wav = torch.stack([F.pad(batch["wav"][b, s * hop:(s + seg) * hop], (0, max(0, (s + seg) * hop - batch["wav"].size(1))))
                           for b, s in enumerate(starts)])
        wav_hat = wav_hat[:, : wav.size(1)]
        # discriminator
        self.opt_d.zero_grad(set_to_none=True)
        real_out, _ = self.disc(wav)
        fake_out, _ = self.disc(wav_hat.detach())
        l_d = discriminator_loss(real_out, fake_out)
        l_d.backward()
        torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.cfg.train.grad_clip)
        self.opt_d.step()
        # generator
        self.opt.zero_grad(set_to_none=True)
        real_out, real_feats = self.disc(wav)
        fake_out, fake_feats = self.disc(wav_hat)
        l_mel = F.l1_loss(self.mel_fn(wav_hat), self.mel_fn(wav))
        l_g = generator_loss(fake_out)
        l_fm = feature_matching_loss(real_feats, fake_feats)
        total = l_g + 2.0 * l_fm + MEL_WEIGHT * l_mel + sum(losses.values())
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.opt.step()
        return {"loss": total.item(), "disc": l_d.item(), "gen": l_g.item(), "fm": l_fm.item(), "mel": l_mel.item(),
                **{k: v.item() for k, v in losses.items()}}

    def mel_step(self, batch: dict) -> dict:
        self.opt.zero_grad(set_to_none=True)
        if hasattr(self.model, "bin_weight"):
            w = self.cfg.train.bin_warmup_steps
            self.model.bin_weight = min(1.0, max(0.0, (self.step - w) / max(w, 1)))
        with torch.autocast(self.device.type, enabled=self.scaler.is_enabled()):
            out = self.model(batch)
        self.scaler.scale(out["loss"]).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        return {k: v.item() for k, v in out.items() if torch.is_tensor(v) and v.dim() == 0}

    # -- validation ----------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        totals: dict[str, float] = {}
        n = 0
        attn = None
        for batch in self.val_loader:
            batch = to_device(batch, self.device)
            if self.gan:
                hop, seg = self.cfg.audio.hop_length, self.cfg.model.segment_frames
                if self.cfg.model.model == "vits":
                    out = self.model(batch)
                    wav_hat, starts = out["wav_hat"], out["starts"]
                    vals = {"kl": out["kl"].item(), "duration": out["duration"].item()}
                    attn = attn if attn is not None else out["attention"][0, : int(batch["mel_lens"][0]), : int(batch["text_lens"][0])]
                else:
                    starts = [0] * batch["mel"].size(0)
                    wav_hat = self.model(batch["mel"][:, :, :seg])
                    vals = {}
                wav = torch.stack([F.pad(batch["wav"][b, s * hop:(s + seg) * hop], (0, max(0, (s + seg) * hop - batch["wav"].size(1))))
                                   for b, s in enumerate(starts)])
                vals["mel"] = F.l1_loss(self.mel_fn(wav_hat[:, : wav.size(1)]), self.mel_fn(wav)).item()
                vals["loss"] = vals["mel"]
            else:
                out = self.model(batch)
                vals = {k: v.item() for k, v in out.items() if torch.is_tensor(v) and v.dim() == 0}
                vals["mel"] = vals.get("mel_after", vals["loss"])
                if attn is None and "attention" in out:  # first item, cropped to its own lengths
                    steps = out["attention"].size(1) * int(batch["mel_lens"][0]) // batch["mel"].size(2)
                    attn = out["attention"][0, :steps, : int(batch["text_lens"][0])]
            for k, v in vals.items():
                totals[k] = totals.get(k, 0.0) + v * batch["mel"].size(0)
            n += batch["mel"].size(0)
        self.model.train()
        result = {f"val_{k}": v / max(n, 1) for k, v in totals.items()}
        if attn is not None:
            plot_alignment(attn, self.out / f"alignment_{self.step:06d}.png", f"{self.cfg.model.model} step {self.step}")
        return result

    # -- main loop -------------------------------------------------------------
    def fit(self) -> dict:
        t = self.cfg.train
        start = time.time()
        self.model.train()
        done = False
        last = {}
        while not done:
            if self.epoch >= t.epochs and not t.max_steps:
                break
            for batch in self.train_loader:
                batch = to_device(batch, self.device)
                for g in self.opt.param_groups:
                    g["lr"] = lr_at(self.step, t.lr, t.warmup_steps)
                if self.opt_d is not None:
                    for g in self.opt_d.param_groups:
                        g["lr"] = lr_at(self.step, t.lr, t.warmup_steps)
                last = self.gan_step(batch) if self.gan else self.mel_step(batch)
                self.step += 1
                if self.step % t.log_every == 0:
                    msg = " ".join(f"{k} {v:.3f}" for k, v in last.items())
                    print(f"epoch {self.epoch} step {self.step} | {msg} | {time.time() - start:.0f}s", flush=True)
                    self.curves.append({"step": self.step, "epoch": self.epoch, **last})
                over_budget = t.time_budget_hours and (time.time() - start) / 3600 > t.time_budget_hours
                if (t.max_steps and self.step >= t.max_steps) or over_budget:
                    done = True
                    break
            self.epoch += 1
            if self.epoch % t.eval_every == 0 or done or self.epoch >= t.epochs:
                val = self.validate()
                print(f"epoch {self.epoch} | " + " ".join(f"{k} {v:.4f}" for k, v in val.items()), flush=True)
                self.curves.append({"step": self.step, "epoch": self.epoch, **val})
                if val["val_mel"] < self.best:
                    self.best = val["val_mel"]
                    self.save("best.pt")
                self.save("last.pt")
                self.write_curves()
        self.save("last.pt")
        self.write_curves()
        return {"step": self.step, "epoch": self.epoch, "best_val_mel": self.best, "seconds": time.time() - start,
                "last": last}

    def write_curves(self) -> None:
        keys = sorted({k for row in self.curves for k in row})
        with open(self.out / "curves.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.curves)
        (self.out / "curves.json").write_text(json.dumps(self.curves))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train a text-to-speech model or a HiFi-GAN vocoder.")
    p.add_argument("--config", default=None, help="config.json to start from (flags override it)")
    add_config_args(p)
    a = p.parse_args(argv)
    cfg = config_from_args(a, Config.load(a.config) if a.config else None)
    summary = Trainer(cfg).fit()
    print(json.dumps({k: v for k, v in summary.items() if k != "last"}))


if __name__ == "__main__":
    main()

__all__ = ["Trainer", "lr_at", "main", "pick_device", "plot_alignment", "set_seed", "to_device"]
