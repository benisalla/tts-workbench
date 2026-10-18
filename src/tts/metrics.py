"""Objective evaluation of synthesized speech, and `tts-eval`.

Numbers that say how close a synthesis is to the recording of the same sentence:

- **MCD** (mel-cepstral distortion, dB): mel-cepstra (DCT of the log-mel, coefficients
  1..12) of the two clips are aligned with **DTW** (the lengths differ), then
  `MCD = 10 / ln 10 * sqrt(2 * sum_k (c_k - c'_k)^2)` averaged along the path. Lower is
  better; good systems on LJSpeech reach ~ 5 dB, and ~ 8+ dB sounds clearly different.
- **log-F0 RMSE** (cents / semitones) over frames voiced in both clips, and the **voicing
  error** (fraction of frames where one is voiced and the other not), on the DTW path.
- **duration error**: |T_syn - T_ref| / T_ref.
- **teacher-forced mel L1**: the training loss on validation clips (needs no vocoder).
- **attention diagonality**: share of attention mass within 10% of the diagonal (AR
  models); 1.0 = perfectly monotonic, ~0.2 = not aligned.
- **WER** through an ASR model (Whisper, `pip install "tts-workbench[metrics]"`): does a
  listener understand the words? Compares the ASR transcript of the synthesis with the
  text (also reported for the recordings, as a ceiling).
- **predicted MOS** with UTMOS22 via `torch.hub` (`utmos`), a model that predicts the
  1..5 naturalness score of a listening test. Real MOS still needs humans (docs/evaluation.md).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tts.audio import MelSpectrogram, estimate_f0, save_wav
from tts.data import read_manifest, split_items
from tts.synthesize import Synthesizer
from tts.text import normalize


def mel_cepstrum(log_mel: torch.Tensor, n_coef: int = 13) -> np.ndarray:
    """(n_mels, T) log-mel -> (T, n_coef) cepstra via DCT-II (c0 = energy, dropped by MCD)."""
    x = log_mel.detach().cpu().numpy().T  # (T, n_mels)
    n = x.shape[1]
    k = np.arange(n_coef)[:, None]
    basis = np.cos(math.pi * k * (np.arange(n)[None, :] + 0.5) / n) * math.sqrt(2.0 / n)
    return x @ basis.T


def dtw_path(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Dynamic time warping on Euclidean frame distances; returns the list of (i, j) pairs."""
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    n, m = d.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = d[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return path[::-1]


def mcd(mel_syn: torch.Tensor, mel_ref: torch.Tensor) -> tuple[float, list[tuple[int, int]]]:
    """Mel-cepstral distortion in dB between two log-mels (n_mels, T), DTW-aligned. Also returns the path."""
    cs, cr = mel_cepstrum(mel_syn)[:, 1:], mel_cepstrum(mel_ref)[:, 1:]
    path = dtw_path(cs, cr)
    diff = np.array([cs[i] - cr[j] for i, j in path])
    return float((10.0 / math.log(10)) * np.mean(np.sqrt(2.0 * (diff**2).sum(-1)))), path


def f0_metrics(f0_syn: np.ndarray, f0_ref: np.ndarray, path: list[tuple[int, int]]) -> dict:
    """log-F0 RMSE in semitones over co-voiced frames + voicing decision error, along a DTW path."""
    s = np.array([f0_syn[min(i, len(f0_syn) - 1)] for i, _ in path])
    r = np.array([f0_ref[min(j, len(f0_ref) - 1)] for _, j in path])
    both = (s > 0) & (r > 0)
    vde = float(((s > 0) != (r > 0)).mean())
    if both.sum() == 0:
        return {"f0_rmse_semitones": float("nan"), "voicing_error": vde}
    semitones = 12 * np.log2(s[both] / r[both])
    return {"f0_rmse_semitones": float(np.sqrt((semitones**2).mean())), "voicing_error": vde}


def attention_diagonality(attn: torch.Tensor, band: float = 0.1) -> float:
    """attn (T, N) -> share of attention mass within `band` of the diagonal."""
    T, N = attn.shape
    t = torch.arange(T, device=attn.device)[:, None].float() / max(T - 1, 1)
    n = torch.arange(N, device=attn.device)[None, :].float() / max(N - 1, 1)
    near = ((t - n).abs() <= band).float()
    return float((attn * near).sum() / attn.sum().clamp_min(1e-8))


def word_error_rate(hyp: str, ref: str) -> float:
    h, r = normalize(hyp).replace(".", "").replace(",", "").split(), normalize(ref).replace(".", "").replace(",", "").split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(len(r) + 1), np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[-1, -1] / max(len(r), 1))


class WhisperASR:
    """Lazy Whisper (transformers) for intelligibility; downloads ~ 300 MB the first time."""

    def __init__(self, name: str = "openai/whisper-base", device: str = "cpu"):
        from transformers import pipeline

        self.pipe = pipeline("automatic-speech-recognition", model=name, device=device)

    def __call__(self, wav: np.ndarray, sample_rate: int) -> str:
        return self.pipe({"raw": wav.astype(np.float32), "sampling_rate": sample_rate})["text"]


class UTMOS:
    """Predicted MOS (1..5) with UTMOS22-strong via torch.hub (needs internet once)."""

    def __init__(self, device: str = "cpu"):
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, wav: np.ndarray, sample_rate: int) -> float:
        return float(self.model(torch.from_numpy(wav.astype(np.float32))[None].to(self.device), sample_rate)[0])


def evaluate(checkpoint: str, data_dir: str, vocoder: str = "griffinlim", vocoder_checkpoint: str | None = None,
             n: int = 10, out_dir: str | None = None, whisper: bool = False, utmos: bool = False,
             device: str = "cpu") -> dict:
    """Synthesize `n` validation sentences and compare them with the recordings."""
    syn = Synthesizer(checkpoint, vocoder, vocoder_checkpoint, device)
    cfg = syn.cfg
    mel_fn = MelSpectrogram(cfg.audio)
    _, val_items = split_items(read_manifest(data_dir), seed=cfg.train.seed)
    asr = WhisperASR(device=device) if whisper else None
    mos = UTMOS(device) if utmos else None
    rows = []
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    from tts.audio import load_wav

    for item in val_items[:n]:
        ref = load_wav(str(item.audio), cfg.audio.sample_rate)
        wav, attn = syn(item.text, item.speaker if cfg.model.n_speakers > 1 else None)
        wav_np = wav.cpu().numpy()
        mel_syn, mel_ref = mel_fn(wav[None].cpu())[0], mel_fn(torch.from_numpy(ref)[None])[0]
        score, path = mcd(mel_syn, mel_ref)
        row = {"id": item.id, "text": item.text, "mcd_db": score,
               "duration_error": abs(len(wav_np) - len(ref)) / len(ref),
               **f0_metrics(estimate_f0(wav_np, cfg.audio), estimate_f0(ref, cfg.audio), path)}
        if attn is not None and attn.dim() == 2:
            row["attention_diagonality"] = attention_diagonality(attn)
        if asr:
            row["wer"] = word_error_rate(asr(wav_np, cfg.audio.sample_rate), item.text)
            row["wer_recording"] = word_error_rate(asr(ref, cfg.audio.sample_rate), item.text)
        if mos:
            row["utmos"] = mos(wav_np, cfg.audio.sample_rate)
            row["utmos_recording"] = mos(ref, cfg.audio.sample_rate)
        rows.append(row)
        if out:
            save_wav(str(out / f"{item.id}_syn.wav"), wav_np, cfg.audio.sample_rate)
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    summary["n"] = len(rows)
    summary["model"] = cfg.model.model
    summary["vocoder"] = "none (end-to-end)" if cfg.model.model == "vits" else vocoder
    result = {"summary": summary, "items": rows}
    if out:
        (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def export_teacher_durations(checkpoint: str, data_dir: str, device: str = "cpu") -> int:
    """Teacher-forced Tacotron 2 attention -> integer durations per text symbol, saved to <data_dir>/durations/."""
    from tts.align import durations_from_attention
    from tts.data import TTSDataset, collate
    from tts.train import to_device

    syn = Synthesizer(checkpoint, device=device)
    cfg = syn.cfg
    items = read_manifest(data_dir)
    ds = TTSDataset(items, cfg.audio, syn.tok, data_dir)
    out = Path(data_dir) / "durations"
    out.mkdir(exist_ok=True)
    r = cfg.model.reduction_factor
    with torch.no_grad():
        for i in range(len(ds)):
            batch = to_device(collate([ds[i]]), device)
            attn = syn.model(batch)["attention"][0]  # (steps, N)
            attn = torch.repeat_interleave(attn, r, dim=0)[: int(batch["mel_lens"][0])]
            dur = durations_from_attention(attn, int(batch["text_lens"][0]), int(batch["mel_lens"][0]))
            torch.save(dur, out / f"{ds.items[i].id}.pt")
    return len(ds)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on validation sentences.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--vocoder", default="griffinlim", choices=["griffinlim", "hifigan"])
    p.add_argument("--vocoder-checkpoint", default=None)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out", default=None, help="folder for synthesized wavs + metrics.json")
    p.add_argument("--whisper", action="store_true", help="WER through Whisper (transformers)")
    p.add_argument("--utmos", action="store_true", help="predicted MOS through UTMOS22 (torch.hub)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--export-durations", action="store_true",
                   help="instead of evaluating: write Tacotron 2 teacher durations for fastspeech2 --duration-source teacher")
    a = p.parse_args(argv)
    if a.export_durations:
        print(f"wrote durations for {export_teacher_durations(a.checkpoint, a.data_dir, a.device)} clips")
        return
    result = evaluate(a.checkpoint, a.data_dir, a.vocoder, a.vocoder_checkpoint, a.n, a.out, a.whisper, a.utmos,
                      a.device)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

__all__ = [
    "UTMOS",
    "WhisperASR",
    "attention_diagonality",
    "dtw_path",
    "evaluate",
    "export_teacher_durations",
    "f0_metrics",
    "main",
    "mcd",
    "mel_cepstrum",
    "word_error_rate",
]
