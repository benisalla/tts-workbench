"""Audio: loading, mel-spectrograms, Griffin-Lim, F0 and energy.

The *mel-spectrogram* is the intermediate representation of every acoustic model here:
`|STFT|` (linear magnitude, n_fft/2+1 bins) -> mel filterbank (n_mels bins on a
perceptual frequency scale) -> log. Shapes are `(batch, n_mels, frames)` with
`frames = samples // hop_length` (the signal is padded by (n_fft - hop) / 2 on each side).

Griffin-Lim turns a magnitude spectrogram back into a waveform by alternating between
"take the STFT phase of the current guess" and "keep the target magnitude"; it needs no
training, which makes it the baseline vocoder (and sounds metallic, which is why
neural vocoders exist).

F0 is estimated with a normalised-autocorrelation (YIN-like) method per frame; FastSpeech 2
predicts it and the metrics compare it between synthesis and recording.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from tts.config import AudioConfig


def load_wav(path: str, sample_rate: int) -> np.ndarray:
    """Mono float32 waveform at `sample_rate` (resampled with a polyphase filter if needed)."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != sample_rate:
        g = math.gcd(sr, sample_rate)
        wav = resample_poly(wav, sample_rate // g, sr // g).astype(np.float32)
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = wav / peak
    return wav


def save_wav(path: str, wav: np.ndarray | torch.Tensor, sample_rate: int) -> None:
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    sf.write(path, wav, sample_rate)


def hz_to_mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> torch.Tensor:
    """Triangular (Slaney-style) filters, shape (n_mels, n_fft // 2 + 1)."""
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2, n_bins)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = np.array([mel_to_hz(m) for m in mels])
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - lo) / max(mid - lo, 1e-6)
        down = (hi - freqs) / max(hi - mid, 1e-6)
        fb[i] = np.clip(np.minimum(up, down), 0, None)
    enorm = 2.0 / (hz[2:] - hz[:-2])  # area normalisation so every filter has the same energy
    fb *= enorm[:, None]
    return torch.from_numpy(fb)


class MelSpectrogram:
    """Waveform (batch, samples) -> log-mel (batch, n_mels, frames); also linear magnitude for VITS."""

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.window = torch.hann_window(cfg.win_length)
        self.fb = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)

    def linear(self, wav: torch.Tensor) -> torch.Tensor:
        """Magnitude spectrogram (batch, n_fft // 2 + 1, frames)."""
        c = self.cfg
        pad = (c.n_fft - c.hop_length) // 2
        wav = torch.nn.functional.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(wav, c.n_fft, c.hop_length, c.win_length, self.window.to(wav.device), center=False,
                          return_complex=True)
        return spec.abs()

    def mel_from_linear(self, mag: torch.Tensor) -> torch.Tensor:
        mel = torch.matmul(self.fb.to(mag.device), mag)
        return torch.log(torch.clamp(mel, min=self.cfg.log_floor))

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        return self.mel_from_linear(self.linear(wav))

    def frames(self, n_samples: int) -> int:
        return n_samples // self.cfg.hop_length


def griffin_lim(log_mel: torch.Tensor, cfg: AudioConfig, n_iter: int = 32, power: float = 1.2) -> torch.Tensor:
    """Log-mel (n_mels, frames) -> waveform (samples). Mel is inverted with the pseudo-inverse
    of the filterbank, then the phase is recovered by Griffin-Lim iterations."""
    fb = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax).to(log_mel.device)
    mag = torch.clamp(torch.linalg.pinv(fb) @ torch.exp(log_mel), min=0.0) ** power
    window = torch.hann_window(cfg.win_length, device=log_mel.device)
    angles = torch.exp(2j * math.pi * torch.rand(mag.shape, device=log_mel.device))
    spec = mag * angles
    for _ in range(n_iter):
        wav = torch.istft(spec, cfg.n_fft, cfg.hop_length, cfg.win_length, window)
        rebuilt = torch.stft(wav, cfg.n_fft, cfg.hop_length, cfg.win_length, window, return_complex=True)
        spec = mag * rebuilt / torch.clamp(rebuilt.abs(), min=1e-8)
    wav = torch.istft(spec, cfg.n_fft, cfg.hop_length, cfg.win_length, window)
    return wav / torch.clamp(wav.abs().max(), min=1e-5) * 0.95


def estimate_f0(wav: np.ndarray, cfg: AudioConfig, threshold: float = 0.3) -> np.ndarray:
    """Per-frame fundamental frequency in Hz (0 = unvoiced), aligned with the mel frames.

    For each frame the normalised autocorrelation is computed over lags between
    sample_rate/f0_max and sample_rate/f0_min; the best lag above `threshold` gives the pitch.
    """
    n_frames = len(wav) // cfg.hop_length
    win = cfg.win_length
    half = win // 2
    padded = np.pad(wav, (half, half + win), mode="constant")
    lag_min = max(2, int(cfg.sample_rate / cfg.f0_max))
    lag_max = min(win // 2, int(cfg.sample_rate / cfg.f0_min))
    f0 = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * cfg.hop_length
        frame = padded[start:start + win]
        frame = frame - frame.mean()
        energy = np.dot(frame, frame)
        if energy < 1e-6:
            continue
        # normalised autocorrelation for lags [lag_min, lag_max]
        spec = np.fft.rfft(frame, 2 * win)
        ac = np.fft.irfft(spec * np.conj(spec))[:win]
        ac = ac / (energy + 1e-9)
        seg = ac[lag_min:lag_max + 1]
        if seg.size == 0:
            continue
        k = int(np.argmax(seg))
        if seg[k] > threshold:
            lag = lag_min + k
            # parabolic interpolation for sub-sample precision
            if 0 < k < seg.size - 1:
                a, b, c = seg[k - 1], seg[k], seg[k + 1]
                denom = a - 2 * b + c
                if abs(denom) > 1e-9:
                    lag = lag + 0.5 * (a - c) / denom
            f0[i] = cfg.sample_rate / lag
    return f0


def frame_energy(mel: torch.Tensor) -> torch.Tensor:
    """Per-frame energy from a log-mel (n_mels, frames): log of the summed linear mel."""
    return torch.log(torch.clamp(torch.exp(mel).sum(dim=0), min=1e-5))


__all__ = [
    "MelSpectrogram",
    "estimate_f0",
    "frame_energy",
    "griffin_lim",
    "hz_to_mel",
    "load_wav",
    "mel_filterbank",
    "mel_to_hz",
    "save_wav",
]
