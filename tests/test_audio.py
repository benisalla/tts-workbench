import numpy as np
import torch

from tts.audio import MelSpectrogram, estimate_f0, frame_energy, griffin_lim, mel_filterbank
from tts.config import AudioConfig


def tone(freq, seconds=1.0, sr=8000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_mel_shapes_and_frames():
    cfg = AudioConfig()
    mel_fn = MelSpectrogram(cfg)
    wav = torch.from_numpy(tone(220))[None]
    mel = mel_fn(wav)
    assert mel.shape == (1, cfg.n_mels, mel_fn.frames(wav.shape[1]))
    assert mel_fn.linear(wav).shape[1] == cfg.n_fft // 2 + 1
    assert frame_energy(mel[0]).shape == (mel.shape[2],)


def test_filterbank_covers_band():
    fb = mel_filterbank(8000, 512, 64, 0.0, 4000.0)
    assert fb.shape == (64, 257) and (fb.sum(1) > 0).all()


def test_griffin_lim_recovers_tone():
    cfg = AudioConfig()
    mel = MelSpectrogram(cfg)(torch.from_numpy(tone(300))[None])[0]
    wav = griffin_lim(mel, cfg, n_iter=16)
    f0 = estimate_f0(wav.numpy(), cfg)
    voiced = f0[f0 > 0]
    assert len(voiced) > 0.8 * len(f0)
    assert abs(np.median(voiced) - 300) < 10


def test_f0_tracks_pitch_and_silence():
    cfg = AudioConfig()
    wav = np.concatenate([tone(150, 0.5), np.zeros(4000, dtype=np.float32)])
    f0 = estimate_f0(wav, cfg)
    assert abs(np.median(f0[2:28]) - 150) < 5
    assert (f0[40:] == 0).all()
