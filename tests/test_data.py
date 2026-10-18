import torch

from tts.config import AudioConfig
from tts.data import TTSDataset, collate, read_manifest, split_items
from tts.datasets import DIGITS, synth_toy_wave, toy_sentence
from tts.text import Tokenizer


def test_toy_language_is_deterministic():
    a, b = synth_toy_wave("ba ko", seed=1), synth_toy_wave("ba ko", seed=1)
    assert (a == b).all() and len(a) > 8000 * 0.3
    assert synth_toy_wave("ba", seed=1).shape != synth_toy_wave("bam", seed=1).shape


def test_manifest_split_and_batch(toy_dir):
    items = read_manifest(toy_dir)
    assert len(items) == 6 and items[0].audio.exists()
    train, val = split_items(items)
    assert len(train) + len(val) == 6 and len(val) == 2
    ds = TTSDataset(train, AudioConfig(), Tokenizer(), toy_dir, need_linear=True)
    batch = collate([ds[0], ds[1]])
    assert batch["mel"].shape[1] == 64 and batch["linear"].shape[1] == 257
    assert batch["text"].shape[0] == 2 and batch["f0"].shape == batch["energy"].shape == batch["mel"].shape[::2]
    assert (batch["mel_lens"] <= batch["mel"].shape[2]).all()
    assert (ds.cache / f"{train[0].id}_8000_512_128_64.pt").exists()  # cached features


def test_toy_vocabulary():
    import random

    assert len(DIGITS) == 10
    assert all(w.isalpha() for w in toy_sentence(random.Random(0)).split())


def test_collate_with_durations(toy_dir):
    items = read_manifest(toy_dir)
    ds = TTSDataset(items[:2], AudioConfig(), Tokenizer(), toy_dir)
    samples = [ds[0], ds[1]]
    for s in samples:
        n, t = s["text"].numel(), s["mel"].shape[1]
        s["durations"] = torch.full((n,), t // n)
    batch = collate(samples)
    assert batch["durations"].shape == batch["text"].shape
