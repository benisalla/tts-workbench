import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest
import torch

from tts.config import Config
from tts.datasets import write_synthetic

torch.set_num_threads(1)


@pytest.fixture(scope="session")
def micro_cfg() -> Config:
    """A model small enough for every test to run in seconds on a CPU."""
    cfg = Config()
    cfg.model.hidden = 32
    cfg.model.prenet_dim = 16
    cfg.model.postnet_channels = 16
    cfg.model.attention_dim = 16
    cfg.model.upsample_initial = 32
    cfg.model.latent_dim = 16
    cfg.model.segment_frames = 8
    cfg.train.batch_size = 2
    cfg.train.epochs = 1
    cfg.train.log_every = 1
    return cfg


@pytest.fixture(scope="session")
def toy_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("toy")
    write_synthetic(out, n=6)
    return out


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return {"text": torch.randint(2, 30, (2, 7)), "text_lens": torch.tensor([7, 5]), "mel": torch.randn(2, 64, 21) - 5,
            "mel_lens": torch.tensor([21, 15]), "f0": torch.rand(2, 21) * 200, "energy": torch.randn(2, 21),
            "linear": torch.rand(2, 257, 21), "wav": torch.rand(2, 21 * 128) - 0.5}
