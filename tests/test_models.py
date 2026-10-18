import pytest
import torch

from tts.fastspeech2 import FastSpeech2, average_by_duration
from tts.losses import discriminator_loss, feature_matching_loss, generator_loss
from tts.models import build_model, count_parameters
from tts.tacotron2 import Tacotron2
from tts.transformer_tts import TransformerTTS
from tts.vits import VITS
from tts.vocoders import Discriminators, HiFiGANGenerator

VOCAB = 34


@pytest.mark.parametrize("cls", [Tacotron2, TransformerTTS])
def test_autoregressive_models_train_and_infer(micro_cfg, batch, cls):
    model = cls(micro_cfg, VOCAB)
    out = model(batch)
    assert out["mel"].shape == batch["mel"].shape
    assert out["attention"].shape == (2, 11, 7)  # ceil(21 / r=2) decoder steps
    for key in ("mel_before", "mel_after", "stop", "guided_attention"):
        assert out[key].item() >= 0
    out["loss"].backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad and "speaker" not in n)
    model.eval()
    mel, attn = model.infer(batch["text"][:1], max_steps=4)
    assert mel.shape[:2] == (1, 64) and mel.shape[2] % micro_cfg.model.reduction_factor == 0
    assert attn.shape[1] == 7


def test_fastspeech2_with_aligner(micro_cfg, batch):
    model = FastSpeech2(micro_cfg, VOCAB)
    out = model(batch)
    assert out["mel"].shape == batch["mel"].shape
    assert (out["durations"].sum(1) == batch["mel_lens"]).all()
    assert out["attention"].shape == (2, 21, 7)
    out["loss"].backward()
    model.eval()
    mel, attn = model.infer(batch["text"][:1], duration_scale=2.0)
    assert mel.shape[2] == attn.shape[0] == int(attn.sum())


def test_fastspeech2_with_teacher_durations(micro_cfg, batch):
    micro_cfg.train.duration_source = "teacher"
    try:
        model = FastSpeech2(micro_cfg, VOCAB)
        assert model.aligner is None
        batch["durations"] = torch.tensor([[3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 0, 0]])
        out = model(batch)
        assert "forward_sum" not in out and out["loss"].item() > 0
    finally:
        micro_cfg.train.duration_source = "aligner"


def test_average_by_duration():
    values = torch.tensor([[1.0, 3.0, 0.0, 5.0]])
    assert average_by_duration(values, torch.tensor([[2, 2]])).tolist() == [[2.0, 5.0]]  # zeros (unvoiced) skipped


def test_hifigan_shapes_and_gan_losses(micro_cfg):
    gen, disc = HiFiGANGenerator(micro_cfg), Discriminators()
    wav = gen(torch.randn(2, 64, 8))
    assert wav.shape == (2, 8 * micro_cfg.audio.hop_length) and wav.abs().max() <= 1
    real_out, real_feats = disc(torch.rand(2, wav.shape[1]) - 0.5)
    fake_out, fake_feats = disc(wav)
    assert len(real_out) == 8  # 5 periods + 3 scales
    assert discriminator_loss(real_out, fake_out).item() > 0
    assert generator_loss(fake_out).item() > 0
    feature_matching_loss(real_feats, fake_feats).backward()


def test_vits_end_to_end(micro_cfg, batch):
    model = VITS(micro_cfg, VOCAB)
    out = model(batch)
    assert out["wav_hat"].shape == (2, micro_cfg.model.segment_frames * micro_cfg.audio.hop_length)
    (out["kl"] + out["duration"]).backward()
    z = torch.randn(1, micro_cfg.model.latent_dim, 10)
    mask = torch.ones(1, 1, 10)
    assert torch.allclose(model.flow(model.flow(z, mask), mask, reverse=True), z, atol=1e-5)
    model.eval()
    wav, attn = model.infer(batch["text"][:1])
    assert wav.dim() == 1 and wav.numel() == attn.shape[0] * micro_cfg.audio.hop_length


def test_registry_and_multispeaker(micro_cfg):
    micro_cfg.model.n_speakers = 3
    try:
        for name in ("tacotron2", "transformer_tts", "fastspeech2", "vits", "hifigan"):
            micro_cfg.model.model = name
            model = build_model(micro_cfg, VOCAB)
            assert count_parameters(model) < 2_000_000
            if name != "hifigan":
                assert model.speaker.enabled
    finally:
        micro_cfg.model.n_speakers = 1
        micro_cfg.model.model = "tacotron2"
