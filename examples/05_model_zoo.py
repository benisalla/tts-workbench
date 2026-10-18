"""Build every model, run one training step and one inference on a random batch.

    python examples/05_model_zoo.py

No data or checkpoint needed. Shows the shared interface: `model(batch)` returns a dict with
"loss" and its terms (different per model), `model.infer(text_ids)` returns a mel (or a
waveform for VITS) and an alignment, and the HiFi-GAN generator maps a mel to a waveform.
"""

import torch

from tts.config import Config
from tts.models import build_model, count_parameters
from tts.vocoders import HiFiGANGenerator

torch.manual_seed(0)
VOCAB = 34

cfg = Config()  # the micro sizes of the test suite
cfg.model.hidden, cfg.model.prenet_dim, cfg.model.postnet_channels, cfg.model.attention_dim = 32, 16, 16, 16
cfg.model.upsample_initial, cfg.model.latent_dim, cfg.model.segment_frames = 32, 16, 8

# Two utterances: 7 and 5 symbols, 21 and 15 mel frames; every model reads the keys it needs.
batch = {"text": torch.randint(2, VOCAB, (2, 7)), "text_lens": torch.tensor([7, 5]),
         "mel": torch.randn(2, cfg.audio.n_mels, 21) - 5, "mel_lens": torch.tensor([21, 15]),
         "f0": torch.rand(2, 21) * 200, "energy": torch.randn(2, 21),
         "linear": torch.rand(2, cfg.audio.n_fft // 2 + 1, 21), "wav": torch.rand(2, 21 * cfg.audio.hop_length) - 0.5}

print("model            params   loss terms")
for name in ("tacotron2", "transformer_tts", "fastspeech2", "vits"):
    cfg.model.model = name
    model = build_model(cfg, VOCAB)
    out = model(batch)
    out["loss"].backward() if "loss" in out else (out["kl"] + out["duration"]).backward()
    terms = ", ".join(f"{k} {float(v.detach()):.2f}" for k, v in out.items() if isinstance(v, torch.Tensor) and v.dim() == 0)
    print(f"{name:16s} {count_parameters(model) / 1e3:6.0f} k  {terms}")
    model.eval()
    with torch.no_grad():
        kw = {"max_steps": 6} if name in ("tacotron2", "transformer_tts") else {}
        y, attn = model.infer(batch["text"][:1], **kw)
    what = "waveform" if name == "vits" else "mel"
    print(f"{'':16s} infer -> {what} {tuple(y.shape)}, alignment {tuple(attn.shape)}")

cfg.model.model = "hifigan"
gen = HiFiGANGenerator(cfg)
wav = gen(torch.randn(1, cfg.audio.n_mels, 8))
print(f"{'hifigan':16s} {count_parameters(gen) / 1e3:6.0f} k  mel (1, {cfg.audio.n_mels}, 8) -> waveform {tuple(wav.shape)}"
      f" = 8 frames x hop {cfg.audio.hop_length}")
