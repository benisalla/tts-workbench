import json

import pytest

from tts.config import Config, add_config_args, config_from_args
from tts.metrics import (
    attention_diagonality,
    dtw_path,
    evaluate,
    export_teacher_durations,
    mcd,
    word_error_rate,
)
from tts.models import load_checkpoint
from tts.synthesize import Synthesizer
from tts.train import Trainer, lr_at


def run(micro_cfg, toy_dir, tmp_path, model, **overrides):
    cfg = Config.from_dict(micro_cfg.to_dict())
    cfg.model.model = model
    cfg.train.data_dir = str(toy_dir)
    cfg.train.output_dir = str(tmp_path / model)
    for k, v in overrides.items():
        setattr(cfg.train, k, v)
    return cfg, Trainer(cfg).fit()


@pytest.mark.parametrize("model", ["tacotron2", "transformer_tts", "fastspeech2", "hifigan", "vits"])
def test_one_epoch_each_model(micro_cfg, toy_dir, tmp_path, model):
    _cfg, summary = run(micro_cfg, toy_dir, tmp_path, model)
    assert summary["step"] >= 1 and summary["best_val_mel"] < float("inf")
    out = tmp_path / model
    assert (out / "last.pt").exists() and (out / "best.pt").exists() and (out / "config.json").exists()
    assert json.loads((out / "curves.json").read_text())
    _loaded, cfg2, _tok, state = load_checkpoint(out / "best.pt")
    assert cfg2.model.model == model and state["step"] == summary["step"]


def test_resume_continues_step_count(micro_cfg, toy_dir, tmp_path):
    cfg, first = run(micro_cfg, toy_dir, tmp_path, "fastspeech2")
    cfg.train.resume, cfg.train.epochs = True, 2
    second = Trainer(cfg).fit()
    assert second["step"] > first["step"]


def test_synthesize_and_evaluate(micro_cfg, toy_dir, tmp_path):
    run(micro_cfg, toy_dir, tmp_path, "fastspeech2")
    run(micro_cfg, toy_dir, tmp_path, "hifigan")
    ckpt = tmp_path / "fastspeech2" / "best.pt"
    syn = Synthesizer(ckpt, griffin_lim_iters=4)
    wav, attn = syn("ba ko", duration_scale=1.0)
    assert wav.dim() == 1 and wav.numel() > 0
    syn_h = Synthesizer(ckpt, "hifigan", tmp_path / "hifigan" / "best.pt")
    wav_h, _ = syn_h("ba ko")
    assert wav_h.numel() == attn.shape[0] * micro_cfg.audio.hop_length
    result = evaluate(str(ckpt), str(toy_dir), n=2, out_dir=str(tmp_path / "eval"))
    assert result["summary"]["n"] == 2 and result["summary"]["mcd_db"] > 0
    assert (tmp_path / "eval" / "metrics.json").exists()


def test_teacher_durations_export(micro_cfg, toy_dir, tmp_path):
    run(micro_cfg, toy_dir, tmp_path, "tacotron2")
    n = export_teacher_durations(str(tmp_path / "tacotron2" / "best.pt"), str(toy_dir))
    assert n == 6 and (toy_dir / "durations").exists()
    _cfg, summary = run(micro_cfg, toy_dir, tmp_path, "fastspeech2", duration_source="teacher")
    assert summary["step"] >= 1


def test_metrics_helpers():
    import torch

    a = torch.randn(64, 10) - 5
    score, path = mcd(a, a)
    assert score < 1e-6 and path[0] == (0, 0) and path[-1] == (9, 9)
    assert len(dtw_path(a.T.numpy(), a.T.numpy()[:5])) >= 10
    assert attention_diagonality(torch.eye(8)) == 1.0
    assert attention_diagonality(torch.ones(8, 8) / 8) < 0.5
    assert word_error_rate("the cat sat", "the cat sat") == 0 and word_error_rate("cat", "the cat sat") == 2 / 3


def test_config_cli_roundtrip(tmp_path):
    import argparse

    p = argparse.ArgumentParser()
    add_config_args(p)
    cfg = config_from_args(p.parse_args(["--model", "vits", "--hidden", "16", "--upsample-rates", "8,4,4", "--amp", "1"]))
    assert cfg.model.model == "vits" and cfg.model.hidden == 16 and cfg.model.upsample_rates == (8, 4, 4) and cfg.train.amp
    cfg.save(tmp_path / "c.json")
    assert Config.load(tmp_path / "c.json").model.upsample_rates == (8, 4, 4)
    assert lr_at(0, 1.0, 10) < lr_at(9, 1.0, 10) and lr_at(1000, 1.0, 10) >= 0.05
