# Inference

```bash
tts-synthesize --checkpoint checkpoints/tacotron2/best.pt --text "hello world" --out synthesized/
tts-synthesize --checkpoint checkpoints/fs2/best.pt --text-file sentences.txt \
               --vocoder hifigan --vocoder-checkpoint checkpoints/hifigan/best.pt
tts-synthesize --checkpoint checkpoints/fs2/best.pt --text "slowly" --duration-scale 1.5
tts-synthesize --checkpoint checkpoints/vits/best.pt --text "end to end"      # no vocoder flags
```

Writes `<out>/<i>.wav` and a `manifest.jsonl` (which `tts-eval`-style tools or the sibling
speech-to-text repo can read).

## From Python

```python
from tts.synthesize import Synthesizer

syn = Synthesizer("checkpoints/fs2/best.pt", vocoder="hifigan", vocoder_checkpoint="checkpoints/hifigan/best.pt")
wav, alignment = syn("Text to speech.")          # wav: torch tensor (samples,), alignment: (frames, symbols)
mel, alignment = syn.mel("Text to speech.")      # the log-mel of a mel model, (n_mels, frames)
```

## What inference does, per model

| model | steps |
|---|---|
| tacotron2 | encode text; loop: prenet(prev frame) -> attention -> LSTMs -> r frames + stop; stop when `sigmoid(stop) > 0.5` or `max_decoder_steps`; postnet |
| transformer_tts | encode; loop: decoder on all frames so far; same stop rule |
| fastspeech2 | encode; predict log-durations, pitch, energy; `round(exp(log d) - 1) * duration_scale`; length-regulate; decode once |
| vits | encode; durations; expand `mu_p, logs_p`; sample `z_p` (`temperature`, default 0.667); inverse flow; HiFi-GAN decoder |

A checkpoint is self-describing (config, tokenizer, speaker map). Multi-speaker models take
`--speaker <name>` as it appears in `speakers.csv`.

## Vocoder choice

Griffin-Lim needs nothing and is what the tests use; its output is intelligible but buzzy.
A HiFi-GAN trained on the same data (`tts-train --model hifigan`) is a drop-in replacement
through `--vocoder hifigan`; its sample rate and mel settings must match the acoustic model
(checked at load time).
