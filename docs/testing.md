# Testing

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"
OMP_NUM_THREADS=1 .venv/bin/pytest -q        # ~ 25 s on a laptop CPU
.venv/bin/ruff check src tests notebooks
python notebooks/build_notebook.py . notebooks/train_colab_kaggle.ipynb   # after changing src/
```

| file | what it checks |
|---|---|
| `tests/test_text.py` | number spelling, normalisation, char tokenizer round-trip, phoneme symbol table |
| `tests/test_audio.py` | mel shapes and frame count, filterbank coverage, Griffin-Lim recovers a 300 Hz tone, F0 tracks pitch and silence |
| `tests/test_align.py` | MAS is monotonic/complete and follows a clear diagonal, the aligner's forward-sum loss decreases and its durations sum to the frame count, teacher durations from attention |
| `tests/test_models.py` | every model trains one step and runs inference on a micro config; FastSpeech 2 with the aligner and with teacher durations; HiFi-GAN shapes and GAN losses; VITS flow invertibility; multi-speaker registry |
| `tests/test_data.py` | the toy language is deterministic, manifest/split/batch shapes, feature cache, durations in a batch |
| `tests/test_train.py` | one epoch of each model end to end on 6 toy clips with checkpoints and curves, resume, synthesis with both vocoders, evaluation, teacher duration export, CLI config round-trip |

The tests use micro models (hidden 32) and a 6-clip toy dataset, so they run in a minute on one
CPU thread. They check that the code runs and behaves (shapes, invertibility, losses going
down), not that the models sound good; the short runs described in [training.md](training.md)
and the README are the proof of learning.
