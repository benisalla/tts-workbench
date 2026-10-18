import numpy as np
import torch

from tts.align import (
    Aligner,
    durations_from_alignment,
    durations_from_attention,
    forward_sum_loss,
    mas,
    mas_batch,
)


def test_mas_is_monotonic_and_complete():
    rng = np.random.default_rng(0)
    N, T = 5, 20
    logp = rng.normal(size=(N, T))
    path = mas(logp, N, T)
    assert path.sum() == T and (path.sum(0) == 1).all() and (path.sum(1) >= 1).all()
    assigned = path.argmax(0)
    assert (np.diff(assigned) >= 0).all() and assigned[0] == 0 and assigned[-1] == N - 1


def test_mas_follows_clear_diagonal():
    N, T = 3, 6
    truth = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1]])
    logp = np.log(truth + 1e-3)
    assert (mas(logp, N, T) == truth).all()


def test_aligner_forward_sum_decreases():
    torch.manual_seed(0)
    aligner = Aligner(8, 64, dim=16)
    text, mel = torch.randn(2, 4, 8), torch.randn(2, 64, 12)
    mask = torch.ones(2, 4, dtype=torch.bool)
    tl, ml = torch.tensor([4, 4]), torch.tensor([12, 10])
    opt = torch.optim.Adam(aligner.parameters(), 1e-2)
    first = None
    for _ in range(15):
        loss = forward_sum_loss(aligner(text, mel, mask), tl, ml)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first
    hard = mas_batch(aligner(text, mel, mask), tl, ml)
    assert (durations_from_alignment(hard).sum(1) == ml).all()


def test_teacher_durations_from_attention():
    attn = torch.zeros(10, 3)
    attn[:4, 0], attn[4:7, 1], attn[7:, 2] = 1, 1, 1
    assert durations_from_attention(attn, 3, 10).tolist() == [4, 3, 3]
