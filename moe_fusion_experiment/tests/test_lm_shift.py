import torch

from conftest import tiny_model_cfg
from moefusion.model import IGNORE_INDEX, build_model, lm_loss, make_inputs_targets


def test_shift_with_plus_one_rows():
    rows = torch.arange(2 * 17).view(2, 17)
    x, y = make_inputs_targets(rows, 16)
    assert x.shape == y.shape == (2, 16)
    assert torch.equal(y, x + 1)  # target at t is the token at t+1


def test_shift_with_exact_rows():
    rows = torch.arange(2 * 16).view(2, 16)
    x, y = make_inputs_targets(rows, 16)
    assert torch.equal(x, rows)
    assert torch.equal(y[:, :-1], rows[:, 1:])
    assert (y[:, -1] == IGNORE_INDEX).all()


def test_loss_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    y = torch.randint(0, 11, (2, 5))
    manual = -torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).mean()
    assert torch.allclose(lm_loss(logits, y), manual, atol=1e-6)


def test_model_learns_shifted_identity():
    """A model trained on 'next token = current token + 1' must reach low loss -> shift is wired correctly."""
    torch.manual_seed(0)
    cfg = tiny_model_cfg("A", vocab_size=16)
    m = build_model(cfg, 0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(150):
        start = torch.randint(0, 16, (8, 1))
        rows = (start + torch.arange(17)) % 16
        x, y = make_inputs_targets(rows, 16)
        loss = lm_loss(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2
