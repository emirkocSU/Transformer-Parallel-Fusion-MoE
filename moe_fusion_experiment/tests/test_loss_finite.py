import torch

from conftest import tiny_model_cfg
from moefusion.model import build_model, lm_loss, make_inputs_targets


def test_one_train_step_finite_all_archs():
    for arch in "ABC":
        cfg = tiny_model_cfg(arch)
        m = build_model(cfg, 0)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        rows = torch.randint(0, cfg.vocab_size, (4, 17))
        x, y = make_inputs_targets(rows, 16)
        logits = m(x)
        lm = lm_loss(logits, y)
        aux = m.aux_losses()
        loss = lm + 0.01 * aux["balance_loss"]
        loss.backward()
        for n, p in m.named_parameters():
            assert p.grad is not None, n
            assert torch.isfinite(p.grad).all(), n
        opt.step()
        assert torch.isfinite(loss)
        # initial loss close to ln(V) for a 0.02-std init
        assert abs(lm.item() - torch.log(torch.tensor(float(cfg.vocab_size))).item()) < 0.5


def test_aux_loss_is_mean_over_moe_layers():
    cfg = tiny_model_cfg("C")
    m = build_model(cfg, 0)
    m(torch.randint(0, cfg.vocab_size, (2, 16)))
    vals = torch.stack([mod.last_aux["balance_loss"] for mod in m.moe_modules()])
    assert torch.allclose(m.aux_losses()["balance_loss"], vals.mean())
