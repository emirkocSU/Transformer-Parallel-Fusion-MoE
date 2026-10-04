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
        loss = lm + m.aux_penalty(type("T", (), {"balance_coef": 0.01, "zloss_coef": 0.0, "aux_loss_reduction": "sum"}))
        loss.backward()
        for n, p in m.named_parameters():
            assert p.grad is not None, n
            assert torch.isfinite(p.grad).all(), n
        opt.step()
        assert torch.isfinite(loss)
        # initial loss close to ln(V) for a 0.02-std init
        assert abs(lm.item() - torch.log(torch.tensor(float(cfg.vocab_size))).item()) < 0.5


def test_aux_penalty_is_per_router_sum():
    """Switch convention: every router contributes balance_coef * L_balance (same pressure per router in A, B, C)."""
    from conftest import tiny_train_cfg

    cfg = tiny_model_cfg("C")
    m = build_model(cfg, 0)
    m(torch.randint(0, cfg.vocab_size, (2, 16)))
    vals = torch.stack([mod.last_aux["balance_loss"] for mod in m.moe_modules()])
    t = tiny_train_cfg(balance_coef=0.01, zloss_coef=0.0)
    assert t.aux_loss_reduction == "sum"
    assert torch.allclose(m.aux_penalty(t), 0.01 * vals.sum())
    assert torch.allclose(m.aux_losses()["balance_loss"], vals.mean())  # logged value: 1.0 = perfectly balanced
    tm = tiny_train_cfg(balance_coef=0.01, zloss_coef=0.0, aux_loss_reduction="mean")
    assert torch.allclose(m.aux_penalty(tm), 0.01 * vals.mean())
    g = torch.autograd.grad(m.aux_penalty(t), m.blocks[0].moe.router.weight)[0]
    assert g.abs().sum() > 0  # the balance term reaches every router
