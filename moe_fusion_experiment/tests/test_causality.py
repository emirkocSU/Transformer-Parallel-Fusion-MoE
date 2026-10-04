import torch

from conftest import tiny_model_cfg
from moefusion.model import build_model


def test_no_future_leakage_all_archs():
    """Changing future tokens must not change logits at earlier positions (MANDATORY)."""
    torch.manual_seed(0)
    for arch in "ABC":
        cfg = tiny_model_cfg(arch)
        m = build_model(cfg, seed=1).eval()
        T = cfg.max_seq_len
        x = torch.randint(0, cfg.vocab_size, (2, T))
        for cut in (1, T // 2, T - 1):
            x2 = x.clone()
            x2[:, cut:] = torch.randint(0, cfg.vocab_size, (2, T - cut))
            with torch.no_grad():
                l1, l2 = m(x), m(x2)
            assert torch.allclose(l1[:, :cut], l2[:, :cut], atol=1e-5, rtol=0), (arch, cut)
            assert not torch.allclose(l1[:, cut:], l2[:, cut:]), "future change should affect later positions"


def test_causality_through_moe_batch_coupling():
    """MoE routing is token-wise; batch composition must not leak information across positions."""
    cfg = tiny_model_cfg("C")
    m = build_model(cfg, seed=2).eval()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    with torch.no_grad():
        full = m(x)
        prefix = m(x[:, :5])
    assert torch.allclose(full[:, :5], prefix, atol=1e-5)
