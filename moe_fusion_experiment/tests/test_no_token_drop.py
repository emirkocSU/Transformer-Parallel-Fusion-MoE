import torch

from moefusion.moe import MoE


def _skewed(backend):
    torch.manual_seed(0)
    m = MoE(8, 4, 2, 8, backend)
    for p in m.experts.parameters():
        torch.nn.init.normal_(p, 0, 0.3)
    with torch.no_grad():  # route (almost) everything to experts 0 and 1 -> heavy overflow vs. any capacity
        m.router.weight.zero_()
        m.router.weight[0, 0] = 30.0
        m.router.weight[1, 0] = 29.0
    m.keep_routing = True
    return m


def test_overflow_tokens_are_processed_reference_and_padded():
    x = torch.randn(1, 100, 8)
    x[..., 0] = x[..., 0].abs() + 1.0
    for backend in ("reference", "padded_bmm"):
        m = _skewed(backend)
        y = m(x).reshape(-1, 8)
        counts = m.last_aux["counts"]
        assert int(counts.sum()) == 200
        assert counts[0] == 100 and counts[1] == 100  # every token went to experts 0 and 1
        # every single token output equals its explicit 2-expert computation (none dropped / zeroed)
        xf = x.reshape(-1, 8)
        topi, topw = m.last_routing["topi"], m.last_routing["topw"]
        for t in range(100):
            ref = sum(topw[t, j] * m.experts.forward_single(int(topi[t, j]), xf[t : t + 1])[0] for j in range(2))
            assert torch.allclose(y[t], ref, atol=1e-5)
        assert (y.abs().sum(-1) > 0).all()
