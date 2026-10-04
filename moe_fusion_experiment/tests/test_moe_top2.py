import torch

from moefusion.moe import MoE


def _moe(backend="reference", E=4, k=2, d=16, h=24):
    torch.manual_seed(0)
    m = MoE(d, E, k, h, backend)
    for p in m.parameters():
        torch.nn.init.normal_(p, 0, 0.2)
    m.keep_routing = True
    return m


def test_exactly_two_experts_per_token():
    m = _moe()
    x = torch.randn(3, 10, 16)
    y = m(x)
    topi, topw = m.last_routing["topi"], m.last_routing["topw"]
    assert topi.shape == (30, 2)
    assert (topi[:, 0] != topi[:, 1]).all()
    assert torch.isfinite(topw).all()
    assert torch.allclose(topw.sum(-1), torch.ones(30), atol=1e-6)
    assert int(m.last_aux["counts"].sum()) == 30 * 2
    assert y.shape == x.shape


def test_combination_matches_explicit_loop():
    m = _moe()
    x = torch.randn(2, 7, 16)
    y = m(x).reshape(-1, 16)
    xf = x.reshape(-1, 16)
    topi, topw = m.last_routing["topi"], m.last_routing["topw"]
    ref = torch.zeros_like(xf)
    for t in range(xf.shape[0]):
        for j in range(2):
            e = int(topi[t, j])
            ref[t] += topw[t, j] * m.experts.forward_single(e, xf[t : t + 1])[0]
    assert torch.allclose(y, ref, atol=1e-5)


def test_selected_are_top_probabilities():
    m = _moe()
    x = torch.randn(1, 12, 16)
    m(x)
    probs, topi = m.last_routing["probs"], m.last_routing["topi"]
    sel = probs.gather(-1, topi)
    other = probs.clone().scatter_(-1, topi, -1.0).max(-1).values
    assert (sel.min(-1).values >= other).all()


def test_padded_bmm_backend_equivalent():
    x = torch.randn(2, 9, 16, requires_grad=True)
    ref = _moe("reference")
    pad = _moe("padded_bmm")
    pad.load_state_dict(ref.state_dict())
    y1, y2 = ref(x), pad(x)
    assert torch.allclose(y1, y2, atol=1e-5)
    g = torch.randn_like(y1)
    gx1 = torch.autograd.grad(y1, [x] + list(ref.parameters()), g)
    gx2 = torch.autograd.grad(y2, [x] + list(pad.parameters()), g)
    for a, b in zip(gx1, gx2):
        assert torch.allclose(a, b, atol=1e-5)


def test_router_receives_gradient():
    m = _moe()
    x = torch.randn(2, 5, 16)
    m(x).sum().backward()
    assert m.router.weight.grad is not None and m.router.weight.grad.abs().sum() > 0
