import torch

from conftest import requires_cuda, tiny_model_cfg
from moefusion.blocks import ParallelBlock


def _grads(blk, x, g):
    xi = x.clone().requires_grad_(True)
    y = blk(xi)
    y.backward(g)
    out = {"x": xi.grad.detach().clone()}
    out.update({n: p.grad.detach().clone() for n, p in blk.named_parameters()})
    blk.zero_grad(set_to_none=True)
    return out


@requires_cuda
def test_concurrent_gradients_match_reference_fp32():
    cfg = tiny_model_cfg("B", d_model=64, n_heads=4, head_dim=16, max_seq_len=32, expert_hidden=96)
    torch.manual_seed(0)
    blk = ParallelBlock(cfg).cuda()
    for p in blk.parameters():
        if p.ndim > 1:
            torch.nn.init.normal_(p, 0, 0.05)
    x = torch.randn(4, 32, 64, device="cuda")
    g = torch.randn_like(x)
    blk.execution = "reference"
    gr = _grads(blk, x, g)
    blk.execution = "concurrent"
    gc = _grads(blk, x, g)
    torch.cuda.synchronize()
    for k in gr:
        assert torch.allclose(gr[k], gc[k], atol=1e-5, rtol=1e-4), k


@requires_cuda
def test_concurrent_gradients_with_accumulation_bf16():
    cfg = tiny_model_cfg("B", d_model=128, n_heads=2, head_dim=64, max_seq_len=128, expert_hidden=96)
    torch.manual_seed(0)
    blk = ParallelBlock(cfg).cuda()
    for p in blk.parameters():  # same initialisation as the fp32 test (the experiment uses init_parameters)
        if p.ndim > 1:
            torch.nn.init.normal_(p, 0, 0.05)
    xs = [torch.randn(2, 128, 128, device="cuda") for _ in range(3)]
    res = {}
    for mode in ("reference", "concurrent"):
        blk.execution = mode
        blk.zero_grad(set_to_none=True)
        for x in xs:  # gradient accumulation across micro-batches
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = blk(x)
            y.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        res[mode] = {n: p.grad.detach().float().clone() for n, p in blk.named_parameters()}
    for n in res["reference"]:
        a, b = res["reference"][n], res["concurrent"][n]
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), n
        assert (a - b).norm() <= 1e-2 * (a.norm() + 1e-8), n
