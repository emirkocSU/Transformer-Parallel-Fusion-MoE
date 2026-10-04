import copy

import torch

from conftest import requires_cuda, tiny_model_cfg
from moefusion.blocks import ParallelBlock
from moefusion.model import build_model


def _block(device, d=64, heads=4, hd=16, T=32):
    cfg = tiny_model_cfg("B", d_model=d, n_heads=heads, head_dim=hd, max_seq_len=T, expert_hidden=96)
    torch.manual_seed(0)
    blk = ParallelBlock(cfg)
    for p in blk.parameters():
        if p.ndim > 1:
            torch.nn.init.normal_(p, 0, 0.05)
    return blk.to(device), cfg


def test_reference_math_matches_definition_cpu():
    blk, cfg = _block("cpu")
    x = torch.randn(2, cfg.max_seq_len, cfg.d_model)
    y = blk(x)
    expected = x + blk.attn(blk.norm_attn(x)) + blk.moe(blk.norm_moe(x))
    assert torch.allclose(y, expected, atol=1e-6)


def test_serial_differs_from_parallel_cpu():
    a = build_model(tiny_model_cfg("A"), 0)
    b = build_model(tiny_model_cfg("B"), 0)
    x = torch.randint(0, 97, (2, 16))
    assert not torch.allclose(a(x), b(x)), "A and B share weights but must compute different functions"


@requires_cuda
def test_concurrent_forward_matches_reference_fp32():
    blk, cfg = _block("cuda")
    x = torch.randn(4, cfg.max_seq_len, cfg.d_model, device="cuda")
    blk.execution = "reference"
    y_ref = blk(x)
    blk.execution = "concurrent"
    y_con = blk(x)
    torch.cuda.synchronize()
    assert torch.allclose(y_ref, y_con, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_concurrent_forward_matches_reference_bf16():
    blk, cfg = _block("cuda", d=128, heads=2, hd=64, T=128)
    x = torch.randn(4, cfg.max_seq_len, cfg.d_model, device="cuda")
    outs = {}
    for mode in ("reference", "concurrent"):
        blk.execution = mode
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs[mode] = blk(x).float()
    torch.cuda.synchronize()
    assert torch.allclose(outs["reference"], outs["concurrent"], atol=2e-2, rtol=2e-2)


@requires_cuda
def test_full_model_concurrent_matches_reference():
    cfg = tiny_model_cfg("C", d_model=64, n_heads=4, head_dim=16, max_seq_len=32)
    m = build_model(cfg, 0, "cuda")
    x = torch.randint(0, cfg.vocab_size, (3, 32), device="cuda")
    m.set_parallel_backend("reference")
    a = m(x)
    m.set_parallel_backend("concurrent")
    b = m(x)
    torch.cuda.synchronize()
    assert torch.allclose(a, b, atol=1e-5)


@requires_cuda
def test_concurrent_repeated_no_race():
    """Many repeated launches; outputs must stay identical (detects missing stream synchronisation)."""
    blk, cfg = _block("cuda")
    x = torch.randn(8, cfg.max_seq_len, cfg.d_model, device="cuda")
    blk.execution = "reference"
    ref = blk(x).detach()
    blk.execution = "concurrent"
    for _ in range(50):
        y = blk(x)
        z = y * 1.0  # consumer on the current stream immediately after the join
    torch.cuda.synchronize()
    assert torch.allclose(z, ref, atol=1e-5)
