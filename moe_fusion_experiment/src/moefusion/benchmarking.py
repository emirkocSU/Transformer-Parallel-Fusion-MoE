"""CUDA-event based benchmarking (separate from training measurements).

All timings: warm-up iterations first, torch.cuda.Event timing, statistics over repeats
(mean, std, median = p50, p95). Inputs are real validation token rows already resident on the GPU,
so this measures MODEL COMPUTE THROUGHPUT (no data pipeline).
"""
from __future__ import annotations

import gc
from typing import Callable, Dict, List

import numpy as np
import torch

from .blocks import FusionBlock, ParallelBlock, SerialBlock
from .config import ModelConfig, TrainConfig
from .model import build_model, lm_loss, make_inputs_targets
from .trainer import autocast_ctx, make_optimizer


def summarize(ms: List[float]) -> Dict[str, float]:
    a = np.asarray(ms, dtype=np.float64)
    return {"mean_ms": float(a.mean()), "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median_ms": float(np.median(a)), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "n": int(a.size)}


def cuda_time(fn: Callable[[], None], warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e))
    return out


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def model_benchmark(mcfg: ModelConfig, tcfg: TrainConfig, rows: torch.Tensor, warmup: int, iters: int, seed: int = 0) -> Dict:
    """forward-only, forward+backward and full optimizer step on ONE micro-batch (`rows` on GPU)."""
    dev = rows.device
    model = build_model(mcfg, seed, dev)
    opt = make_optimizer(model, tcfg, dev)
    x, y = make_inputs_targets(rows, tcfg.seq_len)
    ntok = x.numel()

    def fwd():
        with torch.no_grad(), autocast_ctx(tcfg, dev):
            model(x)

    def fwd_bwd():
        with autocast_ctx(tcfg, dev):
            out = model(x)
            loss = lm_loss(out, y) + tcfg.balance_coef * model.aux_losses()["balance_loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)

    def step():
        with autocast_ctx(tcfg, dev):
            out = model(x)
            loss = lm_loss(out, y) + tcfg.balance_coef * model.aux_losses()["balance_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

    res = {}
    torch.cuda.reset_peak_memory_stats()
    for name, fn in (("forward", fwd), ("forward_backward", fwd_bwd), ("optimizer_step", step)):
        ms = cuda_time(fn, warmup, iters)
        st = summarize(ms)
        st["tokens_per_sec_median"] = ntok / (st["median_ms"] / 1e3)
        st["sequences_per_sec_median"] = x.shape[0] / (st["median_ms"] / 1e3)
        res[name] = st
    res["peak_mem_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["peak_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    res["tokens_per_microbatch"] = ntok
    del fwd, fwd_bwd, step, model, opt
    free_cuda()
    return res


def branch_benchmark(mcfg: ModelConfig, micro_batch: int, seq_len: int, warmup: int, iters: int, seed: int = 0) -> Dict:
    """Times the two branches of ONE parallel block separately and combined (reference vs concurrent),
    plus a serial block and a fusion block, forward and forward+backward, BF16 autocast."""
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    blk = ParallelBlock(mcfg).to(dev)
    ser = SerialBlock(mcfg).to(dev)
    ser.load_state_dict(blk.state_dict())
    fus = FusionBlock(mcfg).to(dev)
    for p in list(blk.parameters()) + list(ser.parameters()) + list(fus.parameters()):
        if p.ndim > 1:
            torch.nn.init.normal_(p, 0.0, 0.02)
    x = torch.randn(micro_batch, seq_len, mcfg.d_model, device=dev, requires_grad=True)
    g = torch.randn_like(x)

    def wrap(f, backward):
        def run():
            with torch.set_grad_enabled(backward), torch.autocast("cuda", dtype=torch.bfloat16):
                out = f()
            if backward:
                out.backward(g)
                x.grad = None
                for m in (blk, ser, fus):
                    m.zero_grad(set_to_none=True)
        return run

    def attn_only():
        return blk.attn(blk.norm_attn(x))

    def moe_only():
        return blk.moe(blk.norm_moe(x))

    def par_ref():
        blk.execution = "reference"
        return blk(x)

    def par_conc():
        blk.execution = "concurrent"
        return blk(x)

    out = {}
    for mode, bw in (("forward", False), ("forward_backward", True)):
        r = {}
        for name, f in (("attention_branch", attn_only), ("moe_branch", moe_only), ("parallel_reference", par_ref),
                        ("parallel_concurrent", par_conc), ("serial_block", lambda: ser(x)), ("fusion_block", lambda: fus(x))):
            r[name] = summarize(cuda_time(wrap(f, bw), warmup, iters))
        ta, tm = r["attention_branch"]["median_ms"], r["moe_branch"]["median_ms"]
        tr, tc = r["parallel_reference"]["median_ms"], r["parallel_concurrent"]["median_ms"]
        saved = (ta + tm) - tc
        r["analysis"] = {
            "sum_of_branches_ms": ta + tm,
            "concurrent_vs_reference_speedup": tr / tc,
            "time_saved_vs_sum_ms": saved,
            "overlap_fraction_of_shorter_branch": saved / min(ta, tm),
            "interpretation": ("No meaningful GPU overlap was achieved." if tc >= 0.97 * tr
                               else f"Concurrent execution is {100 * (1 - tc / tr):.1f}% faster than reference for this block."),
        }
        out[mode] = r
    blk.execution = mcfg.parallel_backend
    del blk, ser, fus, x, g
    free_cuda()
    return out


def concurrent_equivalence(mcfg: ModelConfig, micro_batch: int, seq_len: int, seed: int = 0) -> Dict:
    """Full-size GPU check: parallel_reference vs parallel_concurrent outputs and gradients (BF16 autocast)."""
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    blk = ParallelBlock(mcfg).to(dev)
    for p in blk.parameters():
        if p.ndim > 1:
            torch.nn.init.normal_(p, 0.0, 0.02)
    x = torch.randn(micro_batch, seq_len, mcfg.d_model, device=dev)
    g = torch.randn_like(x)
    res = {}
    outs, grads = {}, {}
    for mode in ("reference", "concurrent"):
        blk.execution = mode
        xi = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = blk(xi)
        o.backward(g)
        torch.cuda.synchronize()
        outs[mode] = o.detach().float()
        grads[mode] = {"x": xi.grad.detach().float(), **{n: p.grad.detach().float().clone() for n, p in blk.named_parameters()}}
        blk.zero_grad(set_to_none=True)
    res["forward_max_abs_diff"] = float((outs["reference"] - outs["concurrent"]).abs().max())
    res["forward_max_abs"] = float(outs["reference"].abs().max())
    worst = 0.0
    for n in grads["reference"]:
        a, b = grads["reference"][n], grads["concurrent"][n]
        rel = float((a - b).norm() / (a.norm() + 1e-12))
        worst = max(worst, rel)
    res["grad_worst_relative_l2_diff"] = worst
    res["passed"] = res["forward_max_abs_diff"] <= 1e-2 * max(1.0, res["forward_max_abs"]) and worst <= 1e-2
    blk.execution = mcfg.parallel_backend
    del blk
    free_cuda()
    return res
