"""Separate systems benchmark (NOT the training measurement).

1. Full model, one micro-batch, real validation tokens resident on the GPU: forward / forward+backward /
   full optimizer step, for A, B, C(same), C(matched) with REFERENCE execution, and B/C with CONCURRENT
   execution. Models are benchmarked in interleaved, rotated rounds to cancel drift (thermal/clock).
2. Branch-level timing of one parallel block: attention-only, MoE-only, reference, concurrent.
3. Full-size GPU equivalence of reference vs concurrent (outputs + gradients).
4. MoE backend comparison (reference vs padded_bmm) -- informational only.
"""
import argparse
import json
import sys
from pathlib import Path

import _common  # noqa: F401
from _common import EXIT_GATE_FAILED, banner

import numpy as np
import torch

from moefusion.benchmarking import branch_benchmark, concurrent_equivalence, model_benchmark
from moefusion.config import load_experiment
from moefusion.data import load_token_array
from moefusion.benchmarking import free_cuda
from moefusion.utils import configure_torch_numerics, write_json


def _guard(fn, what):
    """Run one measurement; an OOM/runtime error is RECORDED (systems measurement), never fatal for the pipeline."""
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001  (torch.cuda.OutOfMemoryError is a RuntimeError subclass)
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
        print(f"  WARNING: {what} failed -> recorded, skipped: {msg}", flush=True)
        free_cuda()
        return None, msg

VARIANTS = [
    ("A_serial", "A_serial", "reference"),
    ("B_parallel[reference]", "B_parallel", "reference"),
    ("B_parallel[concurrent]", "B_parallel", "concurrent"),
    ("C_samewidth[reference]", "C_parallel_fusion_samewidth", "reference"),
    ("C_samewidth[concurrent]", "C_parallel_fusion_samewidth", "concurrent"),
    ("C_matched[reference]", "C_parallel_fusion_matched", "reference"),
    ("C_matched[concurrent]", "C_parallel_fusion_matched", "concurrent"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--micro-batch", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attention-backend", default="sdpa_flash")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    configure_torch_numerics()
    banner("SYSTEMS BENCHMARK (separate from training)")
    dev = torch.device("cuda")
    ov = lambda pb, mb=args.micro_batch: {"model": {"attention_backend": args.attention_backend, "parallel_backend": pb},
                                         "train": {"micro_batch_seqs": mb}}
    m0, t0 = load_experiment("A_serial", args.run, ov("reference"))
    val = load_token_array(Path(args.data_dir) / "validation.npy", t0.seq_len, m0.vocab_size, check_range=False)
    rows = torch.from_numpy(np.ascontiguousarray(val[: args.micro_batch]).astype(np.int64)).to(dev)
    res = {"micro_batch": args.micro_batch, "rounds": args.rounds, "warmup": args.warmup, "iters": args.iters,
           "input": "first micro-batch of validation rows, resident on GPU (model compute throughput)",
           "untrained_models": True, "full_model": {}, "round_order": []}

    # ---- 1. interleaved full-model benchmark
    raw = {label: {"forward": [], "forward_backward": [], "optimizer_step": []} for label, _, _ in VARIANTS}
    mem = {label: [] for label, _, _ in VARIANTS}
    errors = {}
    for r in range(args.rounds):
        order = VARIANTS[r % len(VARIANTS):] + VARIANTS[: r % len(VARIANTS)]
        res["round_order"].append([v[0] for v in order])
        for label, name, pb in order:
            if label in errors:
                continue
            m, t = load_experiment(name, args.run, ov(pb))
            b, err = _guard(lambda: model_benchmark(m, t, rows, args.warmup, args.iters), f"benchmark {label}")
            if err:
                errors[label] = err
                continue
            for k in raw[label]:
                raw[label][k].append(b[k]["median_ms"])
            mem[label].append(b["peak_mem_alloc_gb"])
            print(f"  round {r + 1}/{args.rounds} {label:26s} fwd {b['forward']['median_ms']:8.1f} ms  "
                  f"fwd+bwd {b['forward_backward']['median_ms']:8.1f} ms  step {b['optimizer_step']['median_ms']:8.1f} ms  "
                  f"peak {b['peak_mem_alloc_gb']:.1f} GB", flush=True)
    ntok = args.micro_batch * t0.seq_len
    for label in raw:
        if label in errors:
            res["full_model"][label] = {"error": errors[label]}
            continue
        d = {}
        for k, v in raw[label].items():
            a = np.asarray(v)
            d[k] = {"median_of_round_medians_ms": float(np.median(a)), "mean_ms": float(a.mean()),
                    "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0, "p95_ms": float(np.percentile(a, 95)),
                    "round_medians_ms": a.tolist(), "tokens_per_sec": ntok / (float(np.median(a)) / 1e3),
                    "sequences_per_sec": args.micro_batch / (float(np.median(a)) / 1e3)}
        d["peak_mem_alloc_gb"] = float(max(mem[label]))
        res["full_model"][label] = d
    ok = {k: v for k, v in res["full_model"].items() if "error" not in v}
    ref_step = ok["A_serial"]["optimizer_step"]["median_of_round_medians_ms"] if "A_serial" in ok else None
    print("\n  label                       step ms   rel. to A")
    for label, v in res["full_model"].items():
        if "error" in v:
            print(f"  {label:26s}  FAILED: {v['error']}")
            continue
        s = v["optimizer_step"]["median_of_round_medians_ms"]
        print(f"  {label:26s} {s:9.1f}   " + (f"{s / ref_step:6.3f}x" if ref_step else "n/a"))

    # ---- 2. branch-level timing
    res["branches"] = {}
    for name in ("B_parallel", "C_parallel_fusion_matched"):
        m, t = load_experiment(name, args.run, ov("reference"))
        br, err = _guard(lambda: branch_benchmark(m, args.micro_batch, t.seq_len, args.warmup, 3 * args.iters),
                         f"branch benchmark {name}")
        if err:
            res["branches_errors"] = {**res.get("branches_errors", {}), name: err}
            continue
        res["branches"][f"{name}(hidden={m.expert_hidden})"] = br
        for mode in ("forward", "forward_backward"):
            a = br[mode]
            print(f"  [{name} {mode}] attn {a['attention_branch']['median_ms']:.2f} ms | moe {a['moe_branch']['median_ms']:.2f} ms | "
                  f"ref {a['parallel_reference']['median_ms']:.2f} ms | concurrent {a['parallel_concurrent']['median_ms']:.2f} ms | "
                  f"serial blk {a['serial_block']['median_ms']:.2f} | fusion blk {a['fusion_block']['median_ms']:.2f} -> "
                  f"{a['analysis']['interpretation']}")

    # ---- 3. full-size equivalence on this GPU
    m, t = load_experiment("B_parallel", args.run, ov("reference"))
    eq, err = _guard(lambda: concurrent_equivalence(m, args.micro_batch, t.seq_len), "full-size equivalence check")
    eq = eq or {"passed": False, "error": err}
    res["concurrent_equivalence_full_size"] = eq
    print("  reference vs concurrent (full size, bf16):", json.dumps(eq))

    # ---- 4. MoE backend comparison (informational)
    res["moe_backend_comparison"] = {}
    for backend in ("reference", "padded_bmm"):
        o = ov("reference")
        o["model"]["moe_backend"] = backend
        m, t = load_experiment("A_serial", args.run, o)
        b, err = _guard(lambda: model_benchmark(m, t, rows, args.warmup, args.iters), f"MoE backend {backend}")
        if err:
            res["moe_backend_comparison"][backend] = {"error": err}
            continue
        res["moe_backend_comparison"][backend] = {k: b[k]["median_ms"] for k in ("forward", "forward_backward", "optimizer_step")}
        print(f"  MoE backend {backend:10s}: step {b['optimizer_step']['median_ms']:.1f} ms (A_serial)")

    write_json(Path(args.out) / "benchmark" / "benchmark.json", res)
    if not eq["passed"]:
        # Training uses REFERENCE execution, so this invalidates only the concurrency (systems) results.
        print("  ERROR: concurrent execution does not match reference at full size -> concurrency results INVALID "
              "(recorded in the report; the training comparison is unaffected)")
        sys.exit(EXIT_GATE_FAILED)


if __name__ == "__main__":
    main()
