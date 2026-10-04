"""torch.profiler traces (CPU + CUDA) with named ranges, plus automated overlap analysis.

Traces: <out>/profiles/{A_serial, B_parallel_reference, B_parallel_concurrent, C_parallel_fusion,
        C_parallel_fusion_concurrent}.json   -> open in https://ui.perfetto.dev or chrome://tracing
Manual inspection: search for 'parallel_attention_branch' / 'parallel_moe_branch' on the CPU thread, then
compare the GPU stream rows underneath; in the concurrent traces two extra CUDA streams appear.
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
from _common import banner

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from moefusion.config import load_experiment
from moefusion.model import build_model, lm_loss, make_inputs_targets
from moefusion.trace_analysis import analyze_trace
from moefusion.trainer import autocast_ctx, make_optimizer
from moefusion.utils import configure_torch_numerics, set_profiling_ranges, write_json

TARGETS = [
    ("A_serial", "A_serial", "reference"),
    ("B_parallel_reference", "B_parallel", "reference"),
    ("B_parallel_concurrent", "B_parallel", "concurrent"),
    ("C_parallel_fusion", "C_parallel_fusion_samewidth", "reference"),
    ("C_parallel_fusion_concurrent", "C_parallel_fusion_samewidth", "concurrent"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--micro-batch", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attention-backend", default="sdpa_flash")
    args = ap.parse_args()
    configure_torch_numerics()
    set_profiling_ranges(True)  # identical instrumentation for every architecture
    banner("PROFILING (torch.profiler, CPU+CUDA, named ranges)")
    outdir = Path(args.out) / "profiles"
    outdir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    summary = {}
    for label, name, pb in TARGETS:
        m, t = load_experiment(name, args.run, {"model": {"attention_backend": args.attention_backend, "parallel_backend": pb},
                                                "train": {"micro_batch_seqs": args.micro_batch}})
        try:
            summary[label] = _profile_one(m, t, args, dev, outdir, label)
        except Exception as e:  # noqa: BLE001  - systems measurement: record, never fatal
            summary[label] = {"error": f"profiling failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}"}
            print(f"  WARNING: {label}: {summary[label]['error']}", flush=True)
        torch.cuda.empty_cache()
    write_json(outdir / "overlap_analysis.json", summary)


def _profile_one(m, t, args, dev, outdir, label):
    model = build_model(m, 0, dev)
    opt = make_optimizer(model, t, dev)
    rows = torch.randint(0, m.vocab_size, (args.micro_batch, t.seq_len + 1), device=dev)
    x, y = make_inputs_targets(rows, t.seq_len)

    def step():
        with autocast_ctx(t, dev):
            loss = lm_loss(model(x), y) + t.balance_coef * model.aux_losses()["balance_loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    path = outdir / f"{label}.json"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=0, warmup=1, active=2, repeat=1),
                 on_trace_ready=lambda p, path=path: p.export_chrome_trace(str(path))) as prof:
        for _ in range(3):
            with torch.profiler.record_function("train_step"):
                step()
            torch.cuda.synchronize()
            prof.step()
    try:
        a = analyze_trace(str(path))
    except Exception as e:  # noqa: BLE001
        a = {"error": f"automated parsing failed: {e}; inspect the trace manually"}
    fr = a.get("forward_range_based", {})
    sb = a.get("stream_based_fwd_bwd", {})
    print(f"  {label:30s} kernels={a.get('n_kernels')} fwd-overlap={fr.get('overlap_us', 0):.0f}us "
          f"({100 * fr.get('overlap_fraction_of_shorter', 0):.1f}% of shorter branch)  "
          f"fwd+bwd stream-overlap={'%.1f%%' % (100 * sb['overlap_fraction_of_shorter']) if sb.get('applicable') else 'n/a'}",
          flush=True)
    del model, opt
    return a


if __name__ == "__main__":
    main()
