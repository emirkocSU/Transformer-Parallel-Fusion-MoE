"""Parse a torch.profiler Chrome trace and quantify attention/MoE kernel overlap.

Method (stated so that it can be audited):
 1. GPU kernels ("cat" == "kernel") are linked to the CPU launch call (cudaLaunchKernel & co.)
    through the "correlation" id.
 2. A kernel belongs to a branch if its CPU launch happened inside a user range named
    `parallel_attention_branch` or `parallel_moe_branch` on the same CPU thread (forward pass).
 3. Backward kernels carry no user range. In the concurrent backend every branch runs on its own
    CUDA stream, and autograd runs backward ops on the stream of the forward op, so each stream is
    mapped to the branch that owns the majority of its range-attributed kernels, and every kernel on
    that stream (forward AND backward) is attributed to that branch.
 4. Overlap = total length of the intersection of the union-of-intervals of attention kernels and
    of MoE kernels. Fractions are relative to the busy time of the shorter branch.
If attribution is ambiguous (e.g. both branches on one stream) the stream-based numbers are
reported as not applicable; the trace is kept for manual inspection in https://ui.perfetto.dev .
"""
from __future__ import annotations

import bisect
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

BRANCH_RANGES = {"parallel_attention_branch": "attention", "parallel_moe_branch": "moe"}
LAUNCH_NAMES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel", "cuLaunchKernelEx")


def _union(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _length(iv: List[Tuple[float, float]]) -> float:
    return sum(e - s for s, e in iv)


def _intersection(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def analyze_trace(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    kernels, launches, ranges = [], {}, defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = str(ev.get("cat", "")).lower()
        args = ev.get("args", {}) or {}
        if cat == "kernel":
            kernels.append(ev)
        elif cat in ("cuda_runtime", "cuda_driver") and any(n in ev.get("name", "") for n in LAUNCH_NAMES):
            if "correlation" in args:
                launches[args["correlation"]] = ev
        elif cat == "user_annotation" and ev.get("name") in BRANCH_RANGES:
            ranges[ev.get("tid")].append((float(ev["ts"]), float(ev["ts"]) + float(ev.get("dur", 0)), BRANCH_RANGES[ev["name"]]))

    for tid in ranges:
        ranges[tid].sort()
    starts = {tid: [r[0] for r in rs] for tid, rs in ranges.items()}

    def branch_of_launch(lev) -> str:
        tid = lev.get("tid")
        if tid not in ranges:
            return ""
        ts = float(lev["ts"])
        # branch ranges never nest within each other on one thread -> the last range that started
        # before the launch is the only candidate
        k = bisect.bisect_right(starts[tid], ts) - 1
        if k >= 0:
            s, e, b = ranges[tid][k]
            if s <= ts <= e:
                return b
        return ""

    per_kernel = []
    for kv in kernels:
        args = kv.get("args", {}) or {}
        stream = args.get("stream")
        corr = args.get("correlation")
        b = branch_of_launch(launches[corr]) if corr in launches else ""
        per_kernel.append((float(kv["ts"]), float(kv["ts"]) + float(kv.get("dur", 0)), stream, b))

    # range (forward) attribution
    fwd = {"attention": [], "moe": []}
    stream_votes: Dict[object, Counter] = defaultdict(Counter)
    for s, e, st, b in per_kernel:
        if b:
            fwd[b].append((s, e))
            stream_votes[st][b] += 1
    res: Dict = {"n_kernels": len(per_kernel), "n_range_attributed": sum(len(v) for v in fwd.values())}
    ua, um = _union(fwd["attention"]), _union(fwd["moe"])
    ov = _intersection(ua, um)
    res["forward_range_based"] = {
        "attention_busy_us": _length(ua), "moe_busy_us": _length(um), "overlap_us": ov,
        "overlap_fraction_of_shorter": ov / max(1e-9, min(_length(ua), _length(um))) if ua and um else 0.0,
    }
    # stream attribution (whole step incl. backward)
    stream_map = {st: c.most_common(1)[0][0] for st, c in stream_votes.items()}
    att_streams = {st for st, b in stream_map.items() if b == "attention"}
    moe_streams = {st for st, b in stream_map.items() if b == "moe"}
    res["streams"] = {str(st): dict(c) for st, c in stream_votes.items()}
    if att_streams and moe_streams and not (att_streams & moe_streams):
        sa = _union([(s, e) for s, e, st, _ in per_kernel if st in att_streams])
        sm = _union([(s, e) for s, e, st, _ in per_kernel if st in moe_streams])
        ov2 = _intersection(sa, sm)
        res["stream_based_fwd_bwd"] = {
            "applicable": True, "attention_streams": sorted(map(str, att_streams)), "moe_streams": sorted(map(str, moe_streams)),
            "attention_busy_us": _length(sa), "moe_busy_us": _length(sm), "overlap_us": ov2,
            "overlap_fraction_of_shorter": ov2 / max(1e-9, min(_length(sa), _length(sm))),
        }
    else:
        res["stream_based_fwd_bwd"] = {"applicable": False,
                                       "reason": "attention and MoE kernels share a CUDA stream (reference execution) or attribution failed"}
    total = _union([(s, e) for s, e, _, _ in per_kernel])
    res["gpu_busy_us"] = _length(total)
    return res
