import json

from moefusion.trace_analysis import analyze_trace


def _trace(tmp_path, concurrent: bool):
    ev = []
    # CPU ranges on thread 1
    ev.append({"ph": "X", "cat": "user_annotation", "name": "parallel_attention_branch", "ts": 0, "dur": 10, "tid": 1})
    ev.append({"ph": "X", "cat": "user_annotation", "name": "parallel_moe_branch", "ts": 10, "dur": 10, "tid": 1})
    # launches
    ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 2, "dur": 1, "tid": 1, "args": {"correlation": 1}})
    ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 12, "dur": 1, "tid": 1, "args": {"correlation": 2}})
    # kernels: attention 100-200, moe 150-260 (overlap 50) if concurrent; else serial 100-200, 200-310
    sa, sm = (7, 13) if concurrent else (7, 7)
    k2 = (150, 110) if concurrent else (200, 110)
    ev.append({"ph": "X", "cat": "kernel", "name": "attn_k", "ts": 100, "dur": 100, "args": {"correlation": 1, "stream": sa}})
    ev.append({"ph": "X", "cat": "kernel", "name": "moe_k", "ts": k2[0], "dur": k2[1], "args": {"correlation": 2, "stream": sm}})
    # an unannotated (backward) kernel on the moe stream overlapping a backward kernel on the attention stream
    ev.append({"ph": "X", "cat": "kernel", "name": "bwd_a", "ts": 400, "dur": 40, "args": {"correlation": 9, "stream": sa}})
    ev.append({"ph": "X", "cat": "kernel", "name": "bwd_m", "ts": 420, "dur": 40, "args": {"correlation": 10, "stream": sm}})
    p = tmp_path / ("c.json" if concurrent else "r.json")
    p.write_text(json.dumps({"traceEvents": ev}))
    return str(p)


def test_concurrent_overlap_measured(tmp_path):
    r = analyze_trace(_trace(tmp_path, True))
    assert r["forward_range_based"]["overlap_us"] == 50
    assert abs(r["forward_range_based"]["overlap_fraction_of_shorter"] - 0.5) < 1e-9
    sb = r["stream_based_fwd_bwd"]
    assert sb["applicable"] and sb["overlap_us"] == 50 + 20


def test_reference_no_overlap_and_stream_na(tmp_path):
    r = analyze_trace(_trace(tmp_path, False))
    assert r["forward_range_based"]["overlap_us"] == 0
    assert not r["stream_based_fwd_bwd"]["applicable"]
