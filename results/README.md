# Results

Machine-generated outputs of every phase, exactly as produced by the pipeline on a Colab A100 (40GB). No number in
these folders is hand-written. Write-up and figures: [main README](../README.md) · Turkish analysis:
[docs/BULGULAR_TR.md](../docs/BULGULAR_TR.md).

| folder | phase | tokens / model | report |
|---|---|---|---|
| [`pilot/`](pilot/) | A, B, C_same, C_matched, seed 42 | 25.2M | [`FINAL_REPORT.md`](pilot/FINAL_REPORT.md) |
| [`diagnostics/`](diagnostics/) | A vs B: seed 42, seed 43, dense control, 10% warm-up | 25.2M | [`DIAG_REPORT.md`](diagnostics/DIAG_REPORT.md) |
| [`main/`](main/) | A, B, C_same, C_matched, seed 42 | 100.0M | [`FINAL_REPORT.md`](main/FINAL_REPORT.md) |

Each run folder (`*/runs/<model>_seed42/`) contains:

| file | content |
|---|---|
| `eval_metrics.jsonl` | periodic validation loss, routing churn, wall-clock |
| `train_metrics.jsonl` | training loss, balance loss, gradient norm, learning rate, throughput, expert utilisation |
| `final_val_per_sequence_loss.npy` | loss of every validation sequence (used for the paired statistics) |
| `summary.json`, `experiment_manifest.json`, `events.jsonl` | config, parameter / FLOP counts, alerts |

Notes:
* `main/FINAL_REPORT.md` was regenerated from the same logs with the v1.4.2 report template (it fixes stale wording:
  the single-seed threshold, Q2/Q10 interpretation, the eval-subset size). All numbers are unchanged; the report
  produced in Colab is kept as `main/FINAL_REPORT.colab_v1.4.1.md`.
* The pilot's profiler traces (`profiles/*.json`, ~170 MB) are not committed; their overlap analysis is in
  `pilot/results.json` and the pilot report.
