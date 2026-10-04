# MoE Fusion Experiment — A (serial) vs B (parallel) vs C (parallel + periodic FusionMoE)

A controlled architecture experiment for one question:

> Can the strict Attention → MoE dependency be removed from most Transformer layers (Attention ∥ MoE),
> with the lost interaction recovered by periodic FusionMoE layers, for a better quality-vs-wall-clock trade-off?

* Pre-registration (hypotheses, fixed conditions, decision rules): [`EXPERIMENT_SPEC.md`](EXPERIMENT_SPEC.md)
* How confounders are controlled: [`METHODOLOGY.md`](METHODOLOGY.md)
* Verified literature: [`REFERENCES.md`](REFERENCES.md)

## Quick start (Google Colab, A100) — Hızlı başlangıç

1. Runtime → Change runtime type → **A100 GPU**. Your prepared data should be in `/content/moe_data`
   (`train.npy`, `validation.npy`, `tokenizer.json`, `manifest.json`, `*_documents.jsonl`).
2. Paste the contents of [`colab/PILOT_SINGLE_CELL.py`](colab/PILOT_SINGLE_CELL.py) into ONE cell (or open
   `colab/A100_MoE_Fusion_Experiment.ipynb`) and run it.
3. When asked, upload `moe_fusion_experiment.zip`. Everything up to the end of the pilot runs automatically.

`RUN_MODE = "smoke"` runs only the validation stages (~10 min). `RUN_MODE = "pilot"` (default) adds the systems
benchmark, profiling and four pilot trainings of 25.2M tokens each. Re-running the cell after a disconnect resumes:
finished phases and runs are skipped (results are mirrored to Drive; interrupted runs resume from their local
checkpoint in the same session, or restart from step 0 in a new session).

## What the pipeline does (`scripts/run_pipeline.py`)

| phase | script | gate |
|---|---|---|
| 0 environment | `env_check.py` | CUDA + BF16 required; attention backend selected ONCE and pinned |
| 1 data | `verify_dataset.py` | shapes, token range, sha256 vs manifest, document-level leakage, re-tokenisation provenance |
| 2 unit tests | `pytest tests` | all tests (incl. CUDA-stream equivalence on the GPU) must pass |
| 3 budget | `inspect_models.py` | analytic = instantiated parameter counts; compute matching ≤ 2% |
| 4 memory probe | `memory_probe.py` | largest micro-batch that fits **all** models (fresh process per trial) |
| 5 smoke | `run_smoke.py` | loss decreases, no NaN, checkpoint save → resume, evaluation, no router collapse |
| 6 benchmark | `benchmark_models.py` | full-size reference-vs-concurrent equivalence |
| 7 profiling | `profile_model.py` | Chrome/Perfetto traces + automated kernel-overlap analysis |
| 8 training | `run_experiment.py` ×4 | A, B, C_same, C_matched — same steps/tokens/data order/eval points |
| 9 report | `aggregate_results.py` | `results.json`, plots, `FINAL_REPORT.md`, automated fairness audit |

Outputs: `/content/moe_fusion_runs/<mode>/` (mirrored without checkpoints to
`/content/drive/MyDrive/moe_fusion_experiment/runs/<mode>/`) and `moe_fusion_<mode>_results.zip`.

## Models (pilot/main configuration)

| model | MoE applications | expert hidden | total params | active params/token | train FLOPs/token |
|---|---|---|---|---|---|
| A_serial | 12 | 1920 | 477.65M | 159.15M | 1.068 G |
| B_parallel | 12 | 1920 | 477.65M | 159.15M | 1.068 G |
| C_parallel_fusion_samewidth | 15 | 1920 | 583.84M | 185.71M | 1.227 G |
| C_parallel_fusion_matched | 15 | 1536 | 477.67M | 159.17M | 1.068 G |

## Repository layout

```
configs/            base.yaml (shared) + models/*.yaml (architecture fields only) + runs/{smoke,pilot,main}.yaml
src/moefusion/      config, rmsnorm, rope, attention, moe, blocks, model, data, trainer, checkpoint, metrics,
                    flop_counter, benchmarking, trace_analysis, dataset_verify, tokenizer, analysis, utils
scripts/            pipeline, phases, aggregation, plots, report, fallback data/tokenizer preparation
tests/              pytest suite (CPU + CUDA tests)
colab/              PILOT_SINGLE_CELL.py and the notebook
data_reference/     the manifest + tokenizer.json of the prepared dataset (hash-verified at runtime)
```

Design notes: the project uses an importable package (`src/moefusion`) instead of loose `src/*.py` modules to avoid
import-path ambiguity in Colab; the sixteen modules requested in the specification map 1:1 onto files in it
(`losses` live in `model.py`/`moe.py`, `evaluator` in `trainer.py`).

## Running pieces manually

```bash
python -m pytest -q tests
python scripts/inspect_models.py --run pilot --out /content/moe_fusion_runs/pilot
python scripts/run_experiment.py --model A_serial --run pilot --micro-batch 16 \
       --data-dir /content/moe_data --out /content/moe_fusion_runs/pilot/runs
python scripts/aggregate_results.py --root /content/moe_fusion_runs/pilot
# MAIN (100M tokens) — deliberately not in the single cell:
python scripts/run_pipeline.py --mode main --data-dir /content/moe_data --out /content/moe_fusion_runs \
       --drive-dir /content/drive/MyDrive/moe_fusion_experiment
```
