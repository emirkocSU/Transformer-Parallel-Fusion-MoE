"""End-to-end orchestrator: environment -> data -> tests -> budget -> memory probe -> smoke -> benchmark ->
profiling -> training runs (A, B, C_same, C_matched) -> aggregation/plots/report -> packaging.

Every phase runs in its own Python process (a clean CUDA context per phase and per training run, so no run
inherits allocator state from another). Completed phases/runs are detected and skipped, so re-running the
same command after a Colab disconnect resumes where it stopped (training resumes from its last checkpoint).

  python scripts/run_pipeline.py --mode pilot --data-dir /content/moe_data --out /content/moe_fusion_runs \
         --drive-dir /content/drive/MyDrive/moe_fusion_experiment
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _common  # noqa: F401
from _common import CORE_MODELS, EXIT_OK, EXIT_SCIENTIFIC_FAILURE, ROOT, banner

PY = sys.executable
MB_CANDIDATES = [32, 16, 8, 4, 2, 1]
DATA_FILES = ["train.npy", "validation.npy", "tokenizer.json", "manifest.json", "source_manifest.json", ".complete.json",
              "train_documents.jsonl", "validation_documents.jsonl"]


class PipelineError(RuntimeError):
    pass


def sh(args, cwd=ROOT, capture_prefix=None, allow=(EXIT_OK,)):
    """Run a subprocess, stream its output live, return (exit_code, captured_lines_with_prefix)."""
    print(f"\n$ {' '.join(map(str, args))}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen([str(a) for a in args], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    captured = []
    for line in p.stdout:
        print(line, end="", flush=True)
        if capture_prefix and line.startswith(capture_prefix):
            captured.append(line[len(capture_prefix):].strip())
    rc = p.wait()
    if rc not in allow:
        raise PipelineError(f"command failed with exit code {rc}: {' '.join(map(str, args))}")
    return rc, captured


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, default=str))


def sync_to_drive(root: Path, drive_root: Path):
    """Copy results (no *.pt checkpoints/weights) to Google Drive."""
    if drive_root is None:
        return
    try:
        for src in root.rglob("*"):
            if src.is_dir() or src.suffix in (".pt", ".tmp"):
                continue
            dst = drive_root / src.relative_to(root)
            if dst.exists() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        print(f"  [drive] synced results to {drive_root}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [drive] WARNING: sync failed ({e}); results remain on local disk", flush=True)


# ----------------------------------------------------------------------------------------------- data
def ensure_data(data_dir: Path, drive_dir: Path, allow_rebuild: bool, backup: bool):
    banner("DATA LOCATION")
    have = (data_dir / "train.npy").exists() and (data_dir / "validation.npy").exists()
    if not have and drive_dir is not None:
        for cand in (drive_dir / "moe_data", drive_dir.parent / "moe_data"):
            if (cand / "train.npy").exists():
                print(f"  copying dataset from Drive {cand} -> {data_dir} (local disk; training never reads Drive)")
                data_dir.mkdir(parents=True, exist_ok=True)
                for f in DATA_FILES:
                    if (cand / f).exists():
                        shutil.copy2(cand / f, data_dir / f)
                have = True
                break
    if not have:
        if not allow_rebuild:
            raise PipelineError(f"No dataset in {data_dir} (and none on Drive). Re-run with ALLOW_DATA_REBUILD=True "
                                "to rebuild it from FineWeb-Edu.")
        print("  dataset missing -> rebuilding with scripts/prepare_fineweb.py (fallback; NOT byte-identical to the original)")
        sh([PY, "scripts/prepare_fineweb.py", "--out", data_dir])
    if not (data_dir / "tokenizer.json").exists():
        shutil.copy2(ROOT / "data_reference" / "tokenizer.json", data_dir / "tokenizer.json")
        print("  tokenizer.json missing in data dir -> copied the verified reference artifact")
    for f in DATA_FILES:
        print(f"  {'OK ' if (data_dir / f).exists() else '-- '} {data_dir / f}"
              + (f"  ({(data_dir / f).stat().st_size / 1e6:,.1f} MB)" if (data_dir / f).exists() else "  (missing)"))
    if backup and drive_dir is not None:
        dst = drive_dir / "moe_data"
        if not (dst / "train.npy").exists():
            try:
                dst.mkdir(parents=True, exist_ok=True)
                for f in DATA_FILES:
                    if (data_dir / f).exists():
                        shutil.copy2(data_dir / f, dst / f)
                print(f"  [drive] dataset backed up to {dst} (for future sessions)")
            except Exception as e:  # noqa: BLE001
                print(f"  [drive] WARNING: dataset backup failed: {e}")


# ----------------------------------------------------------------------------------------------- probe
def memory_probe(root: Path, run: str, models, attn: str, global_batch: int):
    banner("COMMON MEMORY PROBE (largest micro-batch safe for ALL architectures)")
    # GPU facts come from PHASE 0 (the orchestrator itself never creates a CUDA context, so it holds no GPU memory)
    env = load_json(root / "env" / "environment.json")
    gpu, total = env["gpu_name"], env["gpu_total_memory_gb"]
    if gpu == "cpu":  # MOE_ALLOW_CPU plumbing test
        mb = int(os.environ.get("MOE_CPU_MB", "4"))
        save_json(root / "probe.json", {"gpu": gpu, "micro_batch": mb, "grad_accum": global_batch // mb, "models": models})
        return mb
    cached = load_json(root / "probe.json")
    if cached and any((root / "runs").glob("*/summary.json")):
        # Runs already completed with this micro-batch: the remaining runs MUST use the same one (fairness).
        if cached.get("gpu") != gpu:
            print(f"  WARNING: GPU changed ({cached.get('gpu')} -> {gpu}); keeping micro-batch {cached['micro_batch']} for "
                  "fairness. The fairness audit will flag the GPU change for the wall-clock comparison.")
        return cached["micro_batch"]
    if cached and cached.get("gpu") == gpu and abs(cached.get("total_gb", 0) - total) < 1 and cached.get("models") == models:
        print(f"  reusing probe result: micro-batch {cached['micro_batch']} ({gpu})")
        return cached["micro_batch"]
    limit = 0.85 * total
    trials = []

    def trial(model, mb):
        _, cap = sh([PY, "scripts/memory_probe.py", "--model", model, "--run", run, "--mb", mb, "--attention-backend", attn],
                    capture_prefix="PROBE_RESULT ")
        r = json.loads(cap[-1]) if cap else {"ok": False, "error": "no result"}
        r["fits"] = bool(r.get("ok")) and r.get("peak_reserved_gb", 1e9) <= limit
        trials.append(r)
        print(f"  probe {model:32s} mb={mb:3d} -> {'fits' if r['fits'] else 'NO'} "
              f"(peak reserved {r.get('peak_reserved_gb', float('nan')):.1f} GB, limit {limit:.1f} GB) {r.get('error', '')}")
        return r["fits"]

    cands = [m for m in MB_CANDIDATES if global_batch % m == 0]
    # the largest model first (most parameters / MoE applications), then confirm on every other model
    order = sorted(models, key=lambda n: ("samewidth" not in n, "matched" not in n))
    chosen = None
    for mb in cands:
        if all(trial(m, mb) for m in order):
            chosen = mb
            break
    if chosen is None:
        raise PipelineError("no micro-batch size fits all architectures")
    save_json(root / "probe.json", {"gpu": gpu, "total_gb": total, "limit_gb": limit, "micro_batch": chosen,
                                    "grad_accum": global_batch // chosen, "models": models, "trials": trials})
    print(f"  COMMON MICRO-BATCH = {chosen} (grad accumulation {global_batch // chosen}) for every architecture")
    return chosen


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["smoke", "pilot", "main"])
    ap.add_argument("--data-dir", default="/content/moe_data")
    ap.add_argument("--out", default="/content/moe_fusion_runs")
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--models", default=",".join(CORE_MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-data-rebuild", action="store_true")
    ap.add_argument("--backup-data-to-drive", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--skip-profile", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    models = args.models.split(",")
    root = Path(args.out) / args.mode
    root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    drive_base = Path(args.drive_dir) if args.drive_dir else None
    drive_root = drive_base / "runs" / args.mode if drive_base else None
    # restore results of a previous session from Drive (completed runs are then skipped)
    if drive_root is not None and drive_root.exists() and not (root / "pipeline_status.json").exists():
        print(f"  restoring previous results from {drive_root}")
        shutil.copytree(drive_root, root, dirs_exist_ok=True)
    status = load_json(root / "pipeline_status.json") or {"phases": {}}

    def mark(phase, info=None):
        status["phases"][phase] = {"done": True, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})}
        save_json(root / "pipeline_status.json", status)
        sync_to_drive(root, drive_root)

    try:
        # PHASE 0 - environment (always re-run: the runtime may have changed)
        sh([PY, "scripts/env_check.py", "--out", root])
        attn = load_json(root / "env" / "attention_backend.json")["selected"]
        prev_attn = status.get("attention_backend")
        if prev_attn and prev_attn != attn:
            raise PipelineError(f"attention backend changed between sessions ({prev_attn} -> {attn}); "
                                "results would mix implementations. Use a fresh --out directory.")
        status["attention_backend"] = attn
        mark("env", {"attention_backend": attn})

        # PHASE 1 - data
        ensure_data(data_dir, drive_base, args.allow_data_rebuild, args.backup_data_to_drive)
        sys.path.insert(0, str(ROOT / "src"))
        from moefusion.config import load_experiment

        mcfg0, tcfg = load_experiment(models[0], args.mode)
        sh([PY, "scripts/verify_dataset.py", "--data-dir", data_dir, "--out", root, "--seq-len", tcfg.seq_len,
            "--vocab-size", mcfg0.vocab_size])
        mark("dataset")

        # PHASE 2 - unit tests (CPU + CUDA tests on this GPU)
        banner("UNIT TESTS (pytest)")
        (root / "tests").mkdir(exist_ok=True)
        rc, _ = sh([PY, "-m", "pytest", "-q", "-rA", "tests", f"--junitxml={root / 'tests' / 'junit.xml'}"], allow=(0, 1))
        import xml.etree.ElementTree as ET

        ts = ET.parse(root / "tests" / "junit.xml").getroot()
        ts = ts if ts.tag == "testsuite" else ts.find("testsuite")
        summ = {k: int(ts.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
        summ["passed"] = rc == 0
        save_json(root / "tests" / "pytest_summary.json", summ)
        print(f"  pytest: {summ}")
        if rc != 0:
            raise PipelineError("unit tests failed - no training is started")
        mark("tests", summ)

        # PHASE 3 - parameter / FLOP budget
        sh([PY, "scripts/inspect_models.py", "--run", args.mode, "--models", ",".join(models), "--out", root])
        mark("inspect")

        # PHASE 4 - common micro-batch
        mb = memory_probe(root, args.mode, models, attn, tcfg.global_batch_seqs)
        status["micro_batch"] = mb
        mark("probe", {"micro_batch": mb})

        # PHASE 5 - smoke
        if not (load_json(root / "smoke" / "smoke_report.json") or {}).get("passed"):
            sh([PY, "scripts/run_smoke.py", "--models", ",".join(models), "--micro-batch", mb, "--data-dir", data_dir,
                "--out", root, "--attention-backend", attn])
            mark("smoke")
        else:
            print("  smoke already passed - skipped")

        if args.mode != "smoke":
            # PHASE 6 - systems benchmark
            if not args.skip_benchmark and not (root / "benchmark" / "benchmark.json").exists():
                sh([PY, "scripts/benchmark_models.py", "--run", args.mode, "--micro-batch", mb, "--data-dir", data_dir,
                    "--out", root, "--attention-backend", attn])
                mark("benchmark")
            # PHASE 7 - profiling
            if not args.skip_profile and not (root / "profiles" / "overlap_analysis.json").exists():
                sh([PY, "scripts/profile_model.py", "--run", args.mode, "--micro-batch", mb, "--out", root,
                    "--attention-backend", attn])
                mark("profile")

            # PHASE 8 - training runs (fixed, documented order)
            for name in models:
                run_dir = root / "runs" / f"{name}_seed{args.seed}"
                if (run_dir / "summary.json").exists():
                    print(f"  {name}: already completed - skipped")
                    continue
                if (run_dir / "failure.json").exists():
                    print(f"  {name}: previously FAILED (see failure.json) - not re-run automatically")
                    continue
                rc, _ = sh([PY, "scripts/run_experiment.py", "--model", name, "--run", args.mode, "--micro-batch", mb,
                            "--data-dir", data_dir, "--out", root / "runs", "--attention-backend", attn, "--seed", args.seed,
                            "--dataset-report", root / "dataset_verification.json"],
                           allow=(EXIT_OK, EXIT_SCIENTIFIC_FAILURE))
                mark(f"train_{name}", {"exit_code": rc})

            # PHASE 9 - aggregation, plots, report
            sh([PY, "scripts/aggregate_results.py", "--root", root, "--mode", args.mode])
            mark("report")

        # PHASE 10 - package
        zip_base = Path(args.out) / f"moe_fusion_{args.mode}_results"
        tmp = Path(args.out) / f"_pack_{args.mode}"
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(root, tmp, ignore=shutil.ignore_patterns("*.pt", "*.tmp"))
        shutil.make_archive(str(zip_base), "zip", tmp)
        shutil.rmtree(tmp)
        if drive_root is not None:
            try:
                shutil.copy2(f"{zip_base}.zip", drive_base / f"{zip_base.name}.zip")
            except Exception as e:  # noqa: BLE001
                print(f"  [drive] WARNING: could not copy results zip: {e}")
        mark("package", {"zip": f"{zip_base}.zip"})
        banner(f"PIPELINE COMPLETE in {(time.time() - t_start) / 60:.1f} min  ->  {root}/FINAL_REPORT.md  |  {zip_base}.zip")
    except PipelineError as e:
        banner(f"PIPELINE STOPPED: {e}")
        status["last_error"] = str(e)
        save_json(root / "pipeline_status.json", status)
        sync_to_drive(root, drive_root)
        sys.exit(1)


if __name__ == "__main__":
    main()
