"""DIAGNOSTIC phase (EXPERIMENT_SPEC Amendment 3): why did B (parallel) beat A (serial) by 0.18 nats in the pilot?

Four groups, each A vs B at the pilot budget (384 steps x 65,536 tokens), each in its own results directory:
  diag_seed42    A_serial, B_parallel, seed 42                 (re-baseline with the current code)
  diag_dense     A_dense,  B_dense,    seed 42                 (D2: dense SwiGLU FFN, same active FLOPs)
  diag_warmup10  A_serial, B_parallel, seed 42, warm-up 10%    (D3)
  diag_seed43    A_serial, B_parallel, seed 43                 (D1)
D4 (routing stability) is measured inside every MoE run. One micro-batch for all groups (from the first group's
micro-batch 16 = pilot and MAIN; the pilot measured that it fits), identical data, identical code. Then scripts/diag_report.py applies the pre-registered rules R1-R4.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import _common  # noqa: F401
from _common import ROOT, banner

PY = sys.executable
GROUPS = [
    # tag,             models,                     run config,        seed, run tests, run smoke
    ("diag_seed42", "A_serial,B_parallel", "pilot", 42, True, True),
    ("diag_dense", "A_dense,B_dense", "pilot", 42, False, True),
    ("diag_warmup10", "A_serial,B_parallel", "pilot_warmup10", 42, False, False),
    ("diag_seed43", "A_serial,B_parallel", "pilot", 43, False, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/content/moe_data")
    ap.add_argument("--out", default="/content/moe_fusion_runs")
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--backup-data-to-drive", action="store_true")
    ap.add_argument("--allow-data-rebuild", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=16,
                    help="identical to the pilot and MAIN (pilot memory probe: A 31.1 GB, B 30.6 GB reserved of 42.4 GB)")
    args = ap.parse_args()
    out = Path(args.out)
    mb = args.micro_batch
    for i, (tag, models, run, seed, tests, smoke) in enumerate(GROUPS):
        banner(f"DIAGNOSTIC GROUP {i + 1}/{len(GROUPS)}: {tag}  ({models}, run={run}, seed={seed})")
        cmd = [PY, "scripts/run_pipeline.py", "--mode", "pilot", "--run", run, "--tag", tag, "--models", models,
               "--seed", str(seed), "--data-dir", args.data_dir, "--out", args.out, "--skip-benchmark", "--skip-profile"]
        if args.drive_dir:
            cmd += ["--drive-dir", args.drive_dir]
            if args.backup_data_to_drive:
                cmd += ["--backup-data-to-drive"]
        if args.allow_data_rebuild:
            cmd += ["--allow-data-rebuild"]
        if not tests:
            cmd += ["--skip-tests"]
        if not smoke:
            cmd += ["--skip-smoke"]
        cmd += ["--micro-batch", str(mb)]
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            print(f"DIAGNOSTICS STOPPED in group {tag} (exit {rc}). Re-run the cell: completed groups/runs are skipped.")
            sys.exit(rc)
    rc = subprocess.run([PY, "scripts/diag_report.py", "--out", args.out] +
                        (["--drive-dir", args.drive_dir] if args.drive_dir else []), cwd=str(ROOT)).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
