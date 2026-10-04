"""Shared helpers for scripts: import path, default locations, logging."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_DATA_DIR = os.environ.get("MOE_DATA_DIR", "/content/moe_data")
DEFAULT_OUT_DIR = os.environ.get("MOE_OUT_DIR", "/content/moe_fusion_runs")
DATA_REFERENCE = ROOT / "data_reference"

CORE_MODELS = ["A_serial", "B_parallel", "C_parallel_fusion_samewidth", "C_parallel_fusion_matched"]

# exit codes understood by run_pipeline.py
EXIT_OK = 0
EXIT_SCIENTIFIC_FAILURE = 3  # NaN / router collapse: a result, recorded, pipeline continues
EXIT_GATE_FAILED = 4  # a correctness gate failed: pipeline must stop


def banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}", flush=True)
