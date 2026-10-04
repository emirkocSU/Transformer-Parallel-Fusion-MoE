"""PHASE 0 - environment validation. Fails early on unsupported configurations.

Writes <out>/env/{environment.json, attention_backend.json, nvidia_smi.txt, pip_freeze.txt}
"""
import argparse
import json
import os
import subprocess
import sys

import _common  # noqa: F401
from _common import EXIT_GATE_FAILED, banner
from pathlib import Path

import torch

from moefusion.attention import select_attention_backend
from moefusion.utils import configure_torch_numerics, environment_info, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preferred-attention", default="sdpa_flash")
    ap.add_argument("--min-disk-gb", type=float, default=15.0)
    args = ap.parse_args()
    out = Path(args.out) / "env"
    out.mkdir(parents=True, exist_ok=True)
    banner("PHASE 0 - ENVIRONMENT VALIDATION")
    numerics = configure_torch_numerics()
    info = environment_info()
    info["numerics"] = numerics
    for k, v in info.items():
        print(f"  {k:28s} {v}")
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        smi = f"nvidia-smi failed: {e}"
    print(smi)
    (out / "nvidia_smi.txt").write_text(smi)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out / "pip_freeze.txt").write_text(freeze)

    errors, warnings = [], []
    cpu_dry_run = os.environ.get("MOE_ALLOW_CPU") == "1"  # developer plumbing test only, never used on Colab
    if not torch.cuda.is_available() and cpu_dry_run:
        warnings.append("CPU DRY RUN (MOE_ALLOW_CPU=1): numbers are meaningless; only the pipeline plumbing is tested")
        write_json(out / "attention_backend.json", {"selected": "sdpa", "fallback_used": False, "efficient_kernel": False})
        info["gpu_name"], info["gpu_total_memory_gb"] = "cpu", 0.0
    elif not torch.cuda.is_available():
        errors.append("CUDA is not available. Select Runtime > Change runtime type > A100 GPU.")
    else:
        if not torch.cuda.is_bf16_supported():
            errors.append("GPU does not support BF16 (needs Ampere or newer, e.g. A100).")
        if "A100" not in info.get("gpu_name", ""):
            warnings.append(f"GPU is {info.get('gpu_name')}, not an A100. Results remain internally fair "
                            "(all architectures run on the same GPU) but absolute speeds are not A100 numbers.")
    if info.get("disk_free_gb") is not None and info["disk_free_gb"] < args.min_disk_gb:
        errors.append(f"Only {info['disk_free_gb']} GB free disk (< {args.min_disk_gb} GB needed for checkpoints).")
    for mod in ("numpy", "yaml", "tokenizers", "matplotlib", "pytest"):
        try:
            __import__(mod)
        except Exception:
            errors.append(f"python package missing: {mod}")

    attn = None
    if torch.cuda.is_available() and not errors:
        attn = select_attention_backend(args.preferred_attention)
        print("  attention backend selection:", json.dumps(attn, indent=2))
        if attn["fallback_used"]:
            warnings.append(f"ATTENTION BACKEND FALLBACK: preferred {args.preferred_attention} unavailable, "
                            f"using {attn['selected']} for ALL architectures.")
        if not attn["efficient_kernel"]:
            warnings.append("Attention runs on the slow MATH kernel (no fused attention). Identical for all archs.")
        write_json(out / "attention_backend.json", attn)
    info["moe_backend"] = {"selected": "reference", "reason": "pre-registered correctness reference; no external MoE "
                           "package is installed (MegaBlocks/grouped-GEMM builds are fragile on Colab). The in-repo "
                           "padded_bmm backend is only benchmarked, never used for the primary comparison."}
    info["warnings"], info["errors"] = warnings, errors
    write_json(out / "environment.json", info)
    for w in warnings:
        print("  WARNING:", w)
    if errors:
        for e in errors:
            print("  ERROR:", e)
        sys.exit(EXIT_GATE_FAILED)
    print("  PHASE 0 PASSED")


if __name__ == "__main__":
    main()
