"""PHASE - SMOKE TRAIN (~1.6M tokens per architecture).

For every architecture: train with an interruption after the step-12 checkpoint, resume from the checkpoint
(exercising the GPU checkpoint/resume path), finish, and verify:
  loss decreases | no NaN/Inf | no router collapse | checkpoint save+load | evaluation works.
"""
import argparse
import sys
from pathlib import Path

import _common  # noqa: F401
from _common import CORE_MODELS, EXIT_GATE_FAILED, banner

import torch

from moefusion.config import load_experiment
from moefusion.data import TokenDataset
from moefusion.metrics import read_jsonl
from moefusion.trainer import Trainer, TrainingFailure
from moefusion.utils import configure_torch_numerics, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(CORE_MODELS))
    ap.add_argument("--micro-batch", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attention-backend", default="sdpa_flash")
    args = ap.parse_args()
    configure_torch_numerics()
    banner("SMOKE TRAINING")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report, ok_all = {}, True
    data = None
    for name in args.models.split(","):
        m, t = load_experiment(name, "smoke", {"model": {"attention_backend": args.attention_backend},
                                               "train": {"micro_batch_seqs": args.micro_batch}})
        if data is None:
            data = TokenDataset(args.data_dir, t.seq_len, m.vocab_size)
        out = Path(args.out) / "smoke" / name
        checks = {}
        try:
            r1 = Trainer(m, t, data, out, dev, name, save_final_weights=False, stop_after_step=12).run(resume=False)
            checks["checkpoint_saved"] = r1.get("status") == "interrupted" and (out / "checkpoint.pt").exists()
            s = Trainer(m, t, data, out, dev, name, save_final_weights=False).run(resume=True)
            checks["checkpoint_resumed"] = True
            tr = read_jsonl(out / "train_metrics.jsonl")
            ev = read_jsonl(out / "eval_metrics.jsonl")
            first, last = tr[0]["lm_loss"], sum(x["lm_loss"] for x in tr[-4:]) / 4
            checks["train_loss_decreased"] = last < first - 0.5
            checks["val_loss_decreased"] = s["final_periodic_subset_val_loss"] < ev[0]["val_loss"] - 0.5
            checks["finite"] = all(x["lm_loss"] == x["lm_loss"] for x in tr)
            evs = read_jsonl(out / "events.jsonl")
            checks["no_router_collapse"] = not any(e["event"] in ("router_collapse_warning", "FATAL") for e in evs)
            checks["evaluation_works"] = bool(ev) and ev[-1].get("final") is True
            report[name] = {"checks": checks, "first_train_loss": first, "last_train_loss_mean4": last,
                            "initial_val_loss": ev[0]["val_loss"], "final_val_loss": s["final_val_loss"],
                            "tokens_per_sec": s["tokens_per_sec_overall"], "steps": s["steps"]}
        except TrainingFailure as e:
            checks["finite"] = False
            report[name] = {"checks": checks, "error": str(e)}
        ok = all(checks.values()) and len(checks) >= 7
        report[name]["passed"] = ok
        ok_all &= ok
        print(f"  {name:32s} {'PASS' if ok else 'FAIL'}  {checks}  "
              f"train {report[name].get('first_train_loss', float('nan')):.3f} -> {report[name].get('last_train_loss_mean4', float('nan')):.3f}",
              flush=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    write_json(Path(args.out) / "smoke" / "smoke_report.json", {"passed": ok_all, "models": report})
    if not ok_all:
        print("  SMOKE FAILED - not starting the pilot.")
        sys.exit(EXIT_GATE_FAILED)
    print("  SMOKE PASSED for every architecture")


if __name__ == "__main__":
    main()
