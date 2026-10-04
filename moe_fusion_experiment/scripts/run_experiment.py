"""Train ONE architecture under ONE run configuration (fresh process per run => identical process state).

Example:
  python scripts/run_experiment.py --model A_serial --run pilot --micro-batch 16 \
      --data-dir /content/moe_data --out /content/moe_fusion_runs/pilot/runs
"""
import argparse
import sys
from pathlib import Path

import _common  # noqa: F401
from _common import EXIT_SCIENTIFIC_FAILURE, ROOT, banner

import torch

from moefusion.config import load_experiment
from moefusion.data import TokenDataset
from moefusion.trainer import Trainer, TrainingFailure
from moefusion.utils import code_fingerprint, configure_torch_numerics, environment_info, git_commit, read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--micro-batch", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True, help="directory that will contain <model>/")
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--parallel-backend", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-final-weights", action="store_true")
    ap.add_argument("--dataset-report", default=None, help="dataset_verification.json to embed hashes in the manifest")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--ckpt-dir", default=None, help="parent dir for the resume checkpoint (default: the run dir)")
    ap.add_argument("--final-weights-dir", default=None, help="parent dir for model_final_bf16.pt (default: the run dir)")
    args = ap.parse_args()

    numerics = configure_torch_numerics()
    ov = {"model": {}, "train": {"micro_batch_seqs": args.micro_batch}}
    if args.attention_backend:
        ov["model"]["attention_backend"] = args.attention_backend
    if args.parallel_backend:
        ov["model"]["parallel_backend"] = args.parallel_backend
    if args.seed is not None:
        ov["train"]["seed"] = args.seed
    mcfg, tcfg = load_experiment(args.model, args.run, ov)
    run_name = args.run_name or f"{mcfg.name}_seed{tcfg.seed}"
    out = Path(args.out) / run_name
    banner(f"TRAIN {run_name}  ({args.run}: {tcfg.total_steps} steps x {tcfg.tokens_per_step:,} tokens = "
           f"{tcfg.total_tokens:,} tokens; micro-batch {tcfg.micro_batch_seqs} x accum {tcfg.grad_accum})")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = TokenDataset(args.data_dir, tcfg.seq_len, mcfg.vocab_size)
    extra = {"git_commit": git_commit(ROOT), "code_fingerprint": code_fingerprint(ROOT), "numerics": numerics,
             "environment": environment_info(), "data_dir": args.data_dir,
             "train_rows": int(data.train.shape[0]), "val_rows": int(data.val.shape[0]), "row_len": int(data.row_len)}
    if args.dataset_report and Path(args.dataset_report).exists():
        dr = read_json(args.dataset_report)
        extra["dataset_hashes"] = {s: dr.get("arrays", {}).get(s, {}).get("sha256_file") for s in ("train", "validation")}
        extra["tokenizer_sha256"] = dr.get("tokenizer", {}).get("sha256")
    tr = Trainer(mcfg, tcfg, data, out, dev, run_name, manifest_extra=extra,
                 save_final_weights=not args.no_final_weights,
                 ckpt_dir=Path(args.ckpt_dir) / run_name if args.ckpt_dir else None,
                 final_weights_dir=Path(args.final_weights_dir) / run_name if args.final_weights_dir else None)
    try:
        s = tr.run(resume=not args.no_resume)
    except TrainingFailure as e:
        print(f"[{run_name}] SCIENTIFIC FAILURE (recorded in failure.json): {e}", flush=True)
        sys.exit(EXIT_SCIENTIFIC_FAILURE)
    print(f"[{run_name}] DONE  final val loss {s['final_val_loss']:.4f}  ppl {s['final_val_ppl']:.2f}  "
          f"train time {s['train_seconds'] / 60:.1f} min  {s['tokens_per_sec_overall']:,.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
