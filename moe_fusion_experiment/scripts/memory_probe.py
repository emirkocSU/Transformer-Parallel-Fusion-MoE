"""One memory-probe trial (run in a fresh subprocess so an OOM cannot contaminate later work).
Runs two full training steps (2 micro-batches each, clip, AdamW) and prints a JSON line."""
import argparse
import json

import _common  # noqa: F401

import torch

from moefusion.config import load_experiment
from moefusion.model import build_model, lm_loss, make_inputs_targets
from moefusion.trainer import autocast_ctx, make_optimizer
from moefusion.utils import configure_torch_numerics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--mb", type=int, required=True)
    ap.add_argument("--attention-backend", default="sdpa_flash")
    args = ap.parse_args()
    configure_torch_numerics()
    m, t = load_experiment(args.model, args.run, {"model": {"attention_backend": args.attention_backend},
                                                  "train": {"micro_batch_seqs": args.mb}})
    res = {"model": args.model, "mb": args.mb, "ok": False}
    try:
        dev = torch.device("cuda")
        model = build_model(m, 0, dev)
        opt = make_optimizer(model, t, dev)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            for _ in range(2):
                rows = torch.randint(0, m.vocab_size, (args.mb, t.seq_len + 1), device=dev)
                x, y = make_inputs_targets(rows, t.seq_len)
                with autocast_ctx(t, dev):
                    loss = lm_loss(model(x), y) + model.aux_penalty(t)
                (loss / 2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        res.update(ok=True, peak_alloc_gb=torch.cuda.max_memory_allocated() / 1e9,
                   peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                   total_gb=torch.cuda.get_device_properties(0).total_memory / 1e9)
    except torch.cuda.OutOfMemoryError as e:
        res["error"] = "OOM: " + str(e).split("\n")[0][:300]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            res["error"] = "OOM: " + str(e).split("\n")[0][:300]
        else:
            raise
    print("PROBE_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
