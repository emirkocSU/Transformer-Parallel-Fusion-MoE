"""Training loop shared verbatim by every architecture.

Fairness properties enforced here (identical for A, B, C):
  * same number of optimizer steps, same global batch, same micro-batch, same data order
    (BatchSchedule is a pure function of (data_seed, step));
  * same optimizer (AdamW, fused on CUDA), same LR schedule, same clipping, same precision;
  * same evaluation points (by step) on the same fixed validation rows;
  * same per-step synchronisation/logging work, so bookkeeping overhead is identical;
  * evaluation, checkpointing and the untimed warm-up are EXCLUDED from `train_seconds`
    (the wall-clock axis), and reported separately.
"""
from __future__ import annotations

import contextlib
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig, TrainConfig, config_dict, config_hash, lr_at
from .data import BatchSchedule, TokenDataset, eval_rows, rows_to_tensor
from .flop_counter import analytic_budget
from .metrics import JsonlLogger, collapse_check, read_jsonl, utilization_stats
from .model import IGNORE_INDEX, MoEFusionLM, build_model, count_parameters, lm_loss, make_inputs_targets
from .utils import now_iso, write_json


class TrainingFailure(RuntimeError):
    pass


def autocast_ctx(tcfg: TrainConfig, device: torch.device):
    if tcfg.precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def make_optimizer(model: torch.nn.Module, tcfg: TrainConfig, device: torch.device) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": tcfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2), eps=tcfg.eps)
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


@torch.no_grad()
def evaluate(model: MoEFusionLM, arr: np.ndarray, rows: np.ndarray, micro_batch: int, tcfg: TrainConfig,
             device: torch.device, return_per_seq: bool = False) -> Dict[str, Any]:
    """Validation LM loss ONLY (no auxiliary router terms). perplexity = exp(token-mean loss)."""
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    tok_sum = torch.zeros((), dtype=torch.float64, device=device)
    per_seq = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(rows), micro_batch):
        batch = rows_to_tensor(arr, rows[i : i + micro_batch], device)
        x, y = make_inputs_targets(batch, tcfg.seq_len)
        with autocast_ctx(tcfg, device):
            logits = model(x)
        l = lm_loss(logits, y, reduction="none")  # (B, T) fp32, 0 at ignored positions
        valid = (y != IGNORE_INDEX).to(torch.float64)
        s = (l.double() * valid).sum(dim=1)
        c = valid.sum(dim=1)
        loss_sum += s.sum()
        tok_sum += c.sum()
        if return_per_seq:
            per_seq.append((s / c).float().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if was_training:
        model.train()
    mean = float(loss_sum / tok_sum)
    out = {"val_loss": mean, "val_ppl": math.exp(mean) if mean < 50 else float("inf"),
           "val_tokens": int(tok_sum.item()), "val_sequences": int(len(rows)), "val_seconds": dt}
    if return_per_seq:
        out["per_seq_loss"] = torch.cat(per_seq).numpy()
    return out


def _truncate_jsonl(path: Path, max_step: int) -> None:
    if not path.exists():
        return
    keep = [r for r in read_jsonl(path) if r.get("step", -1) <= max_step]
    import json

    with open(path, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")


def _step_stats(times) -> Dict[str, float]:
    if len(times) == 0:
        return {}
    a = np.asarray(times)
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "n": int(a.size)}


class Trainer:
    def __init__(self, mcfg: ModelConfig, tcfg: TrainConfig, data: TokenDataset, out_dir, device,
                 run_name: str, manifest_extra: Optional[Dict[str, Any]] = None, save_final_weights: bool = True,
                 keep_resume_checkpoint: bool = False, log_fn=print, stop_after_step: Optional[int] = None,
                 ckpt_dir=None, final_weights_dir=None, keep_final_state: bool = False):
        if tcfg.micro_batch_seqs is None:
            raise ValueError("micro_batch_seqs must be set (use the common memory probe)")
        self.mcfg, self.tcfg, self.data = mcfg, tcfg, data
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        # The resume checkpoint may live elsewhere (e.g. Google Drive) than the run's logs; a snapshot of the
        # logs is stored next to it so that a resume in a NEW session restores logs consistent with the checkpoint.
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else self.out
        self.final_weights_dir = Path(final_weights_dir) if final_weights_dir else self.out
        self.device = torch.device(device)
        self.run_name = run_name
        self.manifest_extra = manifest_extra or {}
        self.save_final_weights = save_final_weights
        self.keep_resume_checkpoint = keep_resume_checkpoint
        self.log = log_fn
        self.stop_after_step = stop_after_step  # testing hook: simulate an interruption after a checkpoint
        self.final_state_dict = None
        self.keep_final_state = keep_final_state  # tests only (a CPU copy of a 0.5B model costs ~2 GB RAM)
        self.cfg_dict = config_dict(mcfg, tcfg)
        self.cfg_hash = config_hash(self.cfg_dict)

    # ------------------------------------------------------------------
    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def run(self, resume: bool = True) -> Dict[str, Any]:
        mcfg, tcfg, dev = self.mcfg, self.tcfg, self.device
        start_iso = now_iso()
        t_wall0 = time.perf_counter()
        torch.manual_seed(tcfg.seed)
        np.random.seed(tcfg.seed % (2**32))

        model = build_model(mcfg, tcfg.seed, dev)
        model.train()
        opt = make_optimizer(model, tcfg, dev)
        params = count_parameters(model)
        budget = analytic_budget(mcfg, tcfg.seq_len)
        sched = BatchSchedule(len(self.data.train), tcfg.global_batch_seqs, tcfg.effective_data_seed)
        val_rows_periodic = eval_rows(len(self.data.val), tcfg.eval_seqs)
        val_rows_final = eval_rows(len(self.data.val), tcfg.final_eval_seqs)
        mb, accum = tcfg.micro_batch_seqs, tcfg.grad_accum
        targets_per_step = tcfg.global_batch_seqs * self.data.targets_per_row

        ckpt_path = self.ckpt_dir / "checkpoint.pt"
        snap_dir = self.ckpt_dir / "log_snapshot"
        train_log_path = self.out / "train_metrics.jsonl"
        eval_log_path = self.out / "eval_metrics.jsonl"
        events_path = self.out / "events.jsonl"

        state = {"train_seconds": 0.0, "tokens": 0, "eval_seconds": 0.0, "ckpt_seconds": 0.0,
                 "step_times": [], "data_wait": [], "collapse_streak": 0, "segments": []}
        start_step = 0
        if resume and ckpt_path.exists():
            payload = load_checkpoint(ckpt_path, model, opt, strict_config=self.cfg_dict)
            del payload["model"], payload["optimizer"]  # free the CPU copy (several GB) right away
            start_step = payload["step"]
            state.update(payload["state"])
            if snap_dir.exists() and snap_dir.resolve() != self.out.resolve():
                for p in (train_log_path, eval_log_path, events_path):
                    if (snap_dir / p.name).exists():
                        shutil.copy2(snap_dir / p.name, p)
            for p in (train_log_path, eval_log_path, events_path):
                _truncate_jsonl(p, start_step)
            self.log(f"[{self.run_name}] resumed from step {start_step}")
        else:  # fresh start: remove logs of any earlier (incomplete) attempt so nothing stale is appended
            for p in (train_log_path, eval_log_path, events_path, self.out / "failure.json"):
                if p.exists():
                    p.unlink()
            if snap_dir.exists():
                shutil.rmtree(snap_dir)
        state["segments"].append({"start_step": start_step, "start_time": start_iso,
                                  "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"})

        tlog, elog, evlog = JsonlLogger(train_log_path), JsonlLogger(eval_log_path), JsonlLogger(events_path)

        # ---- untimed warm-up: forward/backward on validation rows, gradients discarded.
        # Parameters, optimizer state and RNG are untouched; it only warms kernels/allocator.
        for _ in range(tcfg.untimed_warmup_microbatches):
            batch = rows_to_tensor(self.data.val, np.arange(min(mb, len(self.data.val))), dev)
            x, y = make_inputs_targets(batch, tcfg.seq_len)
            with autocast_ctx(tcfg, dev):
                out = model(x)
                l = lm_loss(out, y) + model.aux_penalty(tcfg)
            l.backward()
            opt.zero_grad(set_to_none=True)
            del out, l, batch, x, y
        self._sync()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        def do_eval(step_done: int, final: bool = False):
            rows = val_rows_final if final else val_rows_periodic
            r = evaluate(model, self.data.val, rows, mb, tcfg, dev, return_per_seq=final)
            state["eval_seconds"] += r["val_seconds"]
            rec = {"step": step_done, "tokens": state["tokens"], "train_seconds": state["train_seconds"],
                   "elapsed_seconds": time.perf_counter() - t_wall0, "final": final,
                   **{k: v for k, v in r.items() if k != "per_seq_loss"}}
            elog.log(rec)
            if final:
                np.save(self.out / "final_val_per_sequence_loss.npy", r["per_seq_loss"])
            self.log(f"[{self.run_name}] {'FINAL ' if final else ''}eval step {step_done}: "
                     f"val_loss={r['val_loss']:.4f} ppl={r['val_ppl']:.2f} ({r['val_sequences']} seqs, {r['val_seconds']:.1f}s)")
            return r

        if start_step == 0:
            do_eval(0)

        n_moe = len(model.moe_modules())
        E = mcfg.n_experts
        interval_counts = torch.zeros(n_moe, E, dtype=torch.long, device=dev)
        interval_entropy = torch.zeros(n_moe, dtype=torch.float32, device=dev)
        interval_n = 0
        imbalance_flag = False
        dead_flag = False

        for step in range(start_step, tcfg.total_steps):
            self._sync()
            t0 = time.perf_counter()
            rows_all = sched.rows_for_step(step)
            lm_acc = torch.zeros((), device=dev)
            bal_acc = torch.zeros((), device=dev)
            z_acc = torch.zeros((), device=dev)
            pen_acc = torch.zeros((), device=dev)
            data_wait = 0.0
            for j in range(accum):
                td = time.perf_counter()
                batch = rows_to_tensor(self.data.train, rows_all[j * mb : (j + 1) * mb], dev)
                data_wait += time.perf_counter() - td
                x, y = make_inputs_targets(batch, tcfg.seq_len)
                with autocast_ctx(tcfg, dev):
                    logits = model(x)
                    lm = lm_loss(logits, y)
                    aux = model.aux_losses()
                    aux_pen = model.aux_penalty(tcfg)
                    loss = lm + aux_pen
                (loss / accum).backward()
                lm_acc += lm.detach() / accum
                bal_acc += aux["balance_loss"].detach() / accum
                pen_acc += aux_pen.detach() / accum
                z_acc += aux["z_loss"].detach() / accum
                rs = model.routing_stats()
                interval_counts += rs["counts"]
                interval_entropy += rs["entropy"]
                interval_n += 1
                del logits, lm, aux, aux_pen, loss, batch, x, y
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            lm_v, bal_v, z_v, pen_v, gn_v = torch.stack([lm_acc, bal_acc, z_acc, pen_acc, grad_norm.float()]).tolist()
            if not (math.isfinite(lm_v) and math.isfinite(gn_v) and math.isfinite(bal_v)):
                self._fail(evlog, step, f"non-finite value: lm={lm_v} grad_norm={gn_v} balance={bal_v}", model)
            lr = lr_at(step, tcfg)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
            self._sync()
            step_time = time.perf_counter() - t0

            state["train_seconds"] += step_time
            state["tokens"] += tcfg.tokens_per_step
            state["step_times"].append(step_time)
            state["data_wait"].append(data_wait)
            step_done = step + 1
            rec = {"step": step_done, "tokens": state["tokens"], "train_seconds": state["train_seconds"],
                   "lm_loss": lm_v, "total_loss": lm_v + pen_v, "aux_penalty": pen_v,
                   "balance_loss": bal_v, "z_loss": z_v, "lr": lr, "grad_norm": gn_v, "step_time": step_time,
                   "tokens_per_sec": tcfg.tokens_per_step / step_time, "data_wait": data_wait,
                   "epoch": sched.epoch_of_step(step_done)}
            if dev.type == "cuda":
                rec["peak_mem_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
                rec["peak_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9

            if step_done % tcfg.log_every == 0 or step_done == tcfg.total_steps:
                counts_np = interval_counts.cpu().numpy()
                us = utilization_stats(counts_np, mcfg.top_k)
                rec.update(us)
                rec["router_entropy"] = float((interval_entropy / max(interval_n, 1)).mean().item())
                rec["router_entropy_max"] = math.log(E)
                interval_counts.zero_()
                interval_entropy.zero_()
                interval_n = 0
                # ---- router collapse guard (recorded; stops only if persistent) ----
                if step_done > tcfg.warmup_steps:
                    collapsed = collapse_check(np.asarray(us["per_layer_fraction"]), mcfg.top_k, tcfg.collapse_min_fraction)
                    if collapsed:
                        state["collapse_streak"] += 1
                        evlog.log({"step": step_done, "event": "router_collapse_warning", "layers": collapsed,
                                   "layer_names": [model.moe_names()[i] for i in collapsed],
                                   "fractions": [us["per_layer_fraction"][i] for i in collapsed],
                                   "streak": state["collapse_streak"]})
                        self.log(f"[{self.run_name}] ROUTER COLLAPSE WARNING step {step_done}: "
                                 + "; ".join(f"{model.moe_names()[i]} {[round(v, 3) for v in us['per_layer_fraction'][i]]}"
                                             for i in collapsed) + f" (streak {state['collapse_streak']}/{tcfg.collapse_patience})")
                        if state["collapse_streak"] >= tcfg.collapse_patience:
                            self._fail(evlog, step, f"router collapse persisted in layers {collapsed}", model)
                    else:
                        state["collapse_streak"] = 0
                    severe = us["util_maxmin_ratio_max"] > tcfg.imbalance_alert_ratio
                    if severe != imbalance_flag:
                        evlog.log({"step": step_done, "event": "expert_imbalance_alert" if severe else "expert_imbalance_cleared",
                                   "maxmin_ratio": us["util_maxmin_ratio_max"]})
                        imbalance_flag = severe
                # ---- dead/starved-expert monitor (all steps, monitoring only) ----
                frac = np.asarray(us["per_layer_fraction"])
                dead_layers = [int(i) for i in np.nonzero((frac < tcfg.dead_expert_fraction).any(axis=1))[0]]
                rec["dead_expert_layers"] = len(dead_layers)
                if bool(dead_layers) != dead_flag:
                    evlog.log({"step": step_done, "event": "dead_expert_alert" if dead_layers else "dead_expert_cleared",
                               "layers": dead_layers, "layer_names": [model.moe_names()[i] for i in dead_layers],
                               "min_fraction": float(frac.min())})
                    dead_flag = bool(dead_layers)
                self.log(f"[{self.run_name}] step {step_done}/{tcfg.total_steps} lm={lm_v:.4f} bal={bal_v:.4f} "
                         f"gn={gn_v:.3f} lr={lr:.2e} {rec['tokens_per_sec']:,.0f} tok/s "
                         f"cv={us['util_cv_mean']:.3f} H={rec['router_entropy']:.3f} dead_layers={len(dead_layers)}")
            tlog.log(rec)

            last = step_done == tcfg.total_steps
            if (step_done % tcfg.eval_every == 0) and not last:
                do_eval(step_done)
            if tcfg.ckpt_every and step_done % tcfg.ckpt_every == 0 and not last:
                tc = time.perf_counter()
                ci = save_checkpoint(ckpt_path, model, opt, step_done, state, self.cfg_dict)
                if ci["saved"] and snap_dir.resolve() != self.out.resolve():
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    for p in (train_log_path, eval_log_path, events_path):
                        if p.exists():
                            shutil.copy2(p, snap_dir / p.name)
                state["ckpt_seconds"] += time.perf_counter() - tc
                evlog.log({"step": step_done, "event": "checkpoint_saved" if ci["saved"] else "checkpoint_SKIPPED",
                           "path": str(ckpt_path), "seconds": time.perf_counter() - tc,
                           **{k: v for k, v in ci.items() if k != "saved"}})
                self.log(f"[{self.run_name}] checkpoint {'saved' if ci['saved'] else 'SKIPPED'} at step {step_done} "
                         f"({ci['bytes_estimate'] / 1e9:.2f} GB -> {ckpt_path.parent}) {ci.get('reason', '')}")
                if self.stop_after_step is not None and step_done >= self.stop_after_step:
                    tlog.close(), elog.close(), evlog.close()
                    return {"status": "interrupted", "step": step_done}

        # ---- final evaluation on the full (or configured) validation set + periodic-subset point
        periodic_final = do_eval(tcfg.total_steps)
        final = do_eval(tcfg.total_steps, final=True)
        tlog.close(), elog.close(), evlog.close()

        if self.keep_final_state:
            self.final_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if self.save_final_weights:
            sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
            self.final_weights_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": sd, "config": self.cfg_dict}, self.final_weights_dir / "model_final_bf16.pt")
            del sd
        if not self.keep_resume_checkpoint:
            if ckpt_path.exists():
                ckpt_path.unlink()
            if snap_dir.exists() and snap_dir.resolve() != self.out.resolve():
                shutil.rmtree(snap_dir)

        st = state["step_times"]
        summary = {
            "run_name": self.run_name,
            "status": "completed",
            "config": self.cfg_dict,
            "config_hash": self.cfg_hash,
            "steps": tcfg.total_steps,
            "tokens_input": state["tokens"],
            "targets_trained": tcfg.total_steps * targets_per_step,
            "micro_batch_seqs": mb,
            "grad_accum": accum,
            "train_seconds": state["train_seconds"],
            "eval_seconds": state["eval_seconds"],
            "ckpt_seconds": state["ckpt_seconds"],
            "tokens_per_sec_overall": state["tokens"] / max(state["train_seconds"], 1e-9),
            "step_time_all": _step_stats(st),
            "step_time_after10": _step_stats(st[10:]),
            "data_wait": _step_stats(state["data_wait"]),
            "peak_mem_alloc_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
            "peak_mem_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if dev.type == "cuda" else None,
            "final_val_loss": final["val_loss"],
            "final_val_ppl": final["val_ppl"],
            "final_val_tokens": final["val_tokens"],
            "final_val_sequences": final["val_sequences"],
            "final_periodic_subset_val_loss": periodic_final["val_loss"],
            "parameters_measured": params,
            "budget_analytic": budget,
            "segments": state["segments"],
            "start_time": state["segments"][0]["start_time"],
            "end_time": now_iso(),
        }
        write_json(self.out / "summary.json", summary)
        manifest = {
            "architecture": mcfg.arch, "name": mcfg.name, "seed": tcfg.seed, "data_seed": tcfg.effective_data_seed,
            "config": self.cfg_dict, "config_hash": self.cfg_hash,
            "attention_backend": mcfg.attention_backend, "moe_backend": mcfg.moe_backend,
            "parallel_execution_backend": mcfg.parallel_backend if mcfg.arch != "serial" else "n/a (serial)",
            "parameter_counts": params, "flop_estimates": budget,
            "batch": {"global_batch_seqs": tcfg.global_batch_seqs, "micro_batch_seqs": mb, "grad_accum": accum,
                      "seq_len": tcfg.seq_len, "tokens_per_step": tcfg.tokens_per_step},
            "optimizer": {"name": "AdamW", "fused": dev.type == "cuda", "betas": [tcfg.beta1, tcfg.beta2], "eps": tcfg.eps,
                          "weight_decay": tcfg.weight_decay, "grad_clip": tcfg.grad_clip},
            "scheduler": {"type": "linear_warmup_cosine", "warmup_steps": tcfg.warmup_steps, "lr": tcfg.lr,
                          "min_lr": tcfg.lr * tcfg.min_lr_ratio},
            "tokens": state["tokens"], "start_time": summary["start_time"], "end_time": summary["end_time"],
            "checkpoint_paths": {"final_weights": str(self.final_weights_dir / "model_final_bf16.pt") if self.save_final_weights else None,
                                 "resume_checkpoint_dir": str(self.ckpt_dir)},
            **self.manifest_extra,
        }
        write_json(self.out / "experiment_manifest.json", manifest)
        del model, opt
        return summary

    def _fail(self, evlog: JsonlLogger, step: int, reason: str, model) -> None:
        evlog.log({"step": step + 1, "event": "FATAL", "reason": reason})
        diag = {"run_name": self.run_name, "status": "failed", "step": step + 1, "reason": reason,
                "config": self.cfg_dict, "time": now_iso()}
        try:
            diag["routing"] = {k: v.cpu().tolist() for k, v in model.routing_stats().items()}
        except Exception:  # noqa: BLE001
            pass
        write_json(self.out / "failure.json", diag)
        raise TrainingFailure(reason)
