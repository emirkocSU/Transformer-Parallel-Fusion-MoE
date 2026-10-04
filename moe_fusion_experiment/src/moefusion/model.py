"""Decoder-only causal LM shared by architectures A, B and C."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import FusionBlock, ParallelBlock, SerialBlock
from .config import ModelConfig
from .moe import MoE
from .rmsnorm import RMSNorm
from .utils import stable_seed

IGNORE_INDEX = -100


class MoEFusionLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        block_cls = SerialBlock if cfg.arch == "serial" else ParallelBlock
        self.blocks = nn.ModuleList([block_cls(cfg) for _ in range(cfg.n_layers)])
        self.fusions = nn.ModuleDict({str(i): FusionBlock(cfg) for i in cfg.fusion_positions()})
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    # ------------------------------------------------------------------
    def moe_modules(self) -> List[MoE]:
        """All MoE modules in execution order (backbone + fusion)."""
        mods: List[MoE] = []
        for i, b in enumerate(self.blocks):
            mods.append(b.moe)
            if str(i) in self.fusions:
                mods.append(self.fusions[str(i)].moe)
        return mods

    def moe_names(self) -> List[str]:
        names = []
        for i in range(len(self.blocks)):
            names.append(f"block{i}")
            if str(i) in self.fusions:
                names.append(f"fusion_after_block{i}")
        return names

    def set_parallel_backend(self, backend: str) -> None:
        for b in self.blocks:
            if isinstance(b, ParallelBlock):
                b.execution = backend
        self.cfg.parallel_backend = backend

    # ------------------------------------------------------------------
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            key = str(i)
            if key in self.fusions:
                x = self.fusions[key](x)
        return self.lm_head(self.norm_f(x))

    def aux_losses(self) -> Dict[str, torch.Tensor]:
        """Per-router auxiliary losses of the last forward pass: the MEAN over MoE applications (for logging;
        1.0 = perfectly balanced) and the SUM over MoE applications (used in the objective, Switch convention)."""
        mods = self.moe_modules()
        bal = torch.stack([m.last_aux["balance_loss"] for m in mods])
        z = torch.stack([m.last_aux["z_loss"] for m in mods])
        return {"balance_loss": bal.mean(), "z_loss": z.mean(), "balance_loss_sum": bal.sum(), "z_loss_sum": z.sum()}

    def aux_penalty(self, tcfg) -> torch.Tensor:
        """Auxiliary term added to the LM loss. The single place where the objective's router terms are defined."""
        a = self.aux_losses()
        if tcfg.aux_loss_reduction == "sum":
            return tcfg.balance_coef * a["balance_loss_sum"] + tcfg.zloss_coef * a["z_loss_sum"]
        return tcfg.balance_coef * a["balance_loss"] + tcfg.zloss_coef * a["z_loss"]

    def routing_stats(self) -> Dict[str, torch.Tensor]:
        mods = self.moe_modules()
        return {
            "counts": torch.stack([m.last_aux["counts"] for m in mods]),  # (n_moe, E)
            "prob_mean": torch.stack([m.last_aux["prob_mean"] for m in mods]),
            "entropy": torch.stack([m.last_aux["entropy"] for m in mods]),
        }


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Next-token cross entropy in FP32. `targets` are ALREADY aligned with `logits`
    (targets[:, t] is the token that follows inputs[:, t]); IGNORE_INDEX positions are skipped."""
    V = logits.shape[-1]
    loss = F.cross_entropy(logits.float().view(-1, V), targets.reshape(-1), ignore_index=IGNORE_INDEX, reduction=reduction)
    if reduction == "none":
        return loss.view(targets.shape)
    return loss


def make_inputs_targets(rows: torch.Tensor, seq_len: int):
    """rows: (B, seq_len+1) -> inputs (B, seq_len), targets (B, seq_len)      [preferred layout]
       rows: (B, seq_len)   -> inputs rows, targets rows shifted left, last position ignored."""
    if rows.shape[1] == seq_len + 1:
        return rows[:, :-1], rows[:, 1:]
    if rows.shape[1] == seq_len:
        tgt = torch.full_like(rows, IGNORE_INDEX)
        tgt[:, :-1] = rows[:, 1:]
        return rows, tgt
    raise ValueError(f"row length {rows.shape[1]} incompatible with seq_len {seq_len}")


# ----------------------------------------------------------------------------
# Name-keyed deterministic initialisation.
#
# Every parameter is initialised from its own generator seeded by (seed, parameter name).
# Consequence: any two architectures that contain a parameter with the same name and shape
# receive BIT-IDENTICAL initial values (embeddings, attention, routers, backbone experts, norms).
# A and B are identical at initialisation; C(same-width) additionally has the fusion modules.
# Differently shaped tensors (C-matched experts) are drawn independently (no nonsensical copying).
# ----------------------------------------------------------------------------
def init_parameters(model: MoEFusionLM, seed: int) -> Dict[str, str]:
    cfg = model.cfg
    std = cfg.init_std
    # GPT-2 style scaled init of residual-output projections, using the BACKBONE depth for every
    # architecture (identical formula for A, B, C; the fusion modules use the same value).
    resid_std = std / math.sqrt(2 * cfg.n_layers)
    report = {}
    with torch.no_grad():
        for name, p in model.named_parameters():  # tied weights appear once (tok_emb.weight)
            g = torch.Generator(device="cpu").manual_seed(stable_seed("init", seed, name, tuple(p.shape)))
            if p.ndim == 1:
                val = torch.ones(p.shape)
                rule = "ones"
            elif name.endswith("attn.proj.weight") or name.endswith("experts.w2"):
                val = torch.empty(p.shape).normal_(0.0, resid_std, generator=g)
                rule = f"normal(0,{resid_std:.6f})"
            else:
                val = torch.empty(p.shape).normal_(0.0, std, generator=g)
                rule = f"normal(0,{std})"
            p.copy_(val.to(p.dtype))
            report[name] = rule
    return report


def build_model(cfg: ModelConfig, seed: int, device="cpu") -> MoEFusionLM:
    model = MoEFusionLM(cfg)
    init_parameters(model, seed)
    return model.to(device)


def count_parameters(model: MoEFusionLM) -> Dict[str, int]:
    out = {"total": 0, "trainable": 0, "embedding": 0, "lm_head_untied": 0, "attention": 0, "router": 0,
           "experts_backbone": 0, "experts_fusion": 0, "norms": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        out["total"] += n
        if p.requires_grad:
            out["trainable"] += n
        if name.startswith("tok_emb"):
            out["embedding"] += n
        elif name.startswith("lm_head"):
            out["lm_head_untied"] += n
        elif ".attn." in name:
            out["attention"] += n
        elif ".router." in name:
            out["router"] += n
        elif ".experts." in name:
            out["experts_fusion" if name.startswith("fusions.") else "experts_backbone"] += n
        elif p.ndim == 1:
            out["norms"] += n
        else:  # pragma: no cover
            raise RuntimeError(f"unclassified parameter {name}")
    out["experts"] = out["experts_backbone"] + out["experts_fusion"]
    return out
