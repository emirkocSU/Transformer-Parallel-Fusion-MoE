"""Top-k token-choice Mixture-of-Experts FFN with SwiGLU experts. NO token dropping.

Routing (Shazeer et al. 2017; Fedus et al. 2022; Jiang et al. 2024 / Mixtral):
    logits = W_r x                      (computed in FP32, autocast disabled)
    p      = softmax(logits)
    (w, e) = TopK(p, k)                 every token is processed by exactly k experts
    w      = w / sum(w)                 renormalised over the selected experts (Mixtral)
    y      = sum_j w_j * Expert_{e_j}(x)

Load balancing auxiliary loss (Switch Transformer, Fedus et al. 2022, eq. 4-6, generalised to top-k):
    f_i = (#assignments to expert i) / (N * k)
    P_i = mean_tokens p_i
    L_balance = E * sum_i f_i * P_i      (= 1.0 for perfectly uniform routing)
Router z-loss (ST-MoE, Zoph et al. 2022): mean(logsumexp(logits)^2) -- always LOGGED; only added to
the objective if zloss_coef > 0 (default 0).

Backends (identical math; one is chosen per experiment for ALL architectures):
    reference   sort tokens by expert, one GEMM triple per expert (correctness reference)
    padded_bmm  scatter into an (E, max_count, d) buffer sized by the LARGEST expert load
                (capacity = max load, so no token can ever be dropped) and run 3 batched GEMMs
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLUExperts(nn.Module):
    """E independent SwiGLU FFNs stored as stacked tensors (Shazeer, 2020)."""

    def __init__(self, n_experts: int, d_model: int, hidden: int):
        super().__init__()
        self.n_experts, self.d_model, self.hidden = n_experts, d_model, hidden
        # Zeros, never torch.empty: uninitialised memory can contain NaN/Inf. The experiment's real initialisation
        # is moefusion.model.init_parameters (name-keyed, identical across architectures), which overwrites these.
        self.w1 = nn.Parameter(torch.zeros(n_experts, d_model, hidden))  # gate
        self.w3 = nn.Parameter(torch.zeros(n_experts, d_model, hidden))  # up
        self.w2 = nn.Parameter(torch.zeros(n_experts, hidden, d_model))  # down (residual output)

    def forward_single(self, e: int, x: torch.Tensor) -> torch.Tensor:
        return (F.silu(x @ self.w1[e]) * (x @ self.w3[e])) @ self.w2[e]


class MoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int, hidden: int, backend: str = "reference"):
        super().__init__()
        self.d_model, self.n_experts, self.top_k, self.hidden = d_model, n_experts, top_k, hidden
        self.backend = backend
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = SwiGLUExperts(n_experts, d_model, hidden)
        self.last_aux: Dict[str, torch.Tensor] = {}
        self.last_routing: Dict[str, torch.Tensor] = {}
        self.keep_routing = False  # tests set this to inspect assignments

    # ------------------------------------------------------------------ routing
    def route(self, xf: torch.Tensor):
        with torch.autocast(device_type=xf.device.type, enabled=False):
            logits = F.linear(xf.float(), self.router.weight.float())  # (N, E) fp32
        probs = logits.softmax(dim=-1)
        topw, topi = probs.topk(self.top_k, dim=-1)  # (N, k)
        topw = topw / topw.sum(dim=-1, keepdim=True)
        return logits, probs, topw, topi

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        N, k, E = xf.shape[0], self.top_k, self.n_experts
        logits, probs, topw, topi = self.route(xf)

        flat_e = topi.reshape(-1)  # (N*k,) expert id of every assignment
        order = torch.argsort(flat_e, stable=True)  # assignments grouped by expert
        tok_idx = order // k  # token id of every sorted assignment
        counts = torch.bincount(flat_e, minlength=E)
        counts_list: List[int] = counts.tolist()  # the single host sync of this layer

        x_sorted = xf[tok_idx]  # (N*k, d)
        if self.backend == "reference":
            outs = []
            start = 0
            for e, c in enumerate(counts_list):
                if c == 0:
                    continue
                outs.append(self.experts.forward_single(e, x_sorted[start : start + c]))
                start += c
            y_sorted = torch.cat(outs, dim=0)
        elif self.backend == "padded_bmm":
            e_sorted = flat_e[order]
            offsets = torch.cumsum(counts, 0) - counts
            pos = torch.arange(N * k, device=xf.device) - offsets[e_sorted]
            cap = max(counts_list)
            x_pad = x_sorted.new_zeros(E, cap, shape[-1]).index_put((e_sorted, pos), x_sorted)
            h = F.silu(torch.bmm(x_pad, self.experts.w1)) * torch.bmm(x_pad, self.experts.w3)
            y_pad = torch.bmm(h, self.experts.w2)
            y_sorted = y_pad[e_sorted, pos]
        else:  # pragma: no cover
            raise ValueError(self.backend)

        # Weighted combination in FP32, deterministic (no atomics): un-permute then sum the k slots.
        w_sorted = topw.reshape(-1)[order]
        y_weighted = y_sorted.float() * w_sorted.unsqueeze(-1)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        y = y_weighted[inv].view(N, k, shape[-1]).sum(dim=1)

        # ---- auxiliary losses and diagnostics (losses keep their graph; diagnostics are detached)
        f = counts.float() / float(N * k)
        P = probs.mean(dim=0)
        balance = E * torch.sum(f * P)
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
        self.last_aux = {
            "balance_loss": balance,
            "z_loss": z,
            "counts": counts.detach(),
            "prob_mean": P.detach(),
            "entropy": entropy.detach(),
        }
        if self.keep_routing:
            self.last_routing = {"topi": topi.detach(), "topw": topw.detach(), "probs": probs.detach()}
        return y.view(shape).to(x.dtype) if x.dtype != torch.float32 else y.view(shape)

    def aux_tensors(self) -> List[torch.Tensor]:
        return list(self.last_aux.values())
