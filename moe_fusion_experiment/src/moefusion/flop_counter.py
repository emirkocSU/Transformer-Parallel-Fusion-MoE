"""Independent analytic parameter / FLOP accounting.

Computed from the config ALONE (no model instantiation) so that it can be cross-checked
against the parameter counts of the instantiated nn.Module (tests/test_parameter_budget.py).

Conventions (stated explicitly because FLOP numbers are convention-dependent):
  * a matmul of (m x n) by (n x p) costs 2*m*n*p FLOPs;
  * forward linear FLOPs/token = 2 * (active matmul parameters, incl. the LM head);
  * attention-score FLOPs/token/layer (QK^T and AV) = 4 * T * d_attn for full (non-causal)
    attention, as in the PaLM appendix-B convention; the causal-exact value is ~half of that;
  * training FLOPs = 3 * forward FLOPs (backward ~ 2x forward), as in Kaplan et al. / PaLM;
  * element-wise ops, norms, softmax, RoPE and routing bookkeeping are NOT counted.
"""
from __future__ import annotations

from typing import Dict

from .config import ModelConfig


def analytic_budget(cfg: ModelConfig, seq_len: int = None) -> Dict[str, float]:
    T = seq_len or cfg.max_seq_len
    d, V, E, k, h = cfg.d_model, cfg.vocab_size, cfg.n_experts, cfg.top_k, cfg.expert_hidden
    d_attn = cfg.n_heads * cfg.head_dim
    L, F = cfg.n_layers, cfg.n_fusion_layers
    n_moe = L + F

    emb = V * d
    lm_head_untied = 0 if cfg.tie_embeddings else V * d
    attn_per_layer = d * 3 * d_attn + d_attn * d
    attention = L * attn_per_layer
    router = n_moe * d * E
    expert_one = 3 * d * h
    experts = n_moe * E * expert_one
    norms = 2 * d * L + d * F + d  # block norms + fusion norms + final norm
    total = emb + lm_head_untied + attention + router + experts + norms

    active_experts = n_moe * k * expert_one
    active_matmul = attention + router + active_experts + V * d  # LM head always computed
    active_params_per_token = emb + lm_head_untied + attention + router + active_experts + norms

    fwd_linear = 2 * active_matmul
    attn_scores_full = 4 * T * d_attn * L
    attn_scores_causal = 4 * ((T + 1) / 2.0) * d_attn * L  # average key length (T+1)/2
    fwd_total_palm = fwd_linear + attn_scores_full
    train_per_token = 3 * fwd_total_palm

    return {
        "n_layers": L,
        "n_attention_layers": L,
        "n_moe_backbone": L,
        "n_moe_fusion": F,
        "n_moe_total": n_moe,
        "expert_hidden": h,
        "params_total": total,
        "params_embedding": emb,
        "params_lm_head_untied": lm_head_untied,
        "params_attention": attention,
        "params_router": router,
        "params_experts": experts,
        "params_norms": norms,
        "params_active_per_token": active_params_per_token,
        "params_active_nonembedding": active_params_per_token - emb,
        "flops_fwd_linear_per_token": fwd_linear,
        "flops_fwd_moe_per_token": 2 * active_experts,
        "flops_fwd_attn_scores_per_token_full": attn_scores_full,
        "flops_fwd_attn_scores_per_token_causal": attn_scores_causal,
        "flops_fwd_per_token": fwd_total_palm,
        "flops_train_per_token": train_per_token,
        "bf16_param_memory_gb": total * 2 / 1e9,
        "fp32_param_plus_adam_memory_gb": total * 16 / 1e9,  # fp32 weights + grads + 2 Adam moments
    }


def training_flops(cfg: ModelConfig, tokens: int, seq_len: int = None) -> float:
    return analytic_budget(cfg, seq_len)["flops_train_per_token"] * tokens


def matched_mismatch(baseline: ModelConfig, candidate: ModelConfig) -> Dict[str, float]:
    """Relative mismatch of expert parameters / active MoE FLOPs between two configs."""
    a, b = analytic_budget(baseline), analytic_budget(candidate)
    return {
        "expert_params_rel_diff": (b["params_experts"] - a["params_experts"]) / a["params_experts"],
        "active_moe_flops_rel_diff": (b["flops_fwd_moe_per_token"] - a["flops_fwd_moe_per_token"]) / a["flops_fwd_moe_per_token"],
        "total_params_rel_diff": (b["params_total"] - a["params_total"]) / a["params_total"],
        "train_flops_rel_diff": (b["flops_train_per_token"] - a["flops_train_per_token"]) / a["flops_train_per_token"],
    }


def matched_hidden(baseline_hidden: int, baseline_moe_count: int, c_moe_count: int, multiple: int = 64) -> Dict[str, float]:
    """General compute-matching rule for fusion ablations: baseline_count*baseline_w ~= c_count*c_w,
    rounded to a hardware-friendly multiple. Reports the exact resulting mismatch."""
    exact = baseline_hidden * baseline_moe_count / c_moe_count
    rounded = int(max(multiple, round(exact / multiple) * multiple))
    mismatch = (rounded * c_moe_count - baseline_hidden * baseline_moe_count) / (baseline_hidden * baseline_moe_count)
    return {"exact": exact, "rounded": rounded, "rel_mismatch": mismatch}
