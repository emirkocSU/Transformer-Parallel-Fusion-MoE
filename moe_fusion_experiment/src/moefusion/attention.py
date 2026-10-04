"""Ordinary causal multi-head self-attention (identical for A, B and C).

Backends (chosen ONCE per experiment and shared by every architecture):
  sdpa_flash      torch SDPA restricted to the FlashAttention-2 kernel (fails loudly if unavailable)
  sdpa_efficient  torch SDPA restricted to the memory-efficient kernel
  sdpa_math       torch SDPA restricted to the math (reference) kernel -- slow, visible warning
  sdpa            torch SDPA automatic dispatch (kernel not pinned)
  flash_attn      the optional `flash_attn` package (FlashAttention-2)
On CPU every backend uses plain SDPA (only used by unit tests).
"""
from __future__ import annotations

import contextlib
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .rope import RotaryEmbedding

try:  # torch >= 2.3
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except Exception:  # pragma: no cover
    _HAS_SDPA_KERNEL = False


def _sdpa_context(backend: str):
    if not _HAS_SDPA_KERNEL or backend == "sdpa":
        return contextlib.nullcontext()
    mapping = {
        "sdpa_flash": SDPBackend.FLASH_ATTENTION,
        "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "sdpa_math": SDPBackend.MATH,
    }
    return sdpa_kernel([mapping[backend]])


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str) -> torch.Tensor:
    """q, k, v: (B, H, T, hd) -> (B, H, T, hd)."""
    if not q.is_cuda:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if backend == "flash_attn":
        from flash_attn import flash_attn_func  # type: ignore

        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True)
        return out.transpose(1, 2)
    with _sdpa_context(backend):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, max_seq_len: int, rope_theta: float, backend: str):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.backend = backend
        self.qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.rope = RotaryEmbedding(head_dim, max_seq_len, rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self.rope(q, k)
        y = causal_attention(q, k, v, self.backend)
        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.proj(y)


def probe_attention_backend(backend: str, device: str = "cuda") -> Dict[str, object]:
    """Run one tiny BF16 causal attention with the requested backend. Never raises."""
    info: Dict[str, object] = {"requested": backend, "ok": False, "error": None}
    try:
        q = torch.randn(2, 4, 128, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        out = causal_attention(q, k, v, backend)
        out.float().sum().backward()
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
        info["max_abs_err_vs_fp32"] = float((out.float() - ref).abs().max())
        info["ok"] = bool(torch.isfinite(out).all()) and info["max_abs_err_vs_fp32"] < 5e-2
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def select_attention_backend(preferred: str = "sdpa_flash", device: str = "cuda") -> Dict[str, object]:
    """Pick ONE backend for the whole experiment. Falls back visibly, never silently."""
    order = [preferred] + [b for b in ("sdpa_flash", "flash_attn", "sdpa_efficient", "sdpa_math") if b != preferred]
    tried = []
    for b in order:
        if b == "flash_attn":
            try:
                import flash_attn  # type: ignore  # noqa: F401
            except Exception:
                tried.append({"requested": b, "ok": False, "error": "flash_attn package not installed"})
                continue
        r = probe_attention_backend(b, device)
        tried.append(r)
        if r["ok"]:
            return {
                "selected": b,
                "fallback_used": b != preferred,
                "efficient_kernel": b in ("sdpa_flash", "flash_attn", "sdpa_efficient"),
                "attempts": tried,
            }
    raise RuntimeError(f"No working attention backend: {tried}")
