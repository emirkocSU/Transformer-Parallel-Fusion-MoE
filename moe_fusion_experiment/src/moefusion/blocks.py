"""Transformer blocks for the three architectures.

SerialBlock and ParallelBlock own EXACTLY the same sub-modules with EXACTLY the same names
(norm_attn, attn, norm_moe, moe). Architectures A and B therefore have identical parameter sets
(identical state_dict keys/shapes); only the forward wiring differs.

A  SerialBlock     a = x + Attn(N_a(x));  y = a + MoE(N_m(a))
B  ParallelBlock   y = x + Attn(N_a(x)) + MoE(N_m(x))
C  ParallelBlock (+ FusionBlock after every N blocks:  y = x + FusionMoE(N_f(x)))

ParallelBlock execution backends (mathematically identical):
  reference   ordinary sequential dispatch on the current CUDA stream.
              This is NOT hardware parallelism.
  concurrent  attention branch on CUDA stream S_a, MoE branch on CUDA stream S_m, joined with
              stream waits; record_stream() protects cross-stream tensor lifetimes. Autograd
              runs each backward op on the stream of its forward op. Whether kernels ACTUALLY
              overlap must be measured (scripts/profile_model.py), never assumed.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .attention import CausalSelfAttention
from .moe import MoE
from .rmsnorm import RMSNorm
from .utils import prange


def _make_attn(cfg) -> CausalSelfAttention:
    return CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.attention_backend)


def _make_moe(cfg, hidden: int) -> MoE:
    return MoE(cfg.d_model, cfg.n_experts, cfg.top_k, hidden, cfg.moe_backend)


class _AttnMoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = _make_attn(cfg)
        self.norm_moe = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.moe = _make_moe(cfg, cfg.expert_hidden)


class SerialBlock(_AttnMoEBlock):
    kind = "serial"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with prange("serial_attention"):
            a = x + self.attn(self.norm_attn(x))
        with prange("serial_moe"):
            return a + self.moe(self.norm_moe(a))


# Streams are cached per device and shared by all ParallelBlocks of a process.
_STREAMS: Dict[int, Tuple["torch.cuda.Stream", "torch.cuda.Stream"]] = {}


def _branch_streams(device: torch.device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _STREAMS:
        _STREAMS[idx] = (torch.cuda.Stream(device=idx), torch.cuda.Stream(device=idx))
    return _STREAMS[idx]


class ParallelBlock(_AttnMoEBlock):
    kind = "parallel"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.execution = cfg.parallel_backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.execution == "concurrent" and x.is_cuda:
            return self._forward_concurrent(x)
        return self._forward_reference(x)

    def _forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        with prange("parallel_attention_branch"):
            a = self.attn(self.norm_attn(x))
        with prange("parallel_moe_branch"):
            m = self.moe(self.norm_moe(x))
        with prange("parallel_merge"):
            return x + a + m

    def _forward_concurrent(self, x: torch.Tensor) -> torch.Tensor:
        cur = torch.cuda.current_stream(x.device)
        s_attn, s_moe = _branch_streams(x.device)
        # Both branches may only start after everything that produced x on the current stream.
        s_attn.wait_stream(cur)
        s_moe.wait_stream(cur)
        # Attention is launched first: the MoE branch contains one host sync (expert counts),
        # so launching attention first lets its kernels run while the CPU waits on the router.
        with torch.cuda.stream(s_attn):
            with prange("parallel_attention_branch"):
                a = self.attn(self.norm_attn(x))
        with torch.cuda.stream(s_moe):
            with prange("parallel_moe_branch"):
                m = self.moe(self.norm_moe(x))
        cur.wait_stream(s_attn)
        cur.wait_stream(s_moe)
        # Cross-stream lifetime safety for the caching allocator.
        x.record_stream(s_attn)
        x.record_stream(s_moe)
        a.record_stream(cur)
        m.record_stream(cur)
        for t in self.moe.aux_tensors():
            t.record_stream(cur)
        with prange("parallel_merge"):
            return x + a + m


class FusionBlock(nn.Module):
    """Token-wise FusionMoE (NOT an attention layer): y = x + FusionMoE(RMSNorm_fusion(x))."""

    kind = "fusion"

    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.moe = _make_moe(cfg, cfg.expert_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with prange("fusion_moe"):
            return x + self.moe(self.norm(x))
