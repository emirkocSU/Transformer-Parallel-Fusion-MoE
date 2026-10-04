import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (Su et al., 2021), rotate-half (GPT-NeoX / LLaMA) layout.

    cos/sin tables are non-persistent FP32 buffers; the rotation is computed in FP32.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float64)
        freqs = torch.outer(t, inv_freq)  # (T, hd/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, hd)
        self.register_buffer("cos", emb.cos().float(), persistent=False)
        self.register_buffer("sin", emb.sin().float(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q, k: (B, H, T, hd)
        T = q.shape[-2]
        if T > self.cos.shape[0]:
            raise ValueError(f"sequence length {T} exceeds RoPE table {self.cos.shape[0]}")
        cos = self.cos[:T]
        sin = self.sin[:T]
        qf, kf = q.float(), k.float()
        q_out = qf * cos + self._rotate_half(qf) * sin
        k_out = kf * cos + self._rotate_half(kf) * sin
        return q_out.to(q.dtype), k_out.to(k.dtype)
