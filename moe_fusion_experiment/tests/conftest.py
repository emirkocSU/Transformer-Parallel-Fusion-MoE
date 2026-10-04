import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIGS = ROOT / "configs"  # tests always check the REAL experiment configuration
sys.path.insert(0, str(ROOT / "src"))

from moefusion.config import ModelConfig, TrainConfig  # noqa: E402

ARCH_KW = {
    "A": dict(name="A", arch="serial"),
    "B": dict(name="B", arch="parallel"),
    "C": dict(name="C", arch="parallel_fusion", fusion_interval=2),
}


def tiny_model_cfg(arch="A", **kw) -> ModelConfig:
    base = dict(vocab_size=97, d_model=32, n_layers=4, n_heads=4, head_dim=8, max_seq_len=16, n_experts=4,
                top_k=2, expert_hidden=48, attention_backend="sdpa")
    base.update(ARCH_KW[arch])
    base.update(kw)
    return ModelConfig(**base).validate()


def tiny_train_cfg(**kw) -> TrainConfig:
    base = dict(seq_len=16, global_batch_seqs=8, micro_batch_seqs=4, total_steps=6, lr=3e-3, precision="fp32",
                eval_every=3, eval_seqs=8, final_eval_seqs=16, log_every=2, ckpt_every=0, untimed_warmup_microbatches=1)
    base.update(kw)
    return TrainConfig(**base).validate()


@pytest.fixture
def device():
    return torch.device("cpu")


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA (runs on the Colab A100)")
