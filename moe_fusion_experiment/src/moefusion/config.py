"""Experiment configuration.

Configuration is assembled from three YAML layers that are deep-merged in order:

    configs/base.yaml          -> everything shared by every architecture
    configs/models/<X>.yaml    -> ONLY the architecture-defining fields
    configs/runs/<mode>.yaml   -> token budget / evaluation schedule (shared)

`assert_fair()` verifies programmatically that two or more experiment configs differ
only in the fields that define the architecture under test.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml

ARCHS = ("serial", "parallel", "parallel_fusion")
ATTENTION_BACKENDS = ("sdpa_flash", "sdpa_efficient", "sdpa_math", "sdpa", "flash_attn")
MOE_BACKENDS = ("reference", "padded_bmm")
PARALLEL_BACKENDS = ("reference", "concurrent")

# Fields that are allowed to differ between architectures in a primary comparison.
ARCH_DEFINING_FIELDS = {"name", "arch", "expert_hidden", "fusion_interval", "matched_reference_hidden"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import os

CONFIG_DIR = Path(os.environ.get("MOE_CONFIG_DIR", str(PROJECT_ROOT / "configs")))


@dataclass
class ModelConfig:
    name: str = "unnamed"
    arch: str = "serial"
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    head_dim: int = 64
    max_seq_len: int = 1024
    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 1920
    # parallel_fusion only: int N (fusion after every N parallel blocks) or "final_only"
    fusion_interval: Optional[Union[int, str]] = None
    # compute-matched C only: the baseline expert width this config is matched against
    matched_reference_hidden: Optional[int] = None
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    init_std: float = 0.02
    attention_backend: str = "sdpa_flash"
    moe_backend: str = "reference"
    parallel_backend: str = "reference"

    def validate(self) -> "ModelConfig":
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {self.arch!r}")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError("n_heads * head_dim must equal d_model")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if not (1 <= self.top_k <= self.n_experts):
            raise ValueError("top_k must be in [1, n_experts]")
        if self.attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {ATTENTION_BACKENDS}")
        if self.moe_backend not in MOE_BACKENDS:
            raise ValueError(f"moe_backend must be one of {MOE_BACKENDS}")
        if self.parallel_backend not in PARALLEL_BACKENDS:
            raise ValueError(f"parallel_backend must be one of {PARALLEL_BACKENDS}")
        if self.arch == "parallel_fusion":
            if self.fusion_interval is None:
                raise ValueError("parallel_fusion requires fusion_interval")
            if isinstance(self.fusion_interval, str):
                if self.fusion_interval != "final_only":
                    raise ValueError("string fusion_interval must be 'final_only'")
            elif not (1 <= int(self.fusion_interval) <= self.n_layers):
                raise ValueError("fusion_interval must be in [1, n_layers]")
        elif self.fusion_interval is not None:
            raise ValueError(f"fusion_interval must be None for arch={self.arch}")
        return self

    def fusion_positions(self) -> List[int]:
        """0-based indices of backbone blocks that are FOLLOWED by a FusionMoE."""
        if self.arch != "parallel_fusion":
            return []
        if self.fusion_interval == "final_only":
            return [self.n_layers - 1]
        n = int(self.fusion_interval)
        return [i for i in range(self.n_layers) if (i + 1) % n == 0]

    @property
    def n_fusion_layers(self) -> int:
        return len(self.fusion_positions())

    @property
    def n_moe_applications(self) -> int:
        return self.n_layers + self.n_fusion_layers


@dataclass
class TrainConfig:
    seq_len: int = 1024
    global_batch_seqs: int = 64
    micro_batch_seqs: Optional[int] = None  # chosen by the common memory probe
    total_steps: int = 384
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    balance_coef: float = 0.01
    # "sum": every MoE application (router) adds balance_coef * L_balance to the objective, as in Switch Transformer
    # ("for each Switch layer, this auxiliary loss is added to the total model loss"). Every router of every
    # architecture therefore receives the SAME balancing pressure. "mean" (divide by the number of MoE
    # applications) is kept only for ablations.
    aux_loss_reduction: str = "sum"
    zloss_coef: float = 0.0
    precision: str = "bf16"
    seed: int = 42
    data_seed: Optional[int] = None  # None -> same as seed (paired design)
    eval_every: int = 32
    eval_seqs: int = 1024
    final_eval_seqs: Optional[int] = None  # None -> full validation set
    log_every: int = 8
    ckpt_every: int = 0  # 0 -> only at the end
    untimed_warmup_microbatches: int = 2
    collapse_min_fraction: float = 0.01
    collapse_patience: int = 3
    imbalance_alert_ratio: float = 10.0

    def validate(self) -> "TrainConfig":
        if self.micro_batch_seqs is not None:
            if self.global_batch_seqs % self.micro_batch_seqs:
                raise ValueError("global_batch_seqs must be divisible by micro_batch_seqs")
        if self.aux_loss_reduction not in ("sum", "mean"):
            raise ValueError("aux_loss_reduction must be 'sum' or 'mean'")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        if self.total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        return self

    @property
    def grad_accum(self) -> int:
        assert self.micro_batch_seqs is not None, "micro_batch_seqs not set"
        return self.global_batch_seqs // self.micro_batch_seqs

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch_seqs * self.seq_len

    @property
    def total_tokens(self) -> int:
        return self.total_steps * self.tokens_per_step

    @property
    def warmup_steps(self) -> int:
        return max(1, int(round(self.warmup_frac * self.total_steps)))

    @property
    def effective_data_seed(self) -> int:
        return self.seed if self.data_seed is None else self.data_seed


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build(cls, d: Dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def load_experiment(
    model: str,
    run: str,
    overrides: Optional[Dict[str, Any]] = None,
    config_dir: Union[str, Path] = CONFIG_DIR,
) -> Tuple[ModelConfig, TrainConfig]:
    """Load base + model + run YAML (+ programmatic overrides) into validated dataclasses.

    `model` / `run` may be a short name (e.g. "A_serial", "pilot") or a path to a YAML file.
    """
    config_dir = Path(config_dir)
    mpath = Path(model) if str(model).endswith(".yaml") else config_dir / "models" / f"{model}.yaml"
    rpath = Path(run) if str(run).endswith(".yaml") else config_dir / "runs" / f"{run}.yaml"
    merged = _read_yaml(config_dir / "base.yaml")
    merged = deep_merge(merged, _read_yaml(mpath))
    merged = deep_merge(merged, _read_yaml(rpath))
    merged = deep_merge(merged, overrides or {})
    mcfg = _build(ModelConfig, merged.get("model", {})).validate()
    tcfg = _build(TrainConfig, merged.get("train", {})).validate()
    if tcfg.seq_len != mcfg.max_seq_len:
        raise ValueError("train.seq_len must equal model.max_seq_len")
    return mcfg, tcfg


def config_dict(mcfg: ModelConfig, tcfg: TrainConfig) -> Dict[str, Any]:
    return {"model": asdict(mcfg), "train": asdict(tcfg)}


def config_hash(d: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def assert_fair(configs: Sequence[Tuple[ModelConfig, TrainConfig]]) -> List[str]:
    """Raise if configs differ in anything other than architecture-defining fields.

    Returns a human-readable list of the (allowed) differences.
    """
    if len(configs) < 2:
        return []
    ref_m, ref_t = configs[0]
    notes = []
    for m, t in configs[1:]:
        dt = {k: (getattr(ref_t, k), getattr(t, k)) for k in asdict(ref_t) if getattr(ref_t, k) != getattr(t, k)}
        if dt:
            raise AssertionError(f"FAIRNESS VIOLATION: training configs differ between {ref_m.name} and {m.name}: {dt}")
        for k in asdict(ref_m):
            a, b = getattr(ref_m, k), getattr(m, k)
            if a != b:
                if k not in ARCH_DEFINING_FIELDS:
                    raise AssertionError(
                        f"FAIRNESS VIOLATION: model field {k!r} differs between {ref_m.name} ({a}) and {m.name} ({b})"
                    )
                notes.append(f"{m.name}.{k} = {b} (vs {ref_m.name}: {a})")
    return notes


def lr_at(step: int, tcfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to min_lr_ratio * lr. `step` is 0-based optimizer step."""
    w = tcfg.warmup_steps
    if step < w:
        return tcfg.lr * (step + 1) / w
    progress = (step - w) / max(1, tcfg.total_steps - w)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tcfg.lr * (tcfg.min_lr_ratio + (1.0 - tcfg.min_lr_ratio) * cos)
