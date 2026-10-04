import math

import numpy as np
import torch

from moefusion.metrics import collapse_check, utilization_stats
from moefusion.moe import MoE


def test_balance_loss_uniform_equals_one():
    m = MoE(8, 4, 2, 8)
    torch.nn.init.zeros_(m.router.weight)  # uniform probabilities
    for p in m.experts.parameters():
        torch.nn.init.normal_(p, 0, 0.1)
    m(torch.randn(1, 64, 8))
    # with all-equal probabilities topk picks the same experts, so f is skewed but P is uniform -> loss = 1
    assert abs(float(m.last_aux["balance_loss"].detach()) - 1.0) < 1e-5
    assert abs(float(m.last_aux["entropy"]) - math.log(4)) < 1e-5


def test_balance_loss_penalises_imbalance():
    m = MoE(8, 4, 2, 8)
    with torch.no_grad():
        m.router.weight.zero_()
        m.router.weight[0, 0] = 50.0
        m.router.weight[1, 0] = 49.0
    for p in m.experts.parameters():
        torch.nn.init.normal_(p, 0, 0.1)
    x = torch.randn(1, 64, 8)
    x[..., 0] = x[..., 0].abs() + 1
    m(x)
    assert float(m.last_aux["balance_loss"].detach()) > 1.5


def test_utilization_stats_values():
    counts = np.array([[10, 10, 10, 10], [40, 0, 0, 0]])
    st = utilization_stats(counts, top_k=2)
    assert abs(st["expert_0_fraction"] - 50 / 80) < 1e-9
    assert st["util_min_fraction"] == 0.0
    assert st["util_maxmin_ratio_max"] == math.inf
    assert abs(st["per_layer_fraction"][0][0] - 0.25) < 1e-9
    assert st["util_cv_max"] > st["util_cv_mean"] > 0


def test_collapse_detection():
    frac = np.array([[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0], [0.6, 0.39, 0.005, 0.005]])
    assert collapse_check(frac, top_k=2, min_fraction=0.01) == [1, 2]
