import pytest

from conftest import REAL_CONFIGS, tiny_model_cfg
from moefusion.config import load_experiment
from moefusion.flop_counter import analytic_budget, matched_hidden, matched_mismatch
from moefusion.model import MoEFusionLM, count_parameters


@pytest.mark.parametrize("arch", ["A", "B", "C"])
def test_analytic_matches_module_tiny(arch):
    cfg = tiny_model_cfg(arch)
    p = count_parameters(MoEFusionLM(cfg))
    a = analytic_budget(cfg)
    assert p["total"] == a["params_total"]
    assert p["attention"] == a["params_attention"]
    assert p["router"] == a["params_router"]
    assert p["experts"] == a["params_experts"]
    assert p["embedding"] == a["params_embedding"]
    assert p["norms"] == a["params_norms"]


@pytest.mark.parametrize("name", ["A_serial", "B_parallel", "C_parallel_fusion_samewidth", "C_parallel_fusion_matched"])
def test_analytic_matches_module_full_size(name):
    import torch

    cfg, _ = load_experiment(name, "pilot", config_dir=REAL_CONFIGS)
    with torch.device("meta"):
        m = MoEFusionLM(cfg)
    p = count_parameters(m)
    a = analytic_budget(cfg)
    assert p["total"] == a["params_total"]
    assert p["experts"] == a["params_experts"]


def test_compute_matching_exact():
    a, _ = load_experiment("A_serial", "pilot", config_dir=REAL_CONFIGS)
    c, _ = load_experiment("C_parallel_fusion_matched", "pilot", config_dir=REAL_CONFIGS)
    mm = matched_mismatch(a, c)
    assert mm["expert_params_rel_diff"] == 0.0
    assert mm["active_moe_flops_rel_diff"] == 0.0
    assert abs(mm["total_params_rel_diff"]) < 0.02
    assert abs(mm["train_flops_rel_diff"]) < 0.02
    assert 12 * 1920 == 15 * 1536 == 23040


def test_ab_identical_budget():
    a, _ = load_experiment("A_serial", "pilot", config_dir=REAL_CONFIGS)
    b, _ = load_experiment("B_parallel", "pilot", config_dir=REAL_CONFIGS)
    ba, bb = analytic_budget(a), analytic_budget(b)
    assert ba == bb


def test_matched_hidden_rule():
    r = matched_hidden(1920, 12, 15)
    assert r["rounded"] == 1536 and r["rel_mismatch"] == 0.0
    r2 = matched_hidden(1920, 12, 13)  # fusion every 8 -> 1 fusion layer
    assert r2["rounded"] % 64 == 0 and abs(r2["rel_mismatch"]) < 0.03


def test_verdict_rule_single_and_two_seeds():
    from moefusion.analysis import multi_seed_summary

    def ps(dlist):
        # minimal per-seed structure with B and C_same / C_matched final losses and paired stats
        out = {}
        for s, (d_same, d_match, lb) in dlist.items():
            out[s] = {"runs": {"B": {"final_val_loss": lb, "train_seconds": 60},
                               "C_same": {"final_val_loss": lb + d_same, "train_seconds": 70},
                               "C_matched": {"final_val_loss": lb + d_match, "train_seconds": 65}},
                      "paired": {"C_same_minus_B": {"mean": d_same, "ci95_normal": [d_same - 0.001, d_same + 0.001]},
                                 "C_matched_minus_B": {"mean": d_match, "ci95_normal": [d_match - 0.001, d_match + 0.001]}},
                      "common_wallclock": {"loss": {"B": lb, "C_same": lb + d_same, "C_matched": lb + d_match}}}
        return out

    one = multi_seed_summary(ps({42: (-0.005, -0.03, 5.0)}))
    assert one["comparisons"]["C_same_minus_B"]["verdict"].startswith("approximately equal")
    assert one["comparisons"]["C_matched_minus_B"]["verdict"].startswith("better")
    assert one["fusion_verdict"].startswith("USEFUL") and "single seed" in one["fusion_verdict"]
    small = multi_seed_summary(ps({42: (-0.012, 0.03, 5.0)}))  # CI excludes 0 but |dL| inside the 0.02 margin
    assert small["comparisons"]["C_same_minus_B"]["verdict"].startswith("approximately equal")
    assert small["comparisons"]["C_matched_minus_B"]["verdict"].startswith("worse")
    two = multi_seed_summary(ps({42: (-0.03, 0.0, 5.0), 43: (0.01, 0.0, 5.05)}))  # direction flips between seeds
    assert two["comparisons"]["C_same_minus_B"]["verdict"].startswith("inconclusive")
