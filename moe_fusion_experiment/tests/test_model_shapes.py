import torch

from conftest import REAL_CONFIGS, tiny_model_cfg
from moefusion.model import build_model


def test_logits_shape_all_archs():
    for arch in "ABC":
        cfg = tiny_model_cfg(arch)
        m = build_model(cfg, seed=0)
        x = torch.randint(0, cfg.vocab_size, (3, cfg.max_seq_len))
        out = m(x)
        assert out.shape == (3, cfg.max_seq_len, cfg.vocab_size)


def test_layer_output_dims():
    for arch in "ABC":
        cfg = tiny_model_cfg(arch)
        m = build_model(cfg, seed=0)
        h = torch.randn(2, cfg.max_seq_len, cfg.d_model)
        for b in m.blocks:
            h2 = b(h)
            assert h2.shape == h.shape
        for f in m.fusions.values():
            assert f(h).shape == h.shape


def test_structure_counts():
    a, b, c = (build_model(tiny_model_cfg(x), 0) for x in "ABC")
    assert len(a.moe_modules()) == len(b.moe_modules()) == 4
    assert len(c.moe_modules()) == 4 + 2  # fusion_interval=2 with 4 layers -> after blocks 1 and 3
    assert list(c.fusions.keys()) == ["1", "3"]
    assert sum(1 for mod in a.modules() if mod.__class__.__name__ == "CausalSelfAttention") == 4
    assert sum(1 for mod in c.modules() if mod.__class__.__name__ == "CausalSelfAttention") == 4  # fusion != attention


def test_default_config_fusion_positions():
    from moefusion.config import load_experiment

    m, _ = load_experiment("C_parallel_fusion_samewidth", "pilot", config_dir=REAL_CONFIGS)
    assert m.fusion_positions() == [3, 7, 11] and m.n_moe_applications == 15
    m8 = tiny_model_cfg("C", n_layers=12, fusion_interval=8)
    assert m8.fusion_positions() == [7]
    mf = tiny_model_cfg("C", n_layers=12, fusion_interval="final_only")
    assert mf.fusion_positions() == [11]


def test_identical_init_across_architectures():
    """A and B have identical parameters at init; C(same width) shares every common tensor."""
    a = build_model(tiny_model_cfg("A"), seed=7)
    b = build_model(tiny_model_cfg("B"), seed=7)
    c = build_model(tiny_model_cfg("C"), seed=7)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
        assert torch.equal(sa[k], sc[k]), k
    assert any(k.startswith("fusions.") for k in sc)
    # a different seed gives different weights
    a2 = build_model(tiny_model_cfg("A"), seed=8)
    assert not torch.equal(a2.state_dict()["blocks.0.attn.qkv.weight"], sa["blocks.0.attn.qkv.weight"])


def test_matched_c_shares_non_expert_weights():
    a = build_model(tiny_model_cfg("A"), seed=3)
    cm = build_model(tiny_model_cfg("C", expert_hidden=32), seed=3)
    sa, sc = a.state_dict(), cm.state_dict()
    for k in sa:
        if ".experts." in k:
            assert sa[k].shape != sc[k].shape
        else:
            assert torch.equal(sa[k], sc[k]), k


def test_tied_embeddings():
    m = build_model(tiny_model_cfg("A"), 0)
    assert m.lm_head.weight is m.tok_emb.weight


def test_construction_without_init_is_finite():
    """Modules built WITHOUT init_parameters must never contain uninitialised memory (NaN/Inf garbage)."""
    from moefusion.blocks import FusionBlock, ParallelBlock, SerialBlock
    from moefusion.model import MoEFusionLM

    cfg = tiny_model_cfg("C")
    for mod in (MoEFusionLM(cfg), ParallelBlock(cfg), SerialBlock(cfg), FusionBlock(cfg)):
        for n, p in mod.named_parameters():
            assert torch.isfinite(p).all(), n


def test_build_model_overwrites_default_init():
    from moefusion.model import build_model

    m = build_model(tiny_model_cfg("A"), seed=0)
    w = m.blocks[0].moe.experts.w1.detach()
    assert w.abs().sum() > 0 and abs(float(w.std()) - 0.02) < 0.005
