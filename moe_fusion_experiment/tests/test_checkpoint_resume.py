import numpy as np
import torch

from conftest import tiny_model_cfg, tiny_train_cfg
from moefusion.data import TokenDataset
from moefusion.metrics import read_jsonl
from moefusion.trainer import Trainer


def _data(tmp_path, vocab=97, T=16):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "train.npy", rng.integers(0, vocab, size=(200, T + 1), dtype=np.uint16))
    np.save(tmp_path / "validation.npy", rng.integers(0, vocab, size=(40, T + 1), dtype=np.uint16))
    return TokenDataset(tmp_path, T, vocab)


def test_resume_matches_uninterrupted(tmp_path):
    data = _data(tmp_path)
    for arch in "ABC":
        mcfg = tiny_model_cfg(arch)
        tcfg = tiny_train_cfg(total_steps=6, ckpt_every=3)
        full = Trainer(mcfg, tcfg, data, tmp_path / f"full_{arch}", "cpu", f"full_{arch}", save_final_weights=False,
                       log_fn=lambda *a: None)
        full.run(resume=False)
        part = Trainer(mcfg, tcfg, data, tmp_path / f"part_{arch}", "cpu", f"part_{arch}", save_final_weights=False,
                       log_fn=lambda *a: None, stop_after_step=3)
        r = part.run(resume=False)
        assert r["status"] == "interrupted" and (tmp_path / f"part_{arch}" / "checkpoint.pt").exists()
        resumed = Trainer(mcfg, tcfg, data, tmp_path / f"part_{arch}", "cpu", f"part_{arch}", save_final_weights=False,
                          log_fn=lambda *a: None)
        summ = resumed.run(resume=True)
        assert summ["steps"] == 6 and summ["tokens_input"] == 6 * tcfg.tokens_per_step
        for k, v in full.final_state_dict.items():
            assert torch.allclose(v, resumed.final_state_dict[k], atol=1e-6, rtol=1e-5), (arch, k)
        a = [r["lm_loss"] for r in read_jsonl(tmp_path / f"full_{arch}" / "train_metrics.jsonl")]
        b = [r["lm_loss"] for r in read_jsonl(tmp_path / f"part_{arch}" / "train_metrics.jsonl")]
        assert len(a) == len(b) == 6
        assert np.allclose(a, b, atol=1e-5)


def test_checkpoint_config_mismatch_detected(tmp_path):
    data = _data(tmp_path)
    mcfg = tiny_model_cfg("A")
    t1 = tiny_train_cfg(total_steps=6, ckpt_every=3)
    Trainer(mcfg, t1, data, tmp_path / "r", "cpu", "r", save_final_weights=False, log_fn=lambda *a: None,
            stop_after_step=3).run(resume=False)
    t2 = tiny_train_cfg(total_steps=6, ckpt_every=3, lr=1e-3)
    try:
        Trainer(mcfg, t2, data, tmp_path / "r", "cpu", "r", save_final_weights=False, log_fn=lambda *a: None).run(resume=True)
    except RuntimeError as e:
        assert "CHECKPOINT MISMATCH" in str(e)
    else:
        raise AssertionError("config mismatch not detected")


def test_training_reduces_loss_and_logs(tmp_path):
    rng = np.random.default_rng(0)
    # learnable data: repeating ramps
    base = (np.arange(17)[None, :] + rng.integers(0, 97, size=(200, 1))) % 97
    np.save(tmp_path / "train.npy", base.astype(np.uint16))
    np.save(tmp_path / "validation.npy", base[:40].astype(np.uint16))
    data = TokenDataset(tmp_path, 16, 97)
    for arch in "ABC":
        s = Trainer(tiny_model_cfg(arch), tiny_train_cfg(total_steps=30, lr=1e-2, eval_every=10), data,
                    tmp_path / arch, "cpu", arch, save_final_weights=False, log_fn=lambda *a: None).run(resume=False)
        ev = read_jsonl(tmp_path / arch / "eval_metrics.jsonl")
        assert ev[0]["step"] == 0 and ev[-1]["final"]
        assert s["final_val_loss"] < ev[0]["val_loss"] - 1.0, arch
        tr = read_jsonl(tmp_path / arch / "train_metrics.jsonl")
        assert "expert_0_fraction" in tr[1] and "router_entropy" in tr[1]
