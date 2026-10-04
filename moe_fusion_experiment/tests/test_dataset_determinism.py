import numpy as np

from moefusion.data import BatchSchedule, eval_rows


def test_same_seed_same_batches():
    a = BatchSchedule(1000, 64, seed=42)
    b = BatchSchedule(1000, 64, seed=42)
    for s in (0, 1, 5, 15, 16, 40):
        assert np.array_equal(a.rows_for_step(s), b.rows_for_step(s))


def test_different_seed_different_batches():
    a = BatchSchedule(1000, 64, seed=42)
    b = BatchSchedule(1000, 64, seed=43)
    assert not np.array_equal(a.rows_for_step(0), b.rows_for_step(0))


def test_epoch_is_a_permutation_and_crosses_boundary():
    s = BatchSchedule(100, 30, seed=1)
    rows = np.concatenate([s.rows_for_step(i) for i in range(10)])  # 300 rows = 3 epochs
    for e in range(3):
        assert sorted(rows[e * 100 : (e + 1) * 100].tolist()) == list(range(100))


def test_order_independent_of_query_order_and_microbatch():
    s1 = BatchSchedule(500, 64, seed=3)
    s2 = BatchSchedule(500, 64, seed=3)
    forward = [s1.rows_for_step(i) for i in range(12)]
    backward = [s2.rows_for_step(i) for i in reversed(range(12))][::-1]
    for f, b in zip(forward, backward):
        assert np.array_equal(f, b)
    # micro-batching is a pure partition of the global batch
    g = s1.rows_for_step(3)
    assert np.array_equal(np.concatenate([g[i : i + 16] for i in range(0, 64, 16)]), g)


def test_eval_rows_fixed():
    assert np.array_equal(eval_rows(15088, 1024), eval_rows(15088, 1024))
    assert len(eval_rows(15088, 1024)) == 1024
    assert len(eval_rows(100, None)) == 100
