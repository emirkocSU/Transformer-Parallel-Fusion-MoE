import json

from moefusion.dataset_verify import check_separation, collect_doc_keys


def _write(path, docs):
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


def test_detects_no_overlap(tmp_path):
    _write(tmp_path / "t.jsonl", [{"id": f"t{i}", "text": f"train {i}"} for i in range(50)])
    _write(tmp_path / "v.jsonl", [{"id": f"v{i}", "text": f"val {i}"} for i in range(10)])
    r = check_separation(collect_doc_keys(tmp_path / "t.jsonl"), collect_doc_keys(tmp_path / "v.jsonl"))
    assert r["passed"] and r["source_id_overlap"] == 0 and r["exact_text_overlap_docs"] == 0


def test_detects_id_leak(tmp_path):
    _write(tmp_path / "t.jsonl", [{"id": f"d{i}", "text": f"x {i}"} for i in range(50)])
    _write(tmp_path / "v.jsonl", [{"id": "d3", "text": "x 3"}, {"id": "v1", "text": "y"}])
    r = check_separation(collect_doc_keys(tmp_path / "t.jsonl"), collect_doc_keys(tmp_path / "v.jsonl"))
    assert not r["passed"] and r["source_id_overlap"] == 1


def test_text_hash_fallback_without_ids(tmp_path):
    _write(tmp_path / "t.jsonl", [{"text": f"x {i}"} for i in range(5)])
    _write(tmp_path / "v.jsonl", [{"text": "x 2"}])
    r = check_separation(collect_doc_keys(tmp_path / "t.jsonl"), collect_doc_keys(tmp_path / "v.jsonl"))
    assert not r["source_ids_available"] and not r["passed"]
