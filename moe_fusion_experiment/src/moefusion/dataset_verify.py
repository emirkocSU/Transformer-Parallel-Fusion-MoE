"""Dataset integrity / provenance / leakage checks for the prepared FineWeb-Edu token data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .utils import read_json, sha256_file

ID_KEYS = ("id", "doc_id", "document_id", "source_id", "uid")


def iter_jsonl(path, limit: Optional[int] = None) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def doc_key(rec: dict) -> Tuple[Optional[str], str]:
    """(source id if present, sha256 of the text)."""
    sid = None
    for k in ID_KEYS:
        if k in rec and rec[k] is not None:
            sid = str(rec[k])
            break
    text = rec.get("text", "")
    return sid, hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_doc_keys(path) -> Dict[str, object]:
    ids, hashes, n, n_with_id = set(), [], 0, 0
    for rec in iter_jsonl(path):
        sid, th = doc_key(rec)
        n += 1
        if sid is not None:
            ids.add(sid)
            n_with_id += 1
        hashes.append(th)
    return {"n_docs": n, "ids": ids, "n_with_id": n_with_id, "text_hashes": hashes}


def check_separation(train: Dict[str, object], val: Dict[str, object]) -> Dict[str, object]:
    """Document-level leakage check. Source-ID overlap must be zero; exact-text overlap is reported."""
    id_overlap = len(train["ids"] & val["ids"]) if train["ids"] and val["ids"] else None
    th_train = set(train["text_hashes"])
    text_overlap = sum(1 for h in set(val["text_hashes"]) if h in th_train)
    dup_within_val = len(val["text_hashes"]) - len(set(val["text_hashes"]))
    ids_available = bool(train["n_with_id"] == train["n_docs"] and val["n_with_id"] == val["n_docs"])
    passed = (id_overlap == 0) if ids_available else (text_overlap == 0)
    return {
        "train_docs": train["n_docs"], "val_docs": val["n_docs"],
        "source_ids_available": ids_available,
        "source_id_overlap": id_overlap,
        "exact_text_overlap_docs": text_overlap,
        "exact_text_overlap_fraction_of_val": text_overlap / max(1, val["n_docs"]),
        "duplicate_texts_within_val": dup_within_val,
        "passed": bool(passed),
    }


def spot_check_tokenization(tokenizer, docs_path, arr: np.ndarray, eot_id: int, n_docs: int = 200) -> Dict[str, object]:
    """Re-tokenise the first n documents and compare with the beginning of the packed array.
    Tries 'EOT appended after each document' and 'EOT prepended before each document'."""
    texts = [rec.get("text", "") for rec in iter_jsonl(docs_path, limit=n_docs)]
    if not texts:
        return {"status": "UNVERIFIED", "reason": "no documents"}
    encs = tokenizer.encode_batch(texts)
    flat = arr[: max(1, 1 + sum(len(e.ids) + 1 for e in encs) // arr.shape[1] + 1)].reshape(-1)
    results = {}
    for mode in ("append", "prepend"):
        stream: List[int] = []
        for e in encs:
            if mode == "prepend":
                stream.append(eot_id)
            stream.extend(e.ids)
            if mode == "append":
                stream.append(eot_id)
        m = min(len(stream), len(flat))
        eq = np.asarray(stream[:m], dtype=np.int64) == flat[:m].astype(np.int64)
        first_bad = int(np.argmin(eq)) if not eq.all() else None
        results[mode] = {"compared_tokens": int(m), "match": bool(eq.all()), "first_mismatch": first_bad,
                         "match_fraction": float(eq.mean())}
    matched = [k for k, v in results.items() if v["match"]]
    return {"status": "VERIFIED" if matched else "UNVERIFIED", "packing_mode": matched[0] if matched else None,
            "n_docs": len(texts), "details": results}


def verify_arrays(data_dir, manifest: Optional[dict], vocab_size: int, seq_len: int, eot_id: Optional[int],
                  compute_hashes: bool = True) -> Dict[str, object]:
    data_dir = Path(data_dir)
    rep: Dict[str, object] = {"checks": [], "warnings": [], "errors": []}

    def ok(msg):
        rep["checks"].append(msg)

    for split in ("train", "validation"):
        p = data_dir / f"{split}.npy"
        if not p.exists():
            rep["errors"].append(f"missing {p}")
            continue
        arr = np.load(p, mmap_mode="r")
        info = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if arr.ndim != 2 or arr.shape[1] not in (seq_len, seq_len + 1):
            rep["errors"].append(f"{split}: bad shape {arr.shape}")
        mx, mn = int(arr.max()), int(arr.min())
        info.update({"min_token": mn, "max_token": mx})
        if mn < 0 or mx >= vocab_size:
            rep["errors"].append(f"{split}: token ids out of range [{mn}, {mx}]")
        else:
            ok(f"{split}: token ids within [0, {vocab_size})")
        if eot_id is not None:
            n_eot = int(np.count_nonzero(np.asarray(arr) == eot_id))
            info["eot_count"] = n_eot
        info["row_layout"] = "seq_len+1 (input=row[:-1], target=row[1:])" if arr.shape[1] == seq_len + 1 else "seq_len (last target ignored)"
        if compute_hashes:
            info["sha256_file"] = sha256_file(p)
        if manifest is not None:
            mp = (manifest.get("packed") or {}).get(split, {})
            if mp:
                if mp.get("examples") is not None and mp["examples"] != arr.shape[0]:
                    rep["errors"].append(f"{split}: manifest examples {mp['examples']} != array rows {arr.shape[0]}")
                else:
                    ok(f"{split}: row count matches manifest ({arr.shape[0]})")
                if mp.get("usable_tokens") is not None:
                    usable = arr.shape[0] * (arr.shape[1] - 1 if arr.shape[1] == seq_len + 1 else arr.shape[1])
                    if usable != mp["usable_tokens"]:
                        rep["warnings"].append(f"{split}: usable tokens {usable} != manifest {mp['usable_tokens']}")
                    else:
                        ok(f"{split}: usable (predicted) tokens match manifest ({usable})")
                if compute_hashes and mp.get("sha256"):
                    if mp["sha256"] == info["sha256_file"]:
                        ok(f"{split}: sha256 matches manifest")
                    else:
                        rep["warnings"].append(f"{split}: file sha256 differs from manifest sha256 (manifest may hash a different representation)")
        rep[split] = info
    rep["passed"] = not rep["errors"]
    return rep


def load_manifest(data_dir, reference_dir=None) -> Tuple[Optional[dict], Dict[str, object]]:
    data_dir = Path(data_dir)
    notes: Dict[str, object] = {}
    man = None
    mp = data_dir / "manifest.json"
    if mp.exists():
        man = read_json(mp)
        notes["manifest_sha256"] = sha256_file(mp)
        cp = data_dir / ".complete.json"
        if cp.exists():
            comp = read_json(cp)
            notes["complete_marker"] = comp
            notes["complete_marker_matches_manifest"] = comp.get("manifest_sha256") == notes["manifest_sha256"]
    if reference_dir is not None:
        rp = Path(reference_dir) / "manifest.json"
        if rp.exists():
            ref = read_json(rp)
            if man is None:
                notes["manifest_source"] = "reference copy (data_dir has no manifest.json)"
                man = ref
            else:
                notes["manifest_equals_reference"] = (man == ref)
    return man, notes
