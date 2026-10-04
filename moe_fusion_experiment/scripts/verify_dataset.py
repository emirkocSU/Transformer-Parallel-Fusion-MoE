"""Dataset verification gate: provenance hashes, shapes, token range, train/validation document
separation, and a re-tokenisation spot check proving that the .npy files come from the .jsonl documents
with THIS tokenizer.  Writes <out>/dataset_verification.json.
"""
import argparse
import json
import sys
from pathlib import Path

import _common  # noqa: F401
from _common import DATA_REFERENCE, EXIT_GATE_FAILED, banner

import numpy as np

from moefusion.dataset_verify import check_separation, collect_doc_keys, load_manifest, spot_check_tokenization, verify_arrays
from moefusion.tokenizer import load_tokenizer, tokenizer_info
from moefusion.utils import sha256_file, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--spot-docs", type=int, default=200)
    args = ap.parse_args()
    d = Path(args.data_dir)
    banner(f"DATASET VERIFICATION ({d})")
    rep = {"data_dir": str(d)}
    errors, warnings = [], []

    man, notes = load_manifest(d, DATA_REFERENCE)
    rep["manifest_notes"] = notes
    if man is None:
        warnings.append("no manifest.json found (neither in data dir nor reference copy)")
    elif notes.get("complete_marker_matches_manifest") is False:
        warnings.append(".complete.json does not match manifest.json sha256")
    if notes.get("manifest_equals_reference") is False:
        warnings.append("data_dir/manifest.json differs from the reference manifest shipped with the code "
                        "(data was re-prepared?)")

    tok_path = d / "tokenizer.json"
    if not tok_path.exists():
        tok_path = DATA_REFERENCE / "tokenizer.json"
        warnings.append("tokenizer.json not in data dir; using the reference copy shipped with the code")
    ti = tokenizer_info(tok_path)
    rep["tokenizer"] = ti
    ref_tok = sha256_file(DATA_REFERENCE / "tokenizer.json")
    if man is not None and man.get("tokenizer_id") and man["tokenizer_id"] != ti["sha256"]:
        errors.append(f"tokenizer sha256 {ti['sha256']} != manifest tokenizer_id {man['tokenizer_id']}")
    if ti["sha256"] != ref_tok:
        warnings.append("tokenizer.json differs from the reference copy shipped with the code")
    if ti["vocab_size"] != args.vocab_size:
        errors.append(f"tokenizer vocab {ti['vocab_size']} != model vocab {args.vocab_size}")
    print(f"  tokenizer: {ti['model_type']} vocab={ti['vocab_size']} eot_id={ti['eot_id']} sha256={ti['sha256'][:16]}...")

    arr_rep = verify_arrays(d, man, args.vocab_size, args.seq_len, ti["eot_id"], compute_hashes=True)
    rep["arrays"] = arr_rep
    errors += arr_rep["errors"]
    warnings += arr_rep["warnings"]
    for c in arr_rep["checks"]:
        print("  OK  ", c)
    for s in ("train", "validation"):
        if s in arr_rep:
            print(f"  {s}: {arr_rep[s]}")

    # ---- document-level train/validation separation
    tj, vj = d / "train_documents.jsonl", d / "validation_documents.jsonl"
    if tj.exists() and vj.exists():
        if man is not None:
            for s, p in (("train", tj), ("validation", vj)):
                want = (man.get("source_sha256") or {}).get(s)
                if want:
                    got = sha256_file(p)
                    if got == want:
                        print(f"  OK   {s}_documents.jsonl sha256 matches manifest")
                    else:
                        warnings.append(f"{s}_documents.jsonl sha256 differs from manifest")
        tk, vk = collect_doc_keys(tj), collect_doc_keys(vj)
        sep = check_separation(tk, vk)
        rep["separation"] = sep
        print("  separation:", json.dumps(sep))
        if man is not None:
            if man.get("train_documents") not in (None, tk["n_docs"]):
                errors.append(f"train docs {tk['n_docs']} != manifest {man.get('train_documents')}")
            if man.get("validation_documents") not in (None, vk["n_docs"]):
                errors.append(f"validation docs {vk['n_docs']} != manifest {man.get('validation_documents')}")
        if not sep["passed"]:
            errors.append(f"VALIDATION LEAKAGE: {sep}")
        elif sep["exact_text_overlap_docs"]:
            warnings.append(f"{sep['exact_text_overlap_docs']} validation documents have an exact-text duplicate in "
                            "train under different source ids (web near-duplicates; reported, not removed)")
        # ---- re-tokenisation spot check
        tok = load_tokenizer(tok_path)
        spot = {}
        for s, p in (("validation", vj), ("train", tj)):
            arr = np.load(d / f"{s}.npy", mmap_mode="r")
            spot[s] = spot_check_tokenization(tok, p, arr, ti["eot_id"], args.spot_docs)
            print(f"  spot-check {s}: {spot[s]['status']} (packing={spot[s].get('packing_mode')})")
        rep["tokenization_spot_check"] = spot
        if any(v["status"] != "VERIFIED" for v in spot.values()):
            warnings.append("re-tokenisation spot check could not reproduce the packed stream "
                            "(packing convention unknown or rows shuffled) - provenance UNVERIFIED, data still usable")
    else:
        warnings.append("train/validation_documents.jsonl missing: document-level leakage check UNVERIFIED")
        rep["separation"] = {"status": "UNVERIFIED"}

    rep["errors"], rep["warnings"] = errors, warnings
    rep["passed"] = not errors
    write_json(Path(args.out) / "dataset_verification.json", rep)
    for w in warnings:
        print("  WARNING:", w)
    if errors:
        for e in errors:
            print("  ERROR:", e)
        sys.exit(EXIT_GATE_FAILED)
    print("  DATASET VERIFICATION PASSED")


if __name__ == "__main__":
    main()
