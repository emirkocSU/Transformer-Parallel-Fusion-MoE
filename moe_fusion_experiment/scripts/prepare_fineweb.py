"""FALLBACK dataset preparation (only used if /content/moe_data is missing and no Drive copy exists).

Reproduces the documented recipe of data_reference/manifest.json as closely as is knowable:
  HuggingFaceFW/fineweb-edu, config sample-10BT, pinned revision, first N documents in source order,
  document-level split (doc_index % 20 == 0 -> validation), tokenised ONCE with the shared tokenizer.json,
  EOT appended after every document, stream packed into NON-overLAPPING (seq_len + 1)-token rows
  (input = row[:-1], target = row[1:]), uint16 .npy, manifest + checksums.
A rebuilt dataset is NOT guaranteed to be byte-identical to the original one (the original split/packing
script is not available); the verification report records this. All architectures still share it exactly.
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from _common import DATA_REFERENCE

import numpy as np

from moefusion.tokenizer import load_tokenizer, tokenizer_info
from moefusion.utils import now_iso, sha256_file, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/moe_data")
    ap.add_argument("--n-docs", type=int, default=300000)
    ap.add_argument("--modulus", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--revision", default="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
    ap.add_argument("--tokenizer", default=str(DATA_REFERENCE / "tokenizer.json"))
    args = ap.parse_args()
    from datasets import load_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = out / "tokenizer.json"
    if not tok_path.exists():
        tok_path.write_bytes(Path(args.tokenizer).read_bytes())
    ti = tokenizer_info(tok_path)
    tok = load_tokenizer(tok_path)
    eot = ti["eot_id"]
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True, revision=args.revision)
    files = {"train": open(out / "train_documents.jsonl", "w", encoding="utf-8"),
             "validation": open(out / "validation_documents.jsonl", "w", encoding="utf-8")}
    streams = {"train": [], "validation": []}
    counts = {"train": 0, "validation": 0}
    buf = {"train": [], "validation": []}

    def flush(split):
        if not buf[split]:
            return
        for e in tok.encode_batch([t for t in buf[split]]):
            streams[split].append(np.asarray(e.ids + [eot], dtype=np.uint16))
        buf[split] = []

    for i, rec in enumerate(ds):
        if i >= args.n_docs:
            break
        split = "validation" if i % args.modulus == 0 else "train"
        files[split].write(json.dumps({"id": rec.get("id"), "index": i, "text": rec["text"]}) + "\n")
        buf[split].append(rec["text"])
        counts[split] += 1
        if len(buf[split]) >= 2000:
            flush(split)
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1:,} documents", flush=True)
    for s in buf:
        flush(s)
        files[s].close()
    man = {"dataset": "HuggingFaceFW/fineweb-edu", "configuration": "sample-10BT", "revision": args.revision,
           "source_split": "train", "source_order": "first N documents", "created_at": now_iso(),
           "script_version": "fallback-1", "source_document_count": sum(counts.values()),
           "train_documents": counts["train"], "validation_documents": counts["validation"],
           "validation_modulus": args.modulus, "validation_rule": "doc_index % modulus == 0",
           "tokenizer_id": ti["sha256"], "vocab_size": ti["vocab_size"], "context_length": args.seq_len,
           "packing": "EOT appended; non-overlapping (seq_len+1) rows", "packed": {},
           "source_sha256": {s: sha256_file(out / f"{s}_documents.jsonl") for s in ("train", "validation")}}
    for s in ("train", "validation"):
        stream = np.concatenate(streams[s])
        n = len(stream) // (args.seq_len + 1)
        arr = stream[: n * (args.seq_len + 1)].reshape(n, args.seq_len + 1)
        np.save(out / f"{s}.npy", arr)
        man["packed"][s] = {"path": str(out / f"{s}.npy"), "examples": int(n), "source_tokens": int(len(stream)),
                            "usable_tokens": int(n * args.seq_len), "sha256": sha256_file(out / f"{s}.npy")}
    write_json(out / "manifest.json", man)
    write_json(out / ".complete.json", {"manifest_sha256": sha256_file(out / "manifest.json"), "revision": args.revision})
    print("  rebuilt dataset:", json.dumps(man["packed"], indent=1))


if __name__ == "__main__":
    main()
