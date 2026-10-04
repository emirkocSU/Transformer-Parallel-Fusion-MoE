"""Rebuild the prepared FineWeb-Edu token data and PROVE whether it is identical to the original.

The original preparation script is not available, but its manifest (data_reference/manifest.json) records the
source (dataset, config, pinned revision, first N documents), the split modulus, per-split document and token
counts, and the sha256 of every produced file. This script:

  1. downloads the pinned source parquet files (huggingface_hub, exact revision) and reads the first N documents
     in source order;
  2. tokenises every document ONCE with the shipped tokenizer.json (sha256 must equal manifest tokenizer_id);
  3. INFERS the unknown conventions from the manifest instead of guessing:
       - validation rule  doc_index % modulus == r       (r found from the exact per-split token counts)
       - EOT tokens per document (0 / 1) and placement (append / prepend)
       - packing: non-overlapping rows of seq_len + 1 tokens
       - array dtype (uint16 / int32 / uint32 / int64)
     each candidate is accepted ONLY if the resulting .npy file has exactly the manifest's sha256;
  4. writes train/validation .npy + documents .jsonl (format also matched against the manifest sha256 when possible),
     copies the original manifest when the rebuild is byte-identical, and writes rebuild_report.json.

If no candidate reproduces the original hash, the data is still built deterministically (best-matching convention),
the report says NOT IDENTICAL, and a new manifest is written -- the experiment stays fair (all architectures read the
same files) but is then not byte-identical to the first preparation.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import _common  # noqa: F401
from _common import DATA_REFERENCE, banner

import numpy as np

from moefusion.tokenizer import load_tokenizer, tokenizer_info
from moefusion.utils import now_iso, read_json, sha256_file, write_json

DTYPES = ["uint16", "int32", "uint32", "int64"]


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def npy_sha(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return sha_bytes(buf.getvalue())


def iter_source_docs(args):
    """Yield source records in source order (local parquet dir for testing, else the pinned HF revision)."""
    import pyarrow.parquet as pq

    if args.source_parquet_dir:
        files = sorted(str(p) for p in Path(args.source_parquet_dir).glob("*.parquet"))
        get = lambda f: f  # noqa: E731
    else:
        from huggingface_hub import HfApi, hf_hub_download

        prefix = f"sample/{args.config.split('-')[1]}/"  # sample-10BT -> sample/10BT/
        files = sorted(f for f in HfApi().list_repo_files(args.dataset, repo_type="dataset", revision=args.revision)
                       if f.startswith(prefix) and f.endswith(".parquet"))
        print(f"  source files at revision {args.revision[:12]}: {len(files)} (first: {files[0]})", flush=True)

        def get(f):
            t = time.time()
            p = hf_hub_download(args.dataset, f, repo_type="dataset", revision=args.revision, cache_dir=args.cache_dir)
            print(f"  downloaded {f} ({os.path.getsize(p) / 1e9:.2f} GB, {time.time() - t:.0f}s)", flush=True)
            return p
    for f in files:
        pf = pq.ParquetFile(get(f))
        for batch in pf.iter_batches(batch_size=8192):
            for rec in batch.to_pylist():
                yield rec


def _full(rec):
    return {k: (v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v)) for k, v in rec.items()}


_VARIANTS = {
    "full_ascii": lambda r, i: json.dumps(_full(r)),
    "full_utf8": lambda r, i: json.dumps(_full(r), ensure_ascii=False),
    "id_text_ascii": lambda r, i: json.dumps({"id": r.get("id"), "text": r["text"]}),
    "id_text_utf8": lambda r, i: json.dumps({"id": r.get("id"), "text": r["text"]}, ensure_ascii=False),
    "index_id_text_utf8": lambda r, i: json.dumps({"index": i, "id": r.get("id"), "text": r["text"]}, ensure_ascii=False),
    "index_id_text_ascii": lambda r, i: json.dumps({"index": i, "id": r.get("id"), "text": r["text"]}),
    "text_utf8": lambda r, i: json.dumps({"text": r["text"]}, ensure_ascii=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/moe_data")
    # MOE_REFERENCE_MANIFEST / MOE_SOURCE_PARQUET_DIR: offline test hooks only (never set on Colab)
    ap.add_argument("--reference-manifest", default=os.environ.get("MOE_REFERENCE_MANIFEST", str(DATA_REFERENCE / "manifest.json")))
    ap.add_argument("--tokenizer", default=str(DATA_REFERENCE / "tokenizer.json"))
    ap.add_argument("--cache-dir", default="/content/hf_cache")
    ap.add_argument("--keep-cache", action="store_true", help="keep the downloaded parquet (~2.2 GB)")
    ap.add_argument("--source-parquet-dir", default=os.environ.get("MOE_SOURCE_PARQUET_DIR"), help="testing: read local parquet files instead of the Hub")
    args = ap.parse_args()
    ref = read_json(args.reference_manifest)
    args.dataset, args.config, args.revision = ref["dataset"], ref["configuration"], ref["revision"]
    N, MOD = int(ref["source_document_count"]), int(ref["validation_modulus"])
    L = int(ref["context_length"]) + 1
    want = {s: ref["packed"][s] for s in ("train", "validation")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    banner(f"REBUILD FineWeb-Edu data -> {out}  ({args.dataset} {args.config} @ {args.revision[:12]}, first {N:,} docs)")
    report = {"started": now_iso(), "reference_manifest": args.reference_manifest, "checks": {}}

    # ---- 1. tokenizer (must be the original artifact)
    ti = tokenizer_info(args.tokenizer)
    if ti["sha256"] != ref["tokenizer_id"]:
        sys.exit(f"tokenizer sha256 {ti['sha256']} != manifest tokenizer_id {ref['tokenizer_id']}")
    shutil.copy2(args.tokenizer, out / "tokenizer.json")
    report["checks"]["tokenizer_identical"] = True
    tok = load_tokenizer(out / "tokenizer.json")
    eot = ti["eot_id"]
    print(f"  tokenizer verified (sha256 {ti['sha256'][:16]}..., EOT id {eot})")

    # ---- 2. source documents + 3. tokenisation (once)
    recs, ids, lengths = [], [], []
    t0 = time.time()
    buf = []

    def flush():
        for e in tok.encode_batch([r["text"] for r in buf]):
            a = np.asarray(e.ids, dtype=np.uint32)
            ids.append(a)
            lengths.append(a.size)
        recs.extend(buf)
        buf.clear()

    for rec in iter_source_docs(args):
        if len(recs) + len(buf) >= N:
            break
        buf.append(rec)
        if len(buf) >= 4096:
            flush()
            if len(recs) % 49152 == 0:
                print(f"  tokenised {len(recs):,}/{N:,} documents ({time.time() - t0:.0f}s)", flush=True)
    flush()
    if len(recs) != N:
        sys.exit(f"source yielded only {len(recs)} documents, manifest says {N}")
    lengths = np.asarray(lengths, dtype=np.int64)
    print(f"  {N:,} documents, {int(lengths.sum()):,} tokens (without EOT), {time.time() - t0:.0f}s")

    # ---- infer split residue and EOT count from the exact per-split token counts
    idx = np.arange(N)
    cands = []
    for k in (1, 0):
        for r in range(MOD):
            vmask = idx % MOD == r
            vt = int(lengths[vmask].sum()) + k * int(vmask.sum())
            tt = int(lengths[~vmask].sum()) + k * int((~vmask).sum())
            if vt == want["validation"]["source_tokens"] and tt == want["train"]["source_tokens"] \
                    and int(vmask.sum()) == ref["validation_documents"]:
                cands.append((r, k))
    report["checks"]["split_eot_candidates_matching_token_counts"] = cands
    identical_possible = bool(cands)
    if cands:
        r, k = cands[0]
        print(f"  token counts match the manifest EXACTLY for: validation = doc_index % {MOD} == {r}, {k} EOT per document")
    else:
        r, k = 0, 1
        print("  WARNING: no split/EOT convention reproduces the manifest token counts -> rebuild cannot be identical; "
              f"using doc_index % {MOD} == 0 with EOT appended")
    vmask = idx % MOD == r

    def build(split_mask, mode):
        parts = []
        for i in np.nonzero(split_mask)[0]:
            if k and mode == "prepend":
                parts.append(np.array([eot], dtype=np.uint32))
            parts.append(ids[i])
            if k and mode == "append":
                parts.append(np.array([eot], dtype=np.uint32))
        stream = np.concatenate(parts)
        n = stream.size // L
        return stream, stream[: n * L].reshape(n, L)

    # ---- packing mode + dtype: accept only an exact sha256 match on the validation file
    chosen = None
    modes = ("append", "prepend") if k else ("none",)
    for mode in modes:
        stream, arr = build(vmask, mode)
        for dt in DTYPES:
            h = npy_sha(arr.astype(dt))
            if h == want["validation"]["sha256"]:
                chosen = (mode, dt)
                break
        if chosen:
            break
    report["checks"]["validation_npy_identical"] = chosen is not None
    if chosen is None:
        chosen = (modes[0], "uint16")
        print(f"  WARNING: no packing/dtype candidate reproduces validation.npy sha256 -> using {chosen}")
    else:
        print(f"  validation.npy reproduced BYTE-IDENTICALLY (EOT {chosen[0]}, dtype {chosen[1]})")
    mode, dt = chosen

    packed = {}
    for split, mask in (("validation", vmask), ("train", ~vmask)):
        stream, arr = build(mask, mode)
        arr = arr.astype(dt)
        p = out / f"{split}.npy"
        np.save(p, arr)
        h = sha256_file(p)
        same = h == want[split]["sha256"]
        report["checks"][f"{split}_npy_identical"] = same
        packed[split] = {"path": str(p), "examples": int(arr.shape[0]), "source_tokens": int(stream.size),
                         "usable_tokens": int(arr.shape[0] * (L - 1)), "sha256": h}
        print(f"  {split}.npy {arr.shape} {dt}: sha256 {'MATCHES' if same else 'DIFFERS FROM'} the original manifest")
        del stream, arr

    # ---- documents jsonl (format matched against the original sha256 when possible)
    vidx = np.nonzero(vmask)[0]
    variants = list(_VARIANTS)
    fmt = None
    for v in variants:
        h = hashlib.sha256()
        for i in vidx:
            h.update((_VARIANTS[v](recs[i], int(i)) + "\n").encode("utf-8"))
        if h.hexdigest() == (ref.get("source_sha256") or {}).get("validation"):
            fmt = v
            break
    report["checks"]["jsonl_format_identified"] = fmt
    fmt = fmt or "index_id_text_utf8"
    src_sha = {}
    for split, sel in (("validation", vidx), ("train", np.nonzero(~vmask)[0])):
        p = out / f"{split}_documents.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for i in sel:
                f.write(_VARIANTS[fmt](recs[i], int(i)) + "\n")
        src_sha[split] = sha256_file(p)
        report["checks"][f"{split}_jsonl_identical"] = src_sha[split] == (ref.get("source_sha256") or {}).get(split)
    print(f"  documents jsonl written (format {fmt}); identical to original: "
          f"train={report['checks']['train_jsonl_identical']} validation={report['checks']['validation_jsonl_identical']}")

    identical = report["checks"]["train_npy_identical"] and report["checks"]["validation_npy_identical"]
    report["token_data_identical_to_original"] = identical
    report["conventions"] = {"validation_rule": f"doc_index % {MOD} == {r}", "eot_per_doc": k, "eot_placement": mode,
                             "dtype": dt, "row_len": L, "jsonl_format": fmt}
    if identical:
        # the token data is the original: keep the original manifest (+ markers) so every hash check passes unchanged
        shutil.copy2(args.reference_manifest, out / "manifest.json")
        for name in ("source_manifest.json",):
            if (DATA_REFERENCE / name).exists():
                shutil.copy2(DATA_REFERENCE / name, out / name)
        write_json(out / ".complete.json", {"manifest_sha256": sha256_file(out / "manifest.json"), "revision": args.revision})
    else:
        man = dict(ref)
        man.update({"created_at": now_iso(), "script_version": "rebuild-2", "rebuilt": True,
                    "packed": packed, "source_sha256": src_sha, "conventions": report["conventions"]})
        write_json(out / "manifest.json", man)
        write_json(out / ".complete.json", {"manifest_sha256": sha256_file(out / "manifest.json"), "revision": args.revision})
    report["finished"] = now_iso()
    write_json(out / "rebuild_report.json", report)
    if not args.source_parquet_dir and not args.keep_cache and Path(args.cache_dir).exists():
        shutil.rmtree(args.cache_dir, ignore_errors=True)
    banner("REBUILD RESULT: " + ("BYTE-IDENTICAL to the original train.npy / validation.npy" if identical else
                                 "NOT identical to the original (deterministic rebuild; see rebuild_report.json)"))


if __name__ == "__main__":
    main()
