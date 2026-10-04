"""Single shared tokenizer artifact (ByteLevel BPE, vocab 32,000, special tokens <|unk|>=0, <|endoftext|>=1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .utils import sha256_file

EOT_TOKEN = "<|endoftext|>"


def tokenizer_info(path) -> Dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        tj = json.load(f)
    vocab = tj["model"]["vocab"]
    added = {t["content"]: t["id"] for t in tj.get("added_tokens", [])}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "model_type": tj["model"]["type"],
        "vocab_size": len(vocab),
        "special_tokens": added,
        "eot_id": added.get(EOT_TOKEN),
        "pre_tokenizer": (tj.get("pre_tokenizer") or {}).get("type"),
    }


def load_tokenizer(path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))
