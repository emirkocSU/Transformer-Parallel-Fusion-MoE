"""Train the shared ByteLevel-BPE tokenizer ONCE (vocab 32,000, <|unk|>=0, <|endoftext|>=1).

NOT needed for the current experiment: the tokenizer already exists (data_reference/tokenizer.json,
sha256 2a6d7dca...) and every run verifies that exact artifact. Kept for provenance / re-creation.
Determinism: fixed document subset (first N training documents in source order), single-threaded training.
"""
import argparse
import os

import _common  # noqa: F401

from moefusion.dataset_verify import iter_jsonl
from moefusion.utils import sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-docs", default="/content/moe_data/train_documents.jsonl")
    ap.add_argument("--n-docs", type=int, default=100000)
    ap.add_argument("--out", default="/content/moe_data/tokenizer_retrained.json")
    args = ap.parse_args()
    os.environ["RAYON_NUM_THREADS"] = "1"
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=["<|unk|>", "<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train_from_iterator((r["text"] for r in iter_jsonl(args.train_docs, args.n_docs)), trainer=trainer)
    tok.save(args.out)
    print("saved", args.out, "sha256", sha256_file(args.out))


if __name__ == "__main__":
    main()
