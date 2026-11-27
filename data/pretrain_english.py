"""Prepare and audit an English byte/BPE corpus with common document windows."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unicodedata

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from hnet.corpus import sha256_file

REPO = "IFM/TxT360-v2"
REVISION = "a86bdfb101ebaaf71b445f95fb0f9b9bc2a47511"
SHARD = "web-high-medium/Medium-High.chunk0-9fc7d32df3-00000.parquet"
SPLITS = ("train", "val", "test")


def documents(file):
    """Read bounded local batches. No streaming worker survives early exit."""
    import pyarrow.parquet as pq
    with pq.ParquetFile(file) as source:
        for batch in source.iter_batches(
            batch_size=512, use_threads=False, columns=["text", "meta.lang"]
        ):
            for row in batch.to_pylist():
                if row["meta.lang"] == "en" and row["text"]:
                    yield unicodedata.normalize("NFC", row["text"].strip())


def prepare(output, limit):
    from contextlib import ExitStack, closing
    from huggingface_hub import hf_hub_download

    if output.exists():
        raise SystemExit(f"Destination already exists: {output}. Run 'tokenize' to use it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # A full, locally cached shard avoids datasets/PyArrow remote teardown issues.
    file = hf_hub_download(REPO, SHARD, repo_type="dataset", revision=REVISION)
    counts = dict.fromkeys(SPLITS, 0)
    sizes = dict.fromkeys(SPLITS, 0)
    seen = set()
    size = 0
    with TemporaryDirectory(dir=output.parent) as workspace:
        stage = Path(workspace) / "corpus"
        stage.mkdir()
        with ExitStack() as stack:
            files = {key: stack.enter_context((stage / f"{key}.jsonl").open("w", encoding="utf-8"))
                     for key in SPLITS}
            for text in stack.enter_context(closing(documents(file))):
                raw = text.encode("utf-8")
                if not raw:
                    continue
                identity = hashlib.sha256(raw).hexdigest()
                if identity in seen:
                    continue
                bucket = int(identity[:8], 16) % 100
                split = "val" if bucket == 0 else "test" if bucket == 1 else "train"
                files[split].write(json.dumps({"id": identity, "text": text}, ensure_ascii=False) + "\n")
                seen.add(identity)
                counts[split] += 1
                sizes[split] += len(raw)
                size += len(raw)
                if size >= limit:
                    break
        if size < limit or not all(counts.values()):
            raise ValueError(f"Only {size} bytes with split counts {counts} were available")
        report = {"repo": REPO, "revision": REVISION, "shard": SHARD,
                  "documents": counts, "utf8_bytes": sizes}
        (stage / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(output)
    print(json.dumps(report, indent=2))


def window_ranges(data, limit):
    """Cut on UTF-8 boundaries without dropping or duplicating source bytes."""
    start = 0
    while start < len(data):
        end = min(start + limit, len(data))
        while end < len(data) and data[end] & 0xc0 == 0x80:
            end -= 1
        yield start, end
        start = end


class Windows:
    """Index byte windows by JSONL offset; materialize one batch at a time."""
    def __init__(self, path, limit):
        if limit < 4:
            raise ValueError("context_bytes must be at least 4")
        self.path, self.index = path, []
        with path.open("rb") as stream:
            while line := stream.readline():
                offset = stream.tell() - len(line)
                raw = json.loads(line)["text"].encode("utf-8")
                self.index.extend((offset, a, b) for a, b in window_ranges(raw, limit))
        if not self.index:
            raise ValueError(f"No nonempty windows in {path}")

    def read(self, indices):
        result = []
        with self.path.open("rb") as stream:
            previous = None
            for index in indices:
                offset, start, stop = self.index[index]
                if offset != previous:
                    stream.seek(offset)
                    raw = json.loads(stream.readline())["text"].encode("utf-8")
                    previous = offset
                result.append(raw[start:stop])
        return result

    def __len__(self):
        return len(self.index)


def sampled_bytes(windows, count, seed):
    indices = random.Random(seed).sample(range(len(windows)), min(len(windows), count))
    # Stable read order reduces seeks without changing the chosen sample.
    indices.sort()
    for i in range(0, len(indices), 512):
        yield from windows.read(indices[i:i + 512])


def tokenize(data, vocab, context_bytes, sample_windows):
    train = Windows(data / "train.jsonl", context_bytes)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab - 3, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator((raw.decode() for raw in sampled_bytes(train, sample_windows, 42)), trainer)
    bytes_seen = token_count = 0
    for raw in sampled_bytes(train, 4096, 43):
        ids = tokenizer.encode(raw.decode(), add_special_tokens=False).ids
        if tokenizer.decode(ids).encode() != raw:
            raise ValueError("BPE tokenizer failed exact UTF-8 round-trip")
        bytes_seen += len(raw)
        token_count += len(ids)
    total_vocab = tokenizer.get_vocab_size() + 3
    if total_vocab != vocab:
        raise ValueError(f"Tokenizer produced {total_vocab} IDs, expected {vocab}; use more text or a smaller vocabulary.")
    tokenizer_path = data / "tokenizer.json"
    tokenizer.save(str(tokenizer_path.with_suffix(".tmp")))
    tokenizer_path.with_suffix(".tmp").replace(tokenizer_path)
    meta = {"context_bytes": context_bytes, "vocab": total_vocab, "sample_bytes": bytes_seen,
            "sample_tokens": token_count, "bytes_per_bpe": bytes_seen / token_count,
            "split_sha256": {s: sha256_file(data / f"{s}.jsonl") for s in SPLITS},
            "tokenizer_sha256": sha256_file(tokenizer_path)}
    output = data / "pretrain_manifest.json"
    output.with_suffix(".tmp").write_text(json.dumps(meta, indent=2) + "\n")
    output.with_suffix(".tmp").replace(output)
    print(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--out", type=Path, default=Path("cache/txt360-en-500mb"))
    build.add_argument("--bytes", type=int, default=500_000_000)
    bpe = commands.add_parser("tokenize")
    bpe.add_argument("--data", type=Path, required=True)
    bpe.add_argument("--context-bytes", type=int, default=2048)
    bpe.add_argument("--vocab", type=int, default=8192)
    bpe.add_argument("--sample-windows", type=int, default=50_000)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.bytes < 1:
            parser.error("--bytes must be positive")
        prepare(args.out, args.bytes)
    else:
        if args.vocab < 259 or args.sample_windows < 1:
            parser.error("Vocabulary must fit 256 bytes and 3 control IDs, with a positive sample size")
        tokenize(args.data, args.vocab, args.context_bytes, args.sample_windows)


if __name__ == "__main__":
    main()
