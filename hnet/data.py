"""Audited pair cache, deterministic curriculum and byte/BPE collation."""
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .corpus import Codec, DOMAINS, QUOTAS, sha256_file, train_tokenizer

CURRICULUM = (
    (.10, (90, 10, 0, 0, 0, 0)), (.25, (75, 20, 5, 0, 0, 0)),
    (.50, (55, 25, 12, 3, 3, 2)), (.75, (40, 22, 20, 7, 6, 5)),
    (float("inf"), (30, 20, 23, 9, 9, 9)),
)


def pad(sequences, pad_id, device):
    if not sequences or any(not value for value in sequences):
        raise ValueError("empty sequence batch")
    length = max(map(len, sequences))
    return torch.tensor([value + [pad_id] * (length - len(value)) for value in sequences], device=device)


def lm_batch(codec, rows, device):
    texts = [record[lang]["text"] for record in rows for lang in ("en", "ru")]
    langs = ["en", "ru"] * len(rows)
    ids = pad([codec.ids(text, lang) for text, lang in zip(texts, langs)], codec.pad, device)
    labels = ids[:, 1:].clone()
    labels[labels.eq(codec.pad) | labels.eq(codec.lang("en")) | labels.eq(codec.lang("ru"))] = -100
    return ids[:, :-1].contiguous(), labels.contiguous(), texts


def conditional_batch(codec, examples, device):
    parts, targets, labels, prefixed = [], [], [], []
    for a, b, al, bl, task in examples:
        if len(a.encode("utf-8")) + len(b.encode("utf-8")) + (6 if task else 5) > 8192:
            raise ValueError("conditional example exceeds the shared raw context")
        prompt = codec.ids(a, al, eos=True, task=task)
        target = codec.ids(b, bl, eos=True)
        parts.append(prompt)
        targets.append(target[:-1])
        labels.append([-100] + target[2:])
        prefixed.append(prompt + target)
    if codec.tokenizer is None:
        return pad(parts, codec.pad, device), pad(targets, codec.pad, device), pad(labels, -100, device)
    merged_labels = [[-100] * len(a) + b for a, b in zip(parts, labels)]
    inputs = pad([x[:-1] for x in prefixed], codec.pad, device)
    return None, inputs, pad(merged_labels, -100, device)


class PairCache:
    def __init__(self, path, split="train"):
        self.path = Path(path)
        self.meta = json.loads((self.path / "audit.json").read_text())
        if sha256_file(self.path / "tokenizer.json") != self.meta["tokenizer_sha256"]:
            raise ValueError("tokenizer checksum mismatch")
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self.file = (self.path / f"{split}.jsonl").open("rb")
        self.indices = {d: [] for d in DOMAINS}
        cursor = 0
        for line in self.file:
            row = json.loads(line)
            size = row["bpe_en"] + row["bpe_ru"]
            self.indices[row["domain"]].append((cursor, size))
            cursor += len(line)
        self.file.seek(0)

    def get(self, offset):
        self.file.seek(offset)
        return json.loads(self.file.readline())

    def records(self):
        for group in self.indices.values():
            for offset, _ in group:
                yield self.get(offset)

    def close(self):
        self.file.close()


class CurriculumSampler:
    def __init__(self, cache, seed, horizon):
        self.cache = cache
        self.rng = random.Random(seed)
        self.horizon = horizon
        self.exposure = 0
        self.visits = 0
        self.order = {}
        self.cursor = dict.fromkeys(DOMAINS, 0)
        self.epochs = dict.fromkeys(DOMAINS, 0)
        self.domain_exposure = dict.fromkeys(DOMAINS, 0)
        self.stage_exposure = dict.fromkeys(DOMAINS, 0)
        self.stage_limit = None
        for domain in DOMAINS:
            self.reset(domain, first=True)

    def reset(self, domain, first=False):
        self.order[domain] = list(range(len(self.cache.indices[domain])))
        self.rng.shuffle(self.order[domain])
        self.cursor[domain] = 0
        if not first:
            self.epochs[domain] += 1

    @property
    def progress(self):
        return min(1., self.exposure / self.horizon)

    def next(self):
        limit, quotas = next(pair for pair in CURRICULUM if self.progress < pair[0])
        if self.stage_limit != limit:
            self.stage_exposure = dict.fromkeys(DOMAINS, 0)
            self.stage_limit = limit
        candidates = [(domain, q) for domain, q in zip(DOMAINS, quotas) if q and self.order[domain]]
        if not candidates:
            raise ValueError("No eligible records in the current curriculum stage")
        # For equal size records, this converges to the specified exposure quotas.
        domain = min(candidates, key=lambda item: self.stage_exposure[item[0]] / item[1])[0]
        if self.cursor[domain] == len(self.order[domain]):
            self.reset(domain)
        index = self.order[domain][self.cursor[domain]]
        self.cursor[domain] += 1
        offset, size = self.cache.indices[domain][index]
        self.exposure += size
        self.visits += 1
        self.domain_exposure[domain] += size
        self.stage_exposure[domain] += size
        return self.cache.get(offset)

    def snapshot(self):
        return {name: getattr(self, name) for name in (
            "exposure", "visits", "order", "cursor", "epochs", "domain_exposure",
            "stage_exposure", "stage_limit"
        )} | {"rng": self.rng.getstate(), "horizon": self.horizon}

    def restore(self, data):
        if data["horizon"] != self.horizon:
            raise ValueError("Resume requires identical planned exposure")
        for key, value in data.items():
            if key == "rng":
                self.rng.setstate(value)
            else:
                setattr(self, key, value)
