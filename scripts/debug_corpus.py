"""Create a tiny, isolated synthetic fixture for engineering tests only."""
import argparse
import hashlib
import json
from pathlib import Path

from hnet.data import Codec, DOMAINS, sha256_file, train_tokenizer


def make_fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    tokenizer_path = root / "tokenizer.json"
    examples = [
        (f"Box {i} contains {i+1} cubes.", f"В ящике {i} лежит {i+1} кубиков.")
        for i in range(24)
    ]
    tokenizer = train_tokenizer((s for pair in examples[:18] for s in pair), tokenizer_path)
    codec = Codec(tokenizer)
    manifest = []
    split_sha = {}
    for name, pairs in (("train", examples[:18]), ("val", examples[18:21]), ("test", examples[21:])):
        source = root / f"{name}.jsonl"
        with source.open("w", encoding="utf-8") as stream:
            for index, (en, ru) in enumerate(pairs):
                identity = hashlib.sha256(en.encode()).hexdigest()
                row = {"domain": DOMAINS[index % len(DOMAINS)], "pair_id": identity, "group_id": identity,
                       "split": name, "en": {"text": en}, "ru": {"text": ru},
                       "bytes_en": len(en.encode()), "bytes_ru": len(ru.encode()),
                       "bpe_en": len(codec.content(en)), "bpe_ru": len(codec.content(ru)),
                       "alignment": [{"en_span": [0, len(en.encode())], "ru_span": [0, len(ru.encode())],
                                      "verified": True}]}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                manifest.append({key: row[key] for key in ("pair_id", "split", "domain", "bpe_en", "bpe_ru")})
        split_sha[name] = sha256_file(source)
    path = root / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
    (root / "audit.json").write_text(json.dumps({
        "target_bpe": 1000, "train_bpe": sum(row["bpe_en"] + row["bpe_ru"]
                                            for row in manifest if row["split"] == "train"),
        "approved_for_research": False, "benchmark_variants_reviewed": False, "synthetic_fixture": True,
        "split_sha256": split_sha, "tokenizer_sha256": sha256_file(tokenizer_path),
        "manifest_sha256": sha256_file(path),
    }, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="cache/debug-fixture")
    make_fixture(p.parse_args().out)


if __name__ == "__main__":
    main()
