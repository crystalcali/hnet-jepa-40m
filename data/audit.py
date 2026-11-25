"""Ingest explicitly approved parallel data and write a reproducible cache."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from tokenizers import Tokenizer
import yaml

from hnet.data import Codec, DOMAINS, QUOTAS, sha256_file, train_tokenizer


def fingerprint(text):
    text = " ".join(unicodedata.normalize("NFC", text).split()).casefold()
    return hashlib.sha256(text.encode()).hexdigest()


def field(row, dotted):
    for name in dotted.split("."):
        row = row[name]
    return row


def source_rows(source):
    if source.get("path"):
        path = Path(source["path"]).expanduser()
        if sha256_file(path) != source["revision"]:
            raise ValueError(f"Local source checksum does not match: {source['name']}")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)
        return
    if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        raise ValueError(f"Pin a full Hugging Face revision for {source['name']}")
    from datasets import load_dataset
    yield from load_dataset(
        source["dataset"], source.get("config"), split=source.get("split", "train"),
        revision=source["revision"], streaming=True,
    )


def side_of(value):
    side = {"text": value} if isinstance(value, str) else dict(value)
    if not side["text"].strip():
        raise ValueError("empty")
    normalized = {}
    for key in ("text", "question", "answer"):
        if isinstance(side.get(key), str):
            normalized[key] = unicodedata.normalize("NFC", side[key])
    for key in ("steps", "blocks"):
        if isinstance(side.get(key), list):
            normalized[key] = [unicodedata.normalize("NFC", part) for part in side[key] if isinstance(part, str)]
    if side["text"] == normalized["text"]:
        normalized["step_byte_spans"] = side.get("step_byte_spans")
    return normalized


def bounds_are_valid(text, span):
    a, b = span
    raw = text.encode("utf-8")
    if not 0 <= a < b <= len(raw):
        return False
    try:
        raw[a:b].decode("utf-8", errors="strict")
        return True
    except UnicodeDecodeError:
        return False


def record_from(source, row, index):
    out = {lang: side_of(field(row, source.get(lang + "_field", lang))) for lang in ("en", "ru")}
    lex = {lang: out[lang]["text"] for lang in ("en", "ru")}
    sizes = {lang: len(text.encode("utf-8")) for lang, text in lex.items()}
    # Latin and Cyrillic evidence, leaving room for numbers, formulas and code.
    if not re.search(r"[A-Za-z]", lex["en"]) or not re.search(r"[А-Яа-яЁё]", lex["ru"]):
        raise ValueError("language")
    if max(sizes.values()) > 8 * min(sizes.values()):
        raise ValueError("length_ratio")
    if sum(sizes.values()) + 6 > 8192:
        raise ValueError("needs_verified_split")
    domain = source.get("domain") or row.get("domain")
    if domain not in DOMAINS:
        raise ValueError("domain")
    alignment = row.get("alignment") or []
    if source.get("sentence_aligned"):
        alignment = [{"en_span": [0, sizes["en"]], "ru_span": [0, sizes["ru"]], "verified": True}]
    elif alignment and any(field(row, source.get(lang + "_field", lang))["text"] != lex[lang]
                           for lang in ("en", "ru")
                           if isinstance(field(row, source.get(lang + "_field", lang)), dict)):
        raise ValueError("alignment_offsets_need_nfc_conversion")
    for span in alignment:
        if not all(bounds_are_valid(lex[lang], span[lang + "_span"]) for lang in ("en", "ru")):
            raise ValueError("alignment")
    if domain in {"math", "logic"} and all(out[lang].get("answer") for lang in ("en", "ru")):
        counts = [Counter(re.findall(r"(?<!\w)[-+]?\d+(?:[.,]\d+)?", out[lang]["answer"]))
                  for lang in ("en", "ru")]
        if any(counts) and counts[0] != counts[1]:
            raise ValueError("answer_numbers")
    h_en, h_ru = (fingerprint(lex[lang]) for lang in ("en", "ru"))
    group = row.get("group_id") or (field(row, source["group_field"]) if source.get("group_field") else None)
    group = str(group or h_en)
    val = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    split = "test" if val == 0 else "val" if val == 1 else "train"
    out.update(pair_id=hashlib.sha256((h_en + h_ru).encode()).hexdigest(), group_id=group,
               split=split, domain=domain, alignment=alignment, bytes_en=sizes["en"], bytes_ru=sizes["ru"],
               source={"repo": source.get("dataset", source["name"]), "revision": source["revision"],
                       "config": source.get("config"), "split": source.get("split", "train"),
                       "row_id": row.get("id", index), "license": source["license"]})
    return out, h_en, h_ru


def holdout_hashes(path):
    hashes = set()
    if path:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                for lang in ("en", "ru"):
                    side = row.get(lang)
                    if side:
                        for value in ([side] if isinstance(side, str) else [side.get("text"), side.get("question")]):
                            if value:
                                hashes.add(fingerprint(value))
    return hashes


def dump_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def audit(args):
    config = yaml.safe_load(Path(args.config).read_text())
    sources = config.get("sources") or []
    if not sources:
        raise SystemExit("Configure licensed, paired sources in data_sources.yaml before corpus preparation.")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ("audit.json", "candidates.sqlite", "train.jsonl")):
        raise SystemExit("Choose a fresh audit destination; existing corpus caches are immutable.")
    holdout = holdout_hashes(config.get("benchmark_holdout"))
    rejected, accepted = Counter(), Counter()
    available_bytes = Counter()
    db = sqlite3.connect(root / "candidates.sqlite")
    db.execute("CREATE TABLE pairs (id INTEGER PRIMARY KEY, en TEXT UNIQUE, ru TEXT UNIQUE, split TEXT, domain TEXT, data TEXT)")
    for source in sources:
        name = source["name"]
        if not all(source.get(key) for key in ("license_reviewed", "pairing_reviewed", "license", "revision")):
            rejected[name + ":unreviewed_source"] += 1
            continue
        source_count = 0
        for i, row in enumerate(source_rows(source)):
            if args.max_rows and i >= args.max_rows:
                break
            try:
                item, en, ru = record_from(source, row, i)
                hashes = {en, ru}
                hashes.update(fingerprint(item[lang]["question"]) for lang in ("en", "ru")
                              if item[lang].get("question"))
                if hashes & holdout:
                    raise ValueError("benchmark_exact_overlap")
                db.execute("INSERT INTO pairs(en, ru, split, domain, data) VALUES (?,?,?,?,?)",
                           (en, ru, item["split"], item["domain"], json.dumps(item, ensure_ascii=False)))
                source_count += 1
                available_bytes[item["domain"]] += item["bytes_en"] + item["bytes_ru"]
            except sqlite3.IntegrityError:
                rejected[name + ":duplicate_side"] += 1
            except (ValueError, KeyError, TypeError, UnicodeError) as error:
                rejected[name + ":" + str(error)] += 1
            if i % 2000 == 0:
                db.commit()
        db.commit()
        accepted[name] = source_count
    if not sum(accepted.values()):
        raise SystemExit(f"No usable data: {dict(rejected)}")
    tokenizer_path = root / "tokenizer.json"
    if args.tokenizer:
        tokenizer_path.write_bytes(Path(args.tokenizer).read_bytes())
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        def texts():
            for domain in DOMAINS:
                cursor = db.execute("SELECT data FROM pairs WHERE split='train' AND domain=? LIMIT 50000", (domain,))
                for (line,) in cursor:
                    record = json.loads(line)
                    yield record["en"]["text"]
                    yield record["ru"]["text"]
        tokenizer = train_tokenizer(texts(), tokenizer_path)
    codec = Codec(tokenizer)
    targets = dict(zip(DOMAINS, (round(args.target_bpe * q) for q in QUOTAS)))
    totals, coverage = Counter(), Counter()
    splits = {name: (root / (name + ".jsonl")).open("w", encoding="utf-8") for name in ("train", "val", "test")}
    manifest = (root / "manifest.jsonl").open("w", encoding="utf-8")
    try:
        for (line,) in db.execute("SELECT data FROM pairs ORDER BY id"):
            item = json.loads(line)
            split, domain = item["split"], item["domain"]
            if split == "train" and totals[domain] >= targets[domain]:
                continue
            for lang in ("en", "ru"):
                value = item[lang]["text"]
                ids = codec.content(value)
                if codec.decode(ids) != value:
                    raise RuntimeError("BPE round-trip mismatch")
                item["bpe_" + lang] = len(ids)
            if split == "train":
                totals[domain] += item["bpe_en"] + item["bpe_ru"]
            coverage[split + "_pairs"] += 1
            coverage[split + "_verified_spans"] += sum(bool(span.get("verified")) for span in item["alignment"])
            coverage[split + "_task_sides"] += sum(bool(item[lang].get("question") and item[lang].get("answer"))
                                                  for lang in ("en", "ru"))
            coverage[split + "_trace_sides"] += sum(bool((item[lang].get("question") and item[lang].get("answer") and item[lang].get("steps")) or len(item[lang].get("blocks") or []) > 1)
                                                   for lang in ("en", "ru"))
            splits[split].write(json.dumps(item, ensure_ascii=False) + "\n")
            manifest.write(json.dumps({key: item[key] for key in (
                "pair_id", "group_id", "split", "domain", "source", "bytes_en", "bytes_ru", "bpe_en", "bpe_ru"
            )}, ensure_ascii=False) + "\n")
    finally:
        for stream in [*splits.values(), manifest]:
            stream.close()
        db.close()
    deficits = {name: max(0, targets[name] - totals[name]) for name in DOMAINS}
    declared_coverage = bool(config.get("benchmark_reviewed") and holdout)
    checks = bool(config.get("near_dedup_reviewed") and config.get("quality_reviewed"))
    objective_coverage = all(coverage[name] for name in (
        "train_verified_spans", "train_task_sides", "train_trace_sides", "val_pairs"
    ))
    audit_result = {
        "target_bpe": args.target_bpe, "train_bpe": sum(totals.values()), "targets": targets,
        "domain_bpe": dict(totals), "available_bytes": available_bytes, "deficits": deficits,
        "accepted_by_source": accepted, "rejections": rejected, "coverage": coverage,
        "benchmark_exact_hashes": len(holdout), "benchmark_variants_reviewed": declared_coverage,
        "required_objective_coverage": objective_coverage,
        "near_dedup_and_quality_reviewed": checks, "truncated_scan": bool(args.max_rows),
        "tokenizer_sha256": sha256_file(tokenizer_path), "manifest_sha256": sha256_file(root / "manifest.jsonl"),
        "split_sha256": {name: sha256_file(root / f"{name}.jsonl") for name in splits},
        "approved_for_research": (
            not any(deficits.values()) and not args.max_rows and declared_coverage and checks and objective_coverage
            and bool(args.target_bpe < 1_000_000_000 or args.tokenizer)
        ),
    }
    dump_json(root / "deficits.json", deficits)
    dump_json(root / "audit.json", audit_result)
    (root / "corpus_report.md").write_text(
        "# Corpus audit\n\n" + "\n".join(f"- {key} — {value}" for key, value in audit_result.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit_result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-bpe", required=True, type=int)
    parser.add_argument("--out", default="cache/pilot")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--tokenizer", help="Reuse the exact pilot tokenizer for the full audit")
    args = parser.parse_args()
    if args.target_bpe < 1:
        parser.error("--target-bpe must be positive")
    audit(args)


if __name__ == "__main__":
    main()
