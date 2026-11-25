"""Adapters for pinned public RU–EN pilot datasets. No text is generated here."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import unicodedata
from urllib.parse import quote
from urllib.request import urlopen

import pyarrow.parquet as parquet
import yaml


def download(source, file):
    if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        raise ValueError(f"Pin an exact revision for {source['dataset']}")
    root = source.get("raw_cache", "cache/pilot_raw")
    path = Path(root) / source["dataset"].replace("/", "__") / file
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{source['dataset']}/resolve/{source['revision']}/{quote(file)}"
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with urlopen(url, timeout=45) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
            temporary.replace(path)
            return path
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def parquet_rows(path, language=None):
    data = parquet.ParquetFile(path)
    groups = None
    if language:
        field = data.schema.names.index("og_language")
        groups = []
        for i in range(data.num_row_groups):
            statistics = data.metadata.row_group(i).column(field).statistics
            if statistics is None or not statistics.has_min_max or statistics.min <= language <= statistics.max:
                groups.append(i)
    for batch in data.iter_batches(batch_size=512, row_groups=groups):
        yield from batch.to_pylist()


def canonical(text):
    return unicodedata.normalize("NFC", text.strip())


def pair(en, ru, key, *, aligned=False, group=None, domain=None):
    item = dict(id=key, en=en, ru=ru, group_id=group or key)
    if domain:
        item["domain"] = domain
    if aligned:
        sizes = {lang: len(item[lang]["text"].encode()) for lang in ("en", "ru")}
        item["alignment"] = [{"en_span": [0, sizes["en"]], "ru_span": [0, sizes["ru"]], "verified": True}]
    return item


def texts(en, ru):
    return {"text": canonical(en)}, {"text": canonical(ru)}


def numbers(text):
    return Counter(re.findall(r"\d+(?:[.,]\d+)?", text))


def key_from(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def make_task(question, solution, answer):
    question, solution, answer = map(canonical, (question, solution, answer))
    steps = [line.strip() for line in re.split(r"\n\s*\n|\n(?=(?:Step|Шаг)\s+\d)", solution) if line.strip()]
    return {"text": "\n\n".join((question, solution, answer)),
            "question": question, "steps": steps if len(steps) > 1 else [], "answer": answer}


def stories(row, key):
    en, ru = texts(row["text_en"], row["text_ru"])
    # Authors describe the English texts as adaptations. The gold summaries
    # provide a second anchor for auditing, while local alignment remains unset.
    if not row["summary_ru"].strip() or not row["summary_en"].strip():
        return
    if len(en["text"].encode()) + len(ru["text"].encode()) > 8186:
        en, ru = texts(row["summary_en"], row["summary_ru"])
        key += ":summary"
    yield pair(en, ru, key, group="story:" + key_from(row["summary_en"]))


def education(row, key):
    if row["og_language"] != "rus_Cyrl" or row["early_stop"] or (row["og_language_score"] or 0) < .90:
        return
    original, translated = row["og_chunks"], row["translated_chunks"]
    if not original or not translated or len(original) != len(translated):
        return
    document = "finedu:" + key_from(row["url"] or row["id"])
    for index, (ru, en) in enumerate(zip(original, translated, strict=True)):
        if en and ru and len(en) > 60 and len(ru) > 60:
            a, b = texts(en, ru)
            yield pair(a, b, f'{row["id"]}:{index}', aligned=True, group=document)


def mathematics(row, key):
    en, ru = row["problem"], row["ru_problem"]
    a, b = row["generated_solution"], row["ru_generated_solution"]
    answer = row["expected_answer"]
    if not all((en, ru, a, b, answer)) or numbers(en) != numbers(ru):
        return
    sides = make_task(en, a, answer), make_task(ru, b, answer)
    grouped = "math:" + key_from(en)
    item = pair(*sides, key, group=grouped)
    item["alignment"] = [{
        "en_span": [0, len(sides[0]["question"].encode())],
        "ru_span": [0, len(sides[1]["question"].encode())],
        "verified": True,
    }]
    yield item


def python_prompts(row, key):
    if (row["lang"] or "").casefold() != "python":
        return
    if not row["query"] or not row["ru_query"] or numbers(row["query"]) != numbers(row["ru_query"]):
        return
    en, ru = texts(row["query"], row["ru_query"])
    yield pair(en, ru, key, aligned=True, group="python:" + key_from(row["query"]))


def joined(source):
    en_path, ru_path = (download(source, file) for file in source["files"])
    a, b = list(parquet_rows(en_path)), list(parquet_rows(ru_path))
    # This publisher releases a single parallel corpus projected into EN and RU
    # files. Verify the complete shared metadata sequence before taking any rows.
    fields = ("topic", "correct_option_number")
    if [tuple(row[f] for f in fields) for row in a] != [tuple(row[f] for f in fields) for row in b]:
        raise ValueError("The pinned Logic-701 translation files disagree on alignment")
    topics = {"Optimization of actions and planning", "Operation of mechanisms", "Functions and Algorithms"}
    for i, (en, ru) in enumerate(zip(a, b, strict=True)):
        if not en["solution"] or not ru["solution"]:
            continue
        if numbers(en["problem_statement"]) != numbers(ru["problem_statement"]):
            continue
        sides = []
        for record in (en, ru):
            question = canonical(record["problem_statement"]) + "\n" + "\n".join(
                f'{j}. {record[f"answer_option_{j}"]}' for j in range(1, 6)
            )
            sides.append(make_task(question, record["solution"], str(record["correct_option_number"])))
        key = f"logic701:{i}"
        domain = "reasoning" if en["topic"] in topics else "logic"
        yield pair(*sides, key, aligned=True, domain=domain)


def rows(source):
    adapter = source["pilot_adapter"]
    if adapter == "logic701":
        yield from joined(source)
        return
    adapt = {"stories": stories, "education": education, "math": mathematics, "python": python_prompts}[adapter]
    for file in source["files"]:
        data = parquet_rows(download(source, file), "rus_Cyrl" if adapter == "education" else None)
        for i, row in enumerate(islice(data, source.get("scan_rows_per_shard"))):
            yield from adapt(row, f"{file}:{i}")


def holdout(config, destination):
    with Path(destination).open("w", encoding="utf-8") as target:
        for src in config["heldout_sources"]:
            path = download(src, src["file"])
            for row in parquet_rows(path):
                value = row.get("question")
                if value:
                    target.write(json.dumps({src["language"]: value}, ensure_ascii=False) + "\n")
