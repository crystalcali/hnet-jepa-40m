"""Score pinned, normalized benchmark exports without executing dataset code."""
from collections import Counter, defaultdict
import json
import re

import torch

from generate import generate
from .training import amp, nll


def read_cases(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                case = json.loads(line)
                if not case.get("dataset") or not case.get("id"):
                    raise ValueError("Every benchmark row must identify its dataset and case")
                if case.get("language") not in {"en", "ru"}:
                    raise ValueError("Benchmark cases must specify en or ru")
                yield case


@torch.inference_mode()
def likelihood(model, codec, prompt, continuation, language, device):
    if not continuation:
        raise ValueError("Benchmark continuations must be nonempty")
    context = codec.ids(prompt, language, eos=False)
    ending = codec.content(continuation)
    if len(prompt.encode()) + len(continuation.encode()) + 2 > 8192:
        raise ValueError("Benchmark case exceeds the raw context limit")
    sequence = context + ending
    inputs = torch.tensor([sequence[:-1]], device=device)
    labels = torch.tensor([ending], device=device)
    with amp(device):
        logits = model(inputs)[:, len(context)-1:]
    return -float(nll(logits, labels, "sum"))


def normalize(answer, compare):
    answer = answer.strip()
    if compare == "integer":
        return str(int(answer)) if re.fullmatch(r"[-+]?\d+", answer) else None
    if compare == "normalized_text":
        return " ".join(answer.casefold().split())
    if compare != "exact":
        raise ValueError(f"Unknown comparison: {compare}")
    return answer


def plain_benchmarks(model, codec, path, iterations=1):
    device = next(model.parameters()).device
    correct, trials, invalid = Counter(), Counter(), Counter()
    translations, references = defaultdict(list), defaultdict(list)
    for row in read_cases(path):
        dataset, lang = row["dataset"], row["language"]
        key = f"{dataset}/{lang}"
        task = row["kind"]
        trials[key] += 1
        if task == "choice":
            scores = []
            for continuation in row["choices"]:
                value = likelihood(model, codec, row["prompt"], continuation, lang, device)
                if row.get("score") == "per_byte":
                    value /= len(continuation.encode())
                scores.append(value)
            chosen = max(range(len(scores)), key=scores.__getitem__)
            correct[key] += int(chosen == int(row["answer_index"]))
        elif task in {"exact", "translation"}:
            mode = row.get("mode", "task" if task == "exact" else "translation")
            out = generate(model, codec, row["prompt"], lang, mode=mode, iterations=iterations,
                           source_language=row.get("source_language"), max_bytes=row.get("max_bytes", 128))
            invalid[key] += not out["valid_utf8"]
            if task == "translation":
                translations[key].append(out["text"])
                references[key].append(row["answer"])
            else:
                compare = row.get("compare", "exact")
                correct[key] += int(normalize(out["text"], compare) == normalize(row["answer"], compare))
        else:
            raise ValueError(f"Unknown benchmark kind: {task}")
    result = {}
    for key, count in trials.items():
        if key in translations:
            from sacrebleu.metrics import BLEU, CHRF
            pred, refs = translations[key], references[key]
            bleu = BLEU()
            bleu_value = bleu.corpus_score(pred, [refs]).score
            chrf = CHRF(word_order=2)
            result[key] = {"n": count, "bleu": bleu_value, "bleu_signature": str(bleu.get_signature()),
                           "chrf++": chrf.corpus_score(pred, [refs]).score, "invalid_utf8": invalid[key]}
        else:
            result[key] = {"n": count, "accuracy": correct[key] / count, "invalid_utf8": invalid[key]}
    return result
