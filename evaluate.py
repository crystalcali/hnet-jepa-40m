"""Evaluate byte likelihood and paired K sweeps on held-out data."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch

from generate import generate, load_model
from hnet.benchmarks import plain_benchmarks
from hnet.data import PairCache, lm_batch, sha256_file
from hnet.synthetic_eval import tasks
from hnet.training import amp, nll
from hnet.model import masked_mean


@torch.inference_mode()
def language_eval(model, codec, cache, device, limit):
    totals = defaultdict(lambda: [0., 0])
    latent = {1: [[], []], 2: [[], []]}
    lengths = [0, 0, 0, 0]
    for idx, row in enumerate(cache.records()):
        if idx >= limit:
            break
        ids, targets, _ = lm_batch(codec, [row], device)
        targets[targets.eq(codec.eos)] = -100
        with amp(device):
            if hasattr(model, "routers"):
                encoded = model.encode(ids)
                logits = model.decode(model.reason(encoded.latents[-1]), encoded)
                if all(bool(v) for v in encoded.content[-1].any(1)):
                    for level in latent:
                        z = masked_mean(encoded.latents[level], encoded.content[level]).cpu()
                        for lang in range(2):
                            latent[level][lang].append(z[lang])
                lengths[0] += int(ids.ne(codec.pad).sum())
                for level, route in enumerate(encoded.routes, start=1):
                    lengths[level] += int(route.output_valid.sum())
            else:
                logits = model(ids)
        for row_idx, lang in enumerate(("en", "ru")):
            value = float(nll(logits[row_idx:row_idx+1], targets[row_idx:row_idx+1], "sum"))
            for key in (lang, lang + "/" + row["domain"], "aggregate"):
                totals[key][0] += value
                totals[key][1] += row["bytes_" + lang]
    metric = {key: loss / (.6931471805599453 * count) for key, (loss, count) in totals.items() if count}
    representations = {}
    for level, (en, ru) in latent.items():
        if len(en) < 2:
            continue
        a, b = torch.stack(en), torch.stack(ru)
        similarity = torch.nn.functional.normalize(a) @ torch.nn.functional.normalize(b).T
        index = torch.arange(len(a))
        report = {}
        for name, matrix in (("en_to_ru", similarity), ("ru_to_en", similarity.T)):
            ranks = (matrix.argsort(descending=True) == index[:, None]).nonzero()[:, 1] + 1
            report[name] = {"recall_1": float(ranks.eq(1).float().mean()),
                            "recall_10": float(ranks.le(10).float().mean()),
                            "mrr": float(ranks.float().reciprocal().mean())}
        x = torch.cat((a, b))
        eigen = torch.linalg.svdvals(x - x.mean(0)).square()
        probability = eigen / eigen.sum().clamp_min(1e-12)
        report.update(pair_count=len(a), chance=1/len(a), variance=float(x.var(0).mean()),
                      effective_rank=float(torch.exp(-torch.special.xlogy(probability, probability).sum())))
        representations[f"L{level+1}"] = report
    return {"bpb": metric, "representations": representations,
            "hard_ratios": [a / b for a, b in zip(lengths, lengths[1:]) if b],
            "compression": lengths[0] / lengths[-1] if lengths[-1] else None}


def interval(values):
    if not values:
        return None
    rng = random.Random(42)
    boot = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(1000))
    return [boot[25], boot[974]]


def exact_reasoning(model, codec, ks, count, mode, seed):
    times, grouped, paired = defaultdict(list), defaultdict(list), {}
    test = list(tasks(count, seed))
    device = next(model.parameters()).device
    for task in test:
        for lang in ("en", "ru"):
            for k in ks:
                started = time.monotonic()
                result = generate(model, codec, task["prompt"][lang], lang, mode=mode, iterations=k, max_bytes=32)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                correct = result["text"].strip() == task["answer"]
                times[k].append(time.monotonic() - started)
                grouped[(task["level"], lang, k)].append(correct)
                paired[(task["id"], lang, k)] = correct
    primary = []
    if 1 in ks and 8 in ks:
        for task in test:
            if 2 <= task["level"] <= 5:
                primary.append(statistics.mean(int(paired[(task["id"], lang, 8)])
                                               - int(paired[(task["id"], lang, 1)]) for lang in ("en", "ru")))
    return {"mode": mode, "generator_sha256": sha256_file(Path(__file__).parent / "hnet/synthetic_eval.py"),
            "seed": seed, "per_level_language": {f"{level}/{lang}/{k}": {"n": len(x), "accuracy": statistics.mean(x)}
                                                   for (level, lang, k), x in sorted(grouped.items())},
            "mean_latency_s": {k: statistics.mean(x) for k, x in times.items()},
            "primary_K8_minus_K1": statistics.mean(primary) if primary else None,
            "paired_bootstrap_95ci": interval(primary)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--iterations", default="1,2,4,8,16,32")
    parser.add_argument("--limit-pairs", type=int, default=256)
    parser.add_argument("--reasoning-per-level", type=int, default=0)
    parser.add_argument("--mode", choices=("lm", "task", "both"), default="task")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--benchmark", type=Path, help="Pinned benchmark export in normalized JSONL")
    parser.add_argument("--benchmark-iterations", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, codec, config = load_model(args.checkpoint, torch.device(args.device))
    cache = PairCache(args.data, args.split)
    if sha256_file(args.data / f"{args.split}.jsonl") != cache.meta["split_sha256"][args.split]:
        raise SystemExit("Evaluation split checksum mismatch")
    result = language_eval(model, codec, cache, torch.device(args.device), args.limit_pairs)
    if args.benchmark:
        result["benchmarks"] = {
            "input_sha256": sha256_file(args.benchmark),
            "iterations": args.benchmark_iterations,
            "results": plain_benchmarks(model, codec, args.benchmark, args.benchmark_iterations),
        }
    if args.reasoning_per_level:
        ks = [int(x) for x in args.iterations.split(",")] if config["model"] == "C" else [1]
        modes = ("lm", "task") if args.mode == "both" else (args.mode,)
        seed = 852173 if args.split == "test" else 34789
        result["reasoning"] = [exact_reasoning(model, codec, ks, args.reasoning_per_level, mode, seed)
                               for mode in modes]
    result.update(split=args.split, model=config["model"],
                  checkpoint_sha256=sha256_file(Path(args.checkpoint) / "model.safetensors"))
    result["benchmark_reviewed"] = cache.meta["benchmark_variants_reviewed"]
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.out:
        args.out.write_text(serialized + "\n", encoding="utf-8")
    cache.close()


if __name__ == "__main__":
    main()
