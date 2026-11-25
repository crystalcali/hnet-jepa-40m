"""Train one experiment arm on an audited parallel corpus."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from datetime import datetime, timezone

import numpy as np
from safetensors.torch import save_file
import torch
import yaml

from hnet.compute import ComputeEstimate
from hnet.data import Codec, CurriculumSampler, PairCache, lm_batch, sha256_file
from hnet.diagnostics import probe, select_probe
from hnet.losses import loss_weights
from hnet.model import build_model, count_parameters
from hnet.training import amp, batch_objectives, generative_rng, lr_at, nll, sigreg_step

STOP = False
MILESTONES = (1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000, 300_000_000, 1_000_000_000)


def interrupt(*_):
    global STOP
    STOP = True


def json_line(path, values):
    line = json.dumps(values, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def rng_state(device, rng):
    return {"py": random.getstate(), "np": np.random.get_state(), "cpu": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "local": rng.getstate()}


def restore_rng(state, device, rng):
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.random.set_rng_state(state["cpu"])
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])
    rng.setstate(state["local"])


def checkpoint(root, run_state, model, optimizer, sampler, identity, config, data, device, rng,
               milestone=False):
    tag = f"step-{run_state['step']:09d}"
    export = root / "checkpoints"
    export.mkdir(exist_ok=True)
    final = export / tag
    temp = export / ("." + tag + ".tmp")
    if not final.exists():
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "sampler": sampler.snapshot(), "rng": rng_state(device, rng),
                    "run": run_state, "identity": identity}, temp / "training.pt")
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, temp / "model.safetensors")
        (temp / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        shutil.copyfile(data / "tokenizer.json", temp / "tokenizer.json")
        if milestone:
            (temp / "milestone").touch()
        os.replace(temp, final)
    latest_tmp = root / ".latest.tmp"
    latest_tmp.unlink(missing_ok=True)
    latest_tmp.symlink_to(Path("checkpoints") / final.name, target_is_directory=True)
    os.replace(latest_tmp, root / "latest")
    recent = sorted((p for p in export.glob("step-*") if not (p / "milestone").exists()), reverse=True)
    for old in recent[3:]:
        shutil.rmtree(old)
    return str(final)


def load_state(path, root, model, optimizer, sampler, identity, device, rng):
    path = root / "latest" if path == "latest" else Path(path)
    if not path.is_dir() or not (path / "training.pt").is_file():
        raise FileNotFoundError(f"Trusted local checkpoint not found: {path}")
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if state["identity"] != identity:
        raise ValueError("Resume requires the same config, seed, dataset and tokenizer")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    sampler.restore(state["sampler"])
    restore_rng(state["rng"], device, rng)
    return state["run"]


@torch.no_grad()
def validate(model, codec, cache, device, max_examples=24):
    if not cache or not any(cache.indices.values()):
        return {}
    model.eval()
    totals = {"en": 0., "ru": 0.}
    sizes = Counter()
    try:
        for i, row in enumerate(cache.records()):
            if i >= max_examples:
                break
            ids, labels, _ = lm_batch(codec, [row], device)
            labels[labels.eq(codec.eos)] = -100
            with amp(device):
                logits = model(ids)
            for index, lang in enumerate(("en", "ru")):
                totals[lang] += float(nll(logits[index:index+1], labels[index:index+1], "sum"))
                sizes[lang] += row["bytes_" + lang]
    finally:
        model.train()
    return {lang: totals[lang] / (np.log(2) * n) for lang, n in sizes.items() if n}


@torch.no_grad()
def calibrate_router(model, codec, cache, device):
    """Initialize per-level thresholds with sequential routing on a train-only probe."""
    rows = []
    for row in cache.records():
        rows.append(row)
        if len(rows) >= 24:
            break
    for level, goal in enumerate((4, 3, 3)):
        values = []
        for start in range(0, len(rows), 2):
            ids, _, _ = lm_batch(codec, rows[start:start+2], device)
            with amp(device):
                routed = model.encode(ids).routes[level]
            p = routed.probs[routed.valid & ~routed.forced]
            if p.numel():
                values.append(p.cpu())
        if values:
            threshold = torch.cat(values).quantile(1 - 1 / goal).clamp(.0001, .9999)
            model.routers[level].threshold.copy_(threshold.to(device))
    return [float(router.threshold) for router in model.routers]


def gpu_memory(device):
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.


def update(model, optimizer, codec, rows, device, rng, K, progress, micro, config, step):
    with ComputeEstimate(model) as estimate:
        result = accumulate(model, optimizer, codec, rows, device, rng, K, progress, micro, config, step)
    # The forward is counted with replayed layers. A 2x backward estimate
    # approximates training for contractions; this is not a hardware counter.
    result["estimated_training_flops"] = 3 * estimate.flops
    return result


def accumulate(model, optimizer, codec, rows, device, rng, K, progress, micro, config, step):
    segments = [rows[start:start+micro] for start in range(0, len(rows), micro)]
    byte_count = sum(sum(len(codec.content(row[lang]["text"])) + 1 for lang in ("en", "ru")) for row in rows)
    weights = loss_weights(progress)
    totals, weighted_totals, coverage = Counter(), Counter(), Counter()
    diagnostics, sig_inputs = [], []
    optimizer.zero_grad(set_to_none=True)
    for group in segments:
        weight = len(group) / len(rows)
        count = sum(sum(len(codec.content(row[lang]["text"])) + 1 for lang in ("en", "ru")) for row in group)
        with amp(device):
            gen = generative_rng(group, step, config.get("runtime_seed", 42))
            _, terms, weighted, counts, debug = batch_objectives(
                model, codec, group, K, weights, rng, device, progress, count / byte_count, gen_rng=gen
            )
            loss = sum(value if name in ("byte", "aux") else value * weight for name, value in weighted.items())
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite objective")
        loss.backward()
        totals.update({key: float(value.detach()) * weight for key, value in terms.items()})
        weighted_totals.update({
            key: float(value.detach()) * (1 if key in ("byte", "aux") else weight)
            for key, value in weighted.items()
        })
        coverage.update(counts)
        if debug:
            diagnostics.append(debug)
        if config["model"] == "C" and weights["sigreg"]:
            sig_inputs.append(lm_batch(codec, group, device)[0])
    sig_counts = {}
    if sig_inputs:
        sig, sig_counts = sigreg_step(model, sig_inputs, step, device, config.get("sigreg_min", 256))
        if sig is not None:
            if not torch.isfinite(sig):
                raise FloatingPointError("non-finite SIGReg")
            (weights["sigreg"] * sig).backward()
            totals["sigreg"] = float(sig.detach())
            weighted_totals["sigreg"] = weights["sigreg"] * totals["sigreg"]
            coverage["sigreg"] = 1
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    lr = lr_at(progress, config)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return {"loss": sum(weighted_totals.values()), "raw": dict(totals), "weighted": dict(weighted_totals),
            "coverage": dict(coverage), "sigreg_counts": sig_counts, "lr": lr, "grad_norm": float(norm),
            "compression": sum(r["compression"] for r in diagnostics) / len(diagnostics) if diagnostics else None,
            "hard_ratios": [sum(r["hard_ratios"][i] for r in diagnostics) / len(diagnostics)
                            for i in range(3)] if diagnostics else None,
            "soft_ratios": [sum(r["soft_ratios"][i] for r in diagnostics) / len(diagnostics)
                            for i in range(3)] if diagnostics else None}


def adamw(model, device, config):
    decay, no_decay = [], []
    for param in model.parameters():
        target = no_decay if param.ndim < 2 or getattr(param, "_no_weight_decay", False) else decay
        target.append(param)
    groups = [{"params": decay, "weight_decay": .05}, {"params": no_decay, "weight_decay": 0}]
    fused = device.type == "cuda" and not config.get("disable_fused_adamw", False)
    try:
        return torch.optim.AdamW(groups, lr=config.get("lr", 3e-4), betas=(.9, .95), eps=1e-8, fused=fused)
    except (RuntimeError, TypeError) as error:
        if not fused:
            raise
        print(f"Fused AdamW unavailable: {error}. Using unfused AdamW.", flush=True)
        return torch.optim.AdamW(groups, lr=config.get("lr", 3e-4), betas=(.9, .95), eps=1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--schedule-bpe", type=int)
    parser.add_argument("--probe-every", type=int, default=200)
    parser.add_argument("--resume")
    parser.add_argument("--debug-corpus", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text()) | {"runtime_seed": args.seed}
    device = torch.device(args.device)
    hours = args.max_hours if args.max_hours is not None else (72. if args.full else 6.)
    if not 0 < hours <= 72 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Training limits must be positive and max-hours cannot exceed 72")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + 1)
    cache = PairCache(args.data)
    minimum = 1_000_000_000 if args.full else 30_000_000
    if args.full and args.debug_corpus:
        parser.error("Full runs require a fully audited corpus")
    if not (args.debug_corpus or args.dry_run) and (
        not cache.meta["approved_for_research"] or cache.meta["target_bpe"] < minimum
    ):
        raise SystemExit("Audited corpus quota or quality gate failed; diagnostic runs require --debug-corpus.")
    if sha256_file(args.data / "train.jsonl") != cache.meta["split_sha256"]["train"]:
        raise SystemExit("Training split checksum mismatch")
    horizon = args.schedule_bpe if args.schedule_bpe is not None else minimum
    if horizon < 1:
        parser.error("--schedule-bpe must be positive")
    sampler = CurriculumSampler(cache, args.seed, horizon)
    codec = Codec(cache.tokenizer if config["model"] == "A" else None)
    model = build_model(config, device)
    total = count_parameters(model)
    if total > 40_000_000:
        raise SystemExit(f"{total:,} parameters exceeds budget")
    print(f"Model {config['model']} — {total:,} trainable parameters; {device}", flush=True)
    optimizer = adamw(model, device, config)
    if args.dry_run:
        input_rows = [sampler.next()]
        with amp(device):
            ids, labels, _ = lm_batch(codec, input_rows, device)
            loss = nll(model(ids), labels)
        loss.backward()
        print(json.dumps({"dry_run": True, "loss": float(loss.detach())}))
        return
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    root = args.run
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".train.lock"
    import fcntl
    lock_handle = lock.open("a")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log = root / "metrics.jsonl"
    if log.exists() and not args.resume:
        raise SystemExit("This run already has metrics. Use --resume latest or a fresh path.")
    identity = {"config": config, "seed": args.seed, "train_sha": cache.meta["split_sha256"]["train"],
                "tokenizer_sha": cache.meta["tokenizer_sha256"], "full": args.full, "debug": args.debug_corpus}
    if args.resume:
        state = load_state(args.resume, root, model, optimizer, sampler, identity, device, rng)
    else:
        state = {"step": 0, "elapsed": 0., "micro": config.get("micro_pairs", 2), "estimated_flops": 0}
        (root / "run.json").write_text(json.dumps(identity | {"horizon": horizon, "parameters": total}, indent=2), encoding="utf-8")
    started = time.monotonic()
    prior_elapsed = state["elapsed"]
    val = PairCache(args.data, "val")
    pairs = config.get("micro_pairs", 2) * config.get("grad_accum", 8)
    model.train()
    monitor_rows = select_probe(cache) if hasattr(model, "routers") and args.probe_every > 0 else []
    json_line(log, {"kind": "status", "step": state["step"], "status": "running", "resumed": bool(args.resume),
                    "diagnostic": args.debug_corpus, "at": datetime.now(timezone.utc).isoformat()})
    try:
        if not args.resume and hasattr(model, "routers"):
            json_line(log, {"kind": "calibration", "step": 0, "thresholds": calibrate_router(model, codec, cache, device)})
        last_save, last_val = time.monotonic(), time.monotonic()
        json_line(log, {"kind": "validation", "step": state["step"], "bpb": validate(model, codec, val, device)})
        while not STOP and time.monotonic() - started + prior_elapsed < hours * 3600:
            if args.max_steps is not None and state["step"] >= args.max_steps:
                break
            progress = sampler.progress
            before = sampler.exposure
            batch_started = time.monotonic()
            rows = [sampler.next() for _ in range(pairs)]
            iterations = rng.randrange(1, 9) if config["model"] == "C" and progress >= .20 else 1
            rng_before = rng_state(device, rng)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                report = update(model, optimizer, codec, rows, device, rng, iterations,
                                progress, state["micro"], config, state["step"] + 1)
            except torch.cuda.OutOfMemoryError:
                if state["micro"] <= 1:
                    raise
                optimizer.zero_grad(set_to_none=True)
                state["micro"] = max(1, state["micro"] // 2)
                json_line(log, {"kind": "retry", "step": state["step"], "micro_pairs": state["micro"], "reason": "CUDA OOM"})
                retry = True
            else:
                retry = False
            if retry:
                torch.cuda.empty_cache()
                restore_rng(rng_before, device, rng)
                report = update(model, optimizer, codec, rows, device, rng, iterations,
                                progress, state["micro"], config, state["step"] + 1)
            optimizer.step()
            state["step"] += 1
            state["estimated_flops"] = state.get("estimated_flops", 0) + report["estimated_training_flops"]
            now = time.monotonic()
            state["elapsed"] = now - started + prior_elapsed
            seconds = now - batch_started
            bytes_seen = sum(row["bytes_en"] + row["bytes_ru"] for row in rows)
            json_line(log, {"kind": "train", "step": state["step"], "elapsed": state["elapsed"], "progress": progress,
                            "bpe": sampler.exposure, "source_bytes_step": bytes_seen, "source_bytes_per_s": bytes_seen / seconds,
                            "step_s": seconds, "K": iterations, "gpu_gib": gpu_memory(device),
                            "micro_pairs": state["micro"], "estimated_flops_total": state["estimated_flops"],
                            "diagnostic": args.debug_corpus, "domain_bpe": sampler.domain_exposure,
                            "domain_reshuffles": sampler.epochs, **report})
            if monitor_rows and state["step"] % args.probe_every == 0:
                try:
                    result = probe(model, codec, monitor_rows, device, progress, iterations, root, state["step"])
                except torch.cuda.OutOfMemoryError:
                    result = {"kind": "probe_skipped", "step": state["step"], "reason": "CUDA OOM"}
                if result:
                    json_line(log, result)
            reached = any(before < point <= sampler.exposure for point in MILESTONES)
            if reached or now - last_save >= 300 or state["step"] % 500 == 0:
                path = checkpoint(root, state, model, optimizer, sampler, identity, config, args.data, device, rng, reached)
                json_line(log, {"kind": "checkpoint", "step": state["step"], "path": path})
                last_save = time.monotonic()
            if now - last_val >= 1800 or state["step"] % 500 == 0:
                json_line(log, {"kind": "validation", "step": state["step"], "bpb": validate(model, codec, val, device)})
                last_val = time.monotonic()
        state["elapsed"] = time.monotonic() - started + prior_elapsed
        path = checkpoint(root, state, model, optimizer, sampler, identity, config, args.data, device, rng)
        json_line(log, {"kind": "validation", "step": state["step"], "bpb": validate(model, codec, val, device)})
        json_line(log, {"kind": "status", "status": "interrupted" if STOP else "stopped_at_limit",
                        "step": state["step"], "checkpoint": path, "bpe": sampler.exposure,
                        "elapsed": time.monotonic() - started + prior_elapsed})
    except Exception as error:
        json_line(log, {"kind": "status", "status": "error", "step": state["step"], "error": str(error)})
        raise
    finally:
        cache.close()
        val.close()
        lock_handle.close()


if __name__ == "__main__":
    main()
