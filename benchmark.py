"""Profile real-data training updates in a disposable in-memory model."""
import argparse
import json
from pathlib import Path
import random
import time

import torch
import yaml

from hnet.data import Codec, CurriculumSampler, PairCache
from hnet.model import build_model
from train import adamw, calibrate_router, update


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--full-objectives", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--trace", type=Path)
    a = p.parse_args()
    if a.steps < 1:
        p.error("--steps must be positive")
    torch.manual_seed(42)
    rng = random.Random(43)
    device = torch.device(a.device)
    config = yaml.safe_load(a.config.read_text())
    cache = PairCache(a.data)
    codec = Codec(cache.tokenizer if config["model"] == "A" else None)
    model = build_model(config, device).train()
    optimizer = adamw(model, device, config)
    if hasattr(model, "routers"):
        calibrate_router(model, codec, cache, device)
    sampler = CurriculumSampler(cache, 42, cache.meta["target_bpe"])
    if a.full_objectives:
        sampler.exposure = int(.70 * cache.meta["target_bpe"])
    progress = .70 if a.full_objectives else 0.
    micro = config.get("micro_pairs", 2)
    pairs = micro * config.get("grad_accum", 8)
    samples = []
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def step(i):
        rows = [sampler.next() for _ in range(pairs)]
        k = rng.randrange(1, 9) if a.full_objectives and config["model"] == "C" else 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        report = update(model, optimizer, codec, rows, device, rng, k, progress, micro, config, i)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.monotonic() - started
        size = sum(r["bytes_en"] + r["bytes_ru"] for r in rows)
        return {"seconds": seconds, "K": k, "bytes": size,
                "gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.,
                **report}

    step(0)
    for i in range(1, a.steps + 1):
        report = step(i)
        samples.append(report)
        print(json.dumps({"step": i, **report}), flush=True)
    if a.trace:
        with torch.profiler.profile(activities=activities, record_shapes=True, with_flops=True) as found:
            step(a.steps + 1)
        found.export_chrome_trace(str(a.trace))
    throughput = sum(s["bytes"] for s in samples) / sum(s["seconds"] for s in samples)
    coverage = {key: sum(s["coverage"].get(key, 0) for s in samples)
                for key in set().union(*(s["coverage"] for s in samples))}
    print(json.dumps({"source_bytes_per_s": throughput, "measured_steps": len(samples),
                      "peak_gpu_gib": max(s["gpu_gib"] for s in samples), "observed_coverage": coverage}))
    cache.close()


if __name__ == "__main__":
    main()
