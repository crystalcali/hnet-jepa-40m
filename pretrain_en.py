"""Run the matched English pretraining experiment and score its checkpoints."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import random
import signal
import time

from safetensors.torch import load_file
from tokenizers import Tokenizer
import torch
from torch.nn import functional as F
import yaml

from data.pretrain_english import Windows
from hnet.compute import ComputeEstimate
from hnet.corpus import sha256_file
from hnet.data import pad
from hnet.model import count_parameters
from hnet.pretrain import build_pretrain_model, ratio_loss
from hnet.training import amp, lr_at
import train as runtime


class WindowSampler:
    def __init__(self, windows, batch_size, epochs, seed):
        self.windows = windows
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.epoch = self.cursor = 0
        self.shuffle()

    def shuffle(self):
        self.order = list(range(len(self.windows)))
        random.Random(self.seed + self.epoch).shuffle(self.order)

    def __iter__(self):
        while self.epoch < self.epochs:
            while self.cursor < len(self.order):
                batch = self.order[self.cursor:self.cursor + self.batch_size]
                self.cursor += len(batch)
                yield self.windows.read(batch)
            self.epoch += 1
            self.cursor = 0
            if self.epoch < self.epochs:
                self.shuffle()

    def snapshot(self):
        return {"epoch": self.epoch, "cursor": self.cursor}

    def restore(self, snapshot):
        self.epoch = snapshot["epoch"]
        self.cursor = snapshot["cursor"]
        self.shuffle()


class ByteBPEBatches:
    def __init__(self, tokenizer, kind, device):
        self.tokenizer, self.kind, self.device = tokenizer, kind, device
        vocab = tokenizer.get_vocab_size() if kind == "bpe" else 256
        self.bos, self.eos, self.pad = vocab, vocab + 1, vocab + 2

    def __call__(self, raw):
        bpe = self.tokenizer.encode_batch([x.decode("utf-8") for x in raw], add_special_tokens=False)
        tokens = (row.ids for row in bpe) if self.kind == "bpe" else map(list, raw)
        seq = pad([[self.bos, *x, self.eos] for x in tokens], self.pad, self.device)
        ids, labels = seq[:, :-1].contiguous(), seq[:, 1:].clone()
        labels[labels.eq(self.pad)] = -100
        valid = labels.ne(-100)
        return ids.masked_fill(~valid, self.pad), valid, labels, sum(len(x.ids) for x in bpe)


def get_corpus(data, split):
    info = json.loads((data / "pretrain_manifest.json").read_text())
    path = data / f"{split}.jsonl"
    tokenizer_path = data / "tokenizer.json"
    if sha256_file(path) != info["split_sha256"][split]:
        raise ValueError(f"Corpus checksum mismatch: {path}")
    if sha256_file(tokenizer_path) != info["tokenizer_sha256"]:
        raise ValueError("Tokenizer checksum mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() + 3 != info["vocab"] or info["bytes_per_bpe"] <= 1:
        raise ValueError("Invalid measured BPE vocabulary or compression")
    return info, tokenizer, Windows(path, info["context_bytes"])


@torch.no_grad()
def evaluate(model, batches, windows, device, number, batch_size):
    was_training = model.training
    model.eval()
    try:
        count = min(len(windows), number) if number else len(windows)
        # Fixed spread across the split instead of evaluating only the beginning.
        indices = [i * len(windows) // count for i in range(count)]
        nll = byte_count = positions = chunks = 0
        for start in range(0, count, batch_size):
            raw = windows.read(indices[start:start + batch_size])
            ids, valid, labels, _ = batches(raw)
            labels[labels.eq(batches.eos)] = -100
            with amp(device):
                result = model(ids, valid)
            nll += F.cross_entropy(result.logits.float().flatten(0, 1), labels.flatten(),
                                   reduction="sum").item()
            byte_count += sum(map(len, raw))
            if result.route is not None:
                positions += int(valid.sum())
                chunks += int(result.route.boundaries.sum())
        return {"bpb": {"en": nll / (math.log(2) * byte_count)},
                "bytes": byte_count, "windows": count,
                "compression": positions / chunks if chunks else None}
    finally:
        model.train(was_training)


@torch.no_grad()
def calibrate(model, batches, windows, goal, device, seed):
    indices = random.Random(seed).sample(range(len(windows)), min(24, len(windows)))
    values = []
    for offset in range(0, len(indices), 4):
        ids, valid, _, _ = batches(windows.read(indices[offset:offset+4]))
        with amp(device):
            p = model.router.probabilities(model.encode(ids))
        values.append(p[:, 1:][valid[:, 1:]].cpu())
    threshold = torch.cat(values).quantile(1 - 1 / goal).clamp(.0001, .9999)
    model.router.threshold.copy_(threshold.to(device))
    return float(threshold)


def train(args):
    device = torch.device(args.device)
    config = yaml.safe_load(args.config.read_text())
    info, tokenizer, windows = get_corpus(args.data, "train")
    _, _, val = get_corpus(args.data, "val")
    if (config["bpe_vocab"], config["context_bytes"]) != (info["vocab"], info["context_bytes"]):
        raise ValueError("The corpus BPE vocabulary and window length must match the config")
    if not 0 < config["batch_size"] or config["ratio_weight"] < 0:
        raise ValueError("Expected positive batch size and a nonnegative ratio weight")
    goal = info["bytes_per_bpe"]
    if args.model == "hnet-fixed":
        goal = args.fixed_ratio if args.fixed_ratio is not None else goal
        config = {**config, "fixed_ratio": goal}
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sampler = WindowSampler(windows, config["batch_size"], args.epochs, args.seed)
    identity = {"config": config, "kind": args.model, "seed": args.seed, "epochs": args.epochs, "corpus": info}
    config = {**config, "kind": args.model}
    batches = ByteBPEBatches(tokenizer, args.model, device)
    model = build_pretrain_model(args.model, config, info["vocab"], device).train()
    print(f"{args.model}: {count_parameters(model):,} parameters; "
          f"measured BPE ratio {info['bytes_per_bpe']:.4f}, experiment target {goal:.4f}", flush=True)
    optimizer = runtime.adamw(model, device, config)
    root = args.run
    root.mkdir(parents=True, exist_ok=True)
    logfile = root / "metrics.jsonl"
    with (root / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if logfile.exists() and not args.resume:
            raise ValueError("Run already exists; use --resume or a fresh --run.")
        if args.resume:
            state = runtime.load_state("latest", root, model, optimizer, sampler, identity, device, rng)
        else:
            state = {"step": 0, "elapsed": 0., "bytes": 0, "bpe": 0, "estimated_flops": 0}
        run_loop(args, state, sampler, model, optimizer, identity, config, windows, val,
                 batches, goal, device, rng, logfile)


def run_loop(args, state, sampler, model, optimizer, identity, config, windows, val,
             batches, goal, device, rng, logfile):
    runtime.STOP = False
    signal.signal(signal.SIGINT, runtime.interrupt)
    signal.signal(signal.SIGTERM, runtime.interrupt)
    started = last_save = time.monotonic()
    previous_seconds = state["elapsed"]
    planned = sampler.epochs * sum(b - a for _, a, b in windows.index)

    def log(event, **fields):
        runtime.json_line(logfile, {"kind": event, "step": state["step"], **fields})

    def validate():
        log("validation", **evaluate(model, batches, val, device, args.val_windows, config["batch_size"]),
            source_bytes_total=state["bytes"], estimated_flops_total=state["estimated_flops"])

    def save():
        state["elapsed"] = previous_seconds + time.monotonic() - started
        path = runtime.checkpoint(args.run, state, model, optimizer, sampler, identity,
                                  config, args.data, device, rng)
        log("checkpoint", path=path, bpe=state["bpe"], source_bytes_total=state["bytes"])
        return path

    log("status", status="running", resumed=args.resume, model=config["kind"],
        at=datetime.now(timezone.utc).isoformat())
    try:
        if not args.resume and config["kind"] == "hnet":
            log("calibration", threshold=calibrate(model, batches, windows, goal, device, args.seed),
                compression_target=goal)
        if not args.resume:
            validate()
        iterator = iter(sampler)
        while not runtime.STOP and (not args.max_steps or state["step"] < args.max_steps):
            if previous_seconds + time.monotonic() - started >= args.max_hours * 3600:
                break
            tick = time.monotonic()
            raw = next(iterator, None)
            if raw is None:
                break
            ids, valid, target, bpe_count = batches(raw)
            size = sum(map(len, raw))
            progress = state["bytes"] / planned
            lr = lr_at(min(progress, 1.), config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            with ComputeEstimate(model) as estimate, amp(device):
                result = model(ids, valid)
                nll = F.cross_entropy(result.logits.float().flatten(0, 1), target.flatten())
                loss = nll
                terms = {"byte" if result.route is not None else "bpe": nll.detach()}
                if config["kind"] == "hnet":
                    ratio = ratio_loss(result.route, goal)
                    loss = loss + config["ratio_weight"] * ratio
                    terms["ratio"] = ratio.detach()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite pretraining loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            seconds = time.monotonic() - tick
            state["step"] += 1
            state["bytes"] += size
            state["bpe"] += bpe_count
            state["estimated_flops"] += 3 * estimate.flops
            route = result.route
            hard = float(route.valid.sum() / route.boundaries.sum()) if route is not None else None
            losses = {name: float(value) for name, value in terms.items()}
            weighted = {name: value * (config["ratio_weight"] if name == "ratio" else 1.)
                        for name, value in losses.items()}
            log("train", loss=float(loss.detach()), raw=losses, weighted=weighted, coverage={},
                bpe=state["bpe"], source_bytes_total=state["bytes"], source_bytes_step=size,
                source_bytes_per_s=size / seconds, step_s=seconds, K=1, lr=lr, grad_norm=float(grad),
                gpu_gib=runtime.gpu_memory(device), compression=hard,
                hard_ratios=[hard] if hard else None, compression_target=goal if hard else None,
                epoch=sampler.epoch + 1, estimated_training_flops=3 * estimate.flops,
                estimated_flops_total=state["estimated_flops"])
            if state["step"] % args.val_every == 0:
                validate()
            if state["step"] % args.val_every == 0 or time.monotonic() - last_save >= 300:
                save()
                last_save = time.monotonic()
        path = save()
        validate()
        log("status", status="interrupted" if runtime.STOP else "stopped_at_limit",
            checkpoint=path, bpe=state["bpe"], source_bytes_total=state["bytes"],
            elapsed=previous_seconds + time.monotonic() - started)
    except Exception as error:
        log("status", status="error", error=str(error))
        raise


def score(args):
    device = torch.device(args.device)
    info, tokenizer, windows = get_corpus(args.data, args.split)
    config = json.loads((args.checkpoint / "config.json").read_text())
    if (config["context_bytes"] != info["context_bytes"]
            or sha256_file(args.checkpoint / "tokenizer.json") != info["tokenizer_sha256"]):
        raise ValueError("Evaluation corpus does not match the checkpoint tokenizer or context")
    kind = config["kind"]
    model = build_pretrain_model(kind, config, info["vocab"], device)
    model.load_state_dict(load_file(str(args.checkpoint / "model.safetensors")))
    result = evaluate(model, ByteBPEBatches(tokenizer, kind, device), windows,
                      device, args.windows, config["batch_size"])
    print(json.dumps({"model": kind, "split": args.split, **result}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--model", choices=("bpe", "hnet", "hnet-fixed"), required=True)
    p.add_argument("--fixed-ratio", type=float,
                   help="Fixed H-Net target; defaults to the corpus BPE ratio")
    p.add_argument("--config", type=Path, default=Path("configs/pretrain_en.yaml"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-hours", type=float, default=24)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-windows", type=int, default=128)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    e = commands.add_parser("eval")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--split", choices=("val", "test"), default="test")
    e.add_argument("--windows", type=int, default=0)
    e.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.command == "train":
        if args.fixed_ratio is not None and args.model != "hnet-fixed":
            parser.error("--fixed-ratio requires --model hnet-fixed")
        if not (args.epochs >= 1 and args.val_every >= 1 and args.val_windows >= 0
                and args.max_hours > 0 and (args.max_steps is None or args.max_steps >= 1)):
            parser.error("Specify positive training limits and a nonnegative validation window count")
        train(args)
    else:
        if args.windows < 0:
            parser.error("--windows must be nonnegative")
        score(args)


if __name__ == "__main__":
    main()
