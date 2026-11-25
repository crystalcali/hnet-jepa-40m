"""Bounded diagnostics on a small, fixed real-data probe."""
import random

import numpy as np
import torch
from torch.nn import functional as F

from .data import lm_batch
from .losses import loss_weights
from .training import amp, batch_objectives


def select_probe(cache):
    found = []
    for row in cache.records():
        if row["bytes_en"] <= 384 and row["bytes_ru"] <= 384:
            found.append(row)
            if len(found) == 2:
                break
    return found


def probe(model, codec, rows, device, progress, K, run, step):
    if not rows:
        return None
    trained = model.training
    model.eval()
    try:
        with amp(device):
            _, raw, _, coverage, _ = batch_objectives(
                model, codec, rows, K, loss_weights(progress), random.Random(846), device, progress, 1.
            )
        common = {f"router{i+1}.query": layer.query.weight for i, layer in enumerate(model.routers)}
        common["core.last.q"] = model.blocks[-1].attention.q.weight
        sketches = {}
        for index, (name, value) in enumerate(raw.items()):
            grads = torch.autograd.grad(value, tuple(common.values()), allow_unused=True,
                                        retain_graph=index + 1 < len(raw))
            sketches[name] = torch.cat([
                torch.zeros_like(param).flatten() if grad is None else grad.detach().float().flatten()
                for param, grad in zip(common.values(), grads)
            ])
        with torch.no_grad(), amp(device):
            ids, _, _ = lm_batch(codec, rows, device)
            encoded = model.encode(ids)
            trajectory = model.reason(encoded.latents[-1], K, trajectory=True)
            data = {"ids": ids.cpu().numpy(), "pair_ids": np.asarray([row["pair_id"] for row in rows])}
            for i, route in enumerate(encoded.routes, 1):
                data[f"bounds_{i}"] = route.boundaries.cpu().numpy()
                data[f"input_valid_{i}"] = route.valid.cpu().numpy()
                data[f"parents_{i}"] = route.parents.cpu().numpy()
                data[f"Z{i}"] = encoded.latents[i-1].float().cpu().numpy()
                data[f"byte_offsets_{i}"] = encoded.offsets[i-1].cpu().numpy()
            states = [encoded.latents[-1], *trajectory]
            for i, latent in enumerate(states):
                data[f"h{i}"] = latent.float().cpu().numpy()
            drift = [float((b.float() - a.float())[encoded.valid].square().mean().sqrt())
                     for a, b in zip(states, states[1:])]
        root = run / "probes"
        root.mkdir(exist_ok=True)
        path = root / f"step-{step:09d}.npz"
        np.savez_compressed(path, **data)
        for expired in sorted(root.glob("step-*.npz"))[:-8]:
            expired.unlink()
        return {"kind": "probe", "step": step,
                "subset": list(common), "coverage": coverage, "snapshot": str(path.relative_to(run)),
                "n_pair": len(rows), "state_rms_delta": drift,
                "norms": {name: float(grad.norm()) for name, grad in sketches.items()},
                "cosine": {a: {b: float(F.cosine_similarity(x, y, dim=0))
                               for b, y in sketches.items()} for a, x in sketches.items()}}
    finally:
        model.train(trained)
