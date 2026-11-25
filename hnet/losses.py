"""Training objectives. Targets remain in the autograd graph."""
import math
import re

import torch
from torch.nn import functional as F

from .model import MASK, masked_mean


SCHEDULE = {
    "byte": (1, 1, 1, 1, 1, 1),
    "ratio": (.1, .1, .1, .1, .1, .1),
    "anti": (.05, .05, .05, .05, .05, .05),
    "global": (0, .1, .2, .2, .2, .2),
    "local": (0, .05, .10, .20, .20, .20),
    "sigreg": (0, .10, .10, .15, .15, .15),
    "ablation": (0, .05, .15, .25, .25, .25),
    "translation": (0, 0, .20, .50, .50, .50),
    "dynamics": (0, 0, 0, .15, .25, .25),
    "task": (0, 0, 0, .20, .50, .50),
}


def interpolate(progress, knots, values):
    for left, right, start, end in zip(knots, knots[1:], values, values[1:]):
        if progress < right:
            return start + (end - start) * (progress - left) / (right - left)
    return values[-1]


def loss_weights(progress):
    return {name: interpolate(progress, (0, .05, .15, .30, .50, 1), values)
            for name, values in SCHEDULE.items()}


def routing_loss(encoded, goal=36):
    soft, hard, ste = [], [], []
    for route in encoded.routes:
        p = route.probs.float()
        valid = route.valid.float()
        hard_count = route.boundaries.float().sum()
        soft_count = torch.where(route.forced, 1., p).mul(valid).sum()
        surrogate = torch.sigmoid((p - route.steepness) / .15)
        surrogate = surrogate.mul(route.valid & ~route.forced).sum()
        count = hard_count + (surrogate - surrogate.detach())
        soft.append(valid.sum() / soft_count.clamp_min(1))
        hard.append(valid.sum() / hard_count.clamp_min(1))
        ste.append((valid.sum() / count.clamp_min(.99)).log())
    levels = torch.stack(ste)
    ratio = (levels.sum() - math.log(goal)).square()
    anti = (F.relu(math.log(1.5) - levels).square() + F.relu(levels - math.log(8)).square()).mean()
    return ratio, anti, {"soft_ratios": torch.stack(soft).tolist(),
                         "hard_ratios": torch.stack(hard).tolist(),
                         "compression": float(torch.stack(hard).prod())}


def transport(cost, epsilon=.1, iterations=20):
    """Uniform log-Sinkhorn plan; gradients are passed through the cost."""
    with torch.no_grad():
        kernel = -cost.detach() / epsilon
        u = kernel.new_zeros(cost.shape[0])
        v = kernel.new_zeros(cost.shape[1])
        for _ in range(iterations):
            u = -math.log(cost.shape[0]) - torch.logsumexp(kernel + v[None, :], dim=1)
            v = -math.log(cost.shape[1]) - torch.logsumexp(kernel + u[:, None], dim=0)
        return torch.exp(kernel + u[:, None] + v[None, :])


def limit_points(x, limit=128):
    if len(x) > limit:
        index = torch.linspace(0, len(x) - 1, limit, device=x.device).long()
        return x[index]
    return x


def cross_loss(model, encoded, records, local_on):
    """Rows are ordered EN/RU per source pair."""
    names = ("global", "local")
    levels = {name: [] for name in names}
    coverage = {name: 0 for name in names}
    with torch.autocast(encoded.valid.device.type, enabled=False):
        for level, weight in ((1, .4), (2, .6)):
            z = encoded.latents[level].float()
            mask = encoded.content[level]
            offsets = encoded.offsets[level]
            predictor = model.cross_predictors[level - 1]
            values = {name: [] for name in names}
            for index, record in enumerate(records):
                spans = record.get("alignment") or []
                spans = [span for span in spans if span.get("verified")]
                blocks = spans or [None]
                pair_losses = {name: [] for name in names}
                for span in blocks:
                    chosen = []
                    for side, lang in enumerate(("en", "ru")):
                        row = 2 * index + side
                        valid = mask[row]
                        if span is not None:
                            start, stop = span[lang + "_span"]
                            valid = valid & offsets[row].ge(start) & offsets[row].lt(stop)
                        chosen.append(z[row, valid])
                    en, ru = chosen
                    if not len(en) or not len(ru):
                        continue
                    g_en, g_ru = en.mean(0), ru.mean(0)
                    pair_losses["global"].append(
                        (F.mse_loss(predictor(g_en), g_ru) + F.mse_loss(predictor(g_ru), g_en)) / 2
                    )
                    if local_on and span is not None:
                        en, ru = limit_points(en), limit_points(ru)
                        terms = []
                        for a, b in ((en, ru), (ru, en)):
                            cost = 1 - (F.normalize(predictor(a), dim=-1) @ F.normalize(b, dim=-1).T)
                            terms.append((transport(cost) * cost).sum())
                        pair_losses["local"].append(torch.stack(terms).mean())
                for name in names:
                    if pair_losses[name]:
                        values[name].append(torch.stack(pair_losses[name]).mean())
            for name in names:
                coverage[name] += len(values[name])
                if values[name]:
                    levels[name].append((weight, torch.stack(values[name]).mean()))
    return {
        name: sum(weight * item for weight, item in entries) / sum(weight for weight, _ in entries)
        for name, entries in levels.items() if entries
    }, coverage


def corrupt_bytes(ids, text, fraction, rng, kind, side):
    result = ids.clone()
    for row, value in enumerate(text):
        start_byte = 0
        spans = []
        for char in value:
            length = len(char.encode("utf-8"))
            spans.append((start_byte, start_byte + length))
            start_byte += length
        if not spans:
            continue
        target = max(1, math.ceil(start_byte * fraction))
        if kind == "words":
            choices = [(len(value[:m.start()].encode()), len(value[:m.end()].encode()))
                       for m in re.finditer(r"\S+", value)]
            rng.shuffle(choices)
            removed = 0
            for a, b in choices:
                result[row, 2 + a:2 + b] = MASK
                removed += b - a
                if removed >= target:
                    break
        elif kind == "steps" and side[row].get("step_byte_spans"):
            choices = list(side[row]["step_byte_spans"])
            rng.shuffle(choices)
            removed = 0
            for a, b in choices:
                result[row, 2 + a:2 + b] = MASK
                removed += b - a
                if removed >= target:
                    break
        else:
            start_index = rng.randrange(max(1, int(len(spans) * (1 - fraction))))
            a = spans[start_index][0]
            b = next((end for _, end in spans[start_index:] if end - a >= target), start_byte)
            result[row, 2 + a:2 + b] = MASK
    return result


def ablation_loss(model, clean, corrupted, latent_mask=None):
    terms = []
    with torch.autocast(clean.valid.device.type, enabled=False):
        for level in (1, 2):
            z = corrupted.latents[level]
            if latent_mask is not None and latent_mask[0] == 2 and level == 2:
                z = model.reason(z.float(), K=1)
            clean_mask, view_mask = clean.content[level], corrupted.content[level]
            valid = clean_mask.any(1) & view_mask.any(1)
            if valid.any():
                context = masked_mean(z, view_mask)[valid]
                target = masked_mean(clean.latents[level], clean_mask)[valid]
                predictor = model.abl_predictors[level - 1]
                terms.append(F.mse_loss(predictor(context), target))
            # Only the layer at which the mask was applied has guaranteed 1:1 slots.
            if latent_mask is not None and latent_mask[0] == level:
                start, fraction = latent_mask[1:]
                size = corrupted.routes[level].output_valid.sum(1, keepdim=True)
                lower = (start * size).long()
                position = torch.arange(z.shape[1], device=z.device)[None, :]
                missing = (position >= lower) & (position < lower + (fraction * size).ceil())
                missing = missing & clean_mask & view_mask
                if missing.any():
                    # L2 has no contextual block at its own positions after masking.
                    # Use its clean-aligned global objective and the downstream level there.
                    if level == 2:
                        terms.append(F.mse_loss(model.abl_predictors[1](z[missing].float()),
                                                clean.latents[2][missing].float()))
    return torch.stack(terms).mean() if terms else None


def dynamics_index(iteration, stages):
    if not 1 <= iteration <= 8 or stages < 1:
        raise ValueError("Dynamics supervision uses iterations 1–8 and at least one stage")
    return (iteration * stages + 7) // 8 - 1


def dynamics_loss(model, trajectory, prompt_valid, targets, target_valid):
    with torch.autocast(prompt_valid.device.type, enabled=False):
        target = masked_mean(targets, target_valid)
        indices = [dynamics_index(k, len(target)) for k in range(1, len(trajectory) + 1)]
        predictions = [model.dyn_predictor(masked_mean(h, prompt_valid)) for h in trajectory]
        return F.mse_loss(torch.cat(predictions), target[indices])


def sigreg(x, seed, slices=128, knots=17, maximum=3.):
    """Sliced Epps–Pulley Gaussian characteristic-function statistic (LeJEPA)."""
    with torch.autocast(x.device.type, enabled=False):
        g = torch.Generator(device=x.device).manual_seed(seed)
        directions = torch.randn(x.shape[1], slices, generator=g, device=x.device, dtype=torch.float32)
        projection = x.float() @ F.normalize(directions, dim=0)
        t = torch.linspace(0, maximum, knots, device=x.device)
        normal = torch.exp(-t.square() / 2)
        phase = projection[..., None] * t
        deviation = (phase.cos().mean(0) - normal).square() + phase.sin().mean(0).square()
        return 2 * len(x) * torch.trapezoid(deviation * normal, t, dim=-1).mean()
