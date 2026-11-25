"""Shared forward paths for training, validation and diagnostics."""
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import conditional_batch, lm_batch, pad
from .losses import ablation_loss, corrupt_bytes, cross_loss, dynamics_loss, interpolate, routing_loss, sigreg


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def nll(logits, labels, reduction="mean"):
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction=reduction)


def rollout(model, ids, iterations, rng=None):
    encoded = model.encode(ids)
    states = model.reason(encoded.latents[-1], iterations, trajectory=True)
    auxiliary = sorted(rng.sample(range(iterations - 1), min(2, iterations - 1))) if rng else []
    return encoded, states, auxiliary


def conditional(model, codec, samples, iterations, device, rng=None):
    prompt, prefix, labels = conditional_batch(codec, samples, device)
    if prompt is None:
        return nll(model(prefix), labels), None
    encoded = model.encode(prompt, terminal=True)
    target = model.encode(prefix)
    states = model.reason(encoded.latents[-1], iterations, trajectory=True)
    loss = nll(model.condition_decode(target, states[-1], encoded.valid), labels)
    if iterations > 1 and rng:
        indices = rng.sample(range(iterations - 1), min(2, iterations - 1))
        loss = loss + .25 * torch.stack([
            nll(model.condition_decode(target, states[k], encoded.valid), labels) for k in indices
        ]).mean()
    return loss, (states, encoded.valid)


def fits(a, b, task=False):
    return bool(a and b) and len(a.encode()) + len(b.encode()) + (6 if task else 5) <= 8192


def eligible_tasks(rows, rng):
    translations, tasks, dynamics = [], [], []
    for row in rows:
        lang = rng.choice(("en", "ru"))
        other = "en" if lang == "ru" else "ru"
        a, b = row[lang]["text"], row[other]["text"]
        if fits(a, b):
            translations.append((a, b, lang, other, False))
        for side in ("en", "ru"):
            record = row[side]
            question, answer = record.get("question"), record.get("answer")
            if fits(question, answer, True):
                tasks.append((question, answer, side, side, True))
            steps = record.get("steps") or []
            if question and answer and steps:
                segments = [question, *steps, answer]
                complete = ["\n".join(segments[:i]) for i in range(2, len(segments) + 1)]
                if all(len(item.encode()) + 3 <= 8192 for item in complete):
                    dynamics.append((question, complete, side, answer))
            elif len(blocks := record.get("blocks") or []) > 1 and all(
                len(item.encode()) + 3 <= 8192 for item in blocks
            ):
                prompt, *stages = blocks
                if fits(prompt, stages[0]):
                    dynamics.append((prompt, stages[:8], side, None))
    return translations, tasks, dynamics


def dynamics_objective(model, codec, items, K, device, task_cache):
    losses, continuations, h_samples = [], [], []
    for prompt, stages, lang, answer in items:
        selected = sorted({(iteration * len(stages) + 7) // 8 - 1 for iteration in range(1, K + 1)})
        # Repeat selected outputs by using the original index equation for all K.
        effective = encode_stages(model, codec, [stages[j] for j in selected], lang, device)
        chosen = {step: idx for idx, step in enumerate(selected)}
        key = (prompt, answer, lang, lang, True)
        if key in task_cache:
            trajectory, mask = task_cache[key]
        else:
            prompt_state = model.encode(pad([codec.ids(prompt, lang, task=bool(answer))], codec.pad, device),
                                        terminal=True)
            trajectory = model.reason(prompt_state.latents[-1], K, trajectory=True)
            mask = prompt_state.valid
        # dynamics_loss maps K iterations onto at least eight slots. Instead use
        # the original stage count with repeated selections expanded in place.
        expand = [chosen.get(j, 0) for j in range(len(stages))]
        z, zmask = effective
        losses.append(dynamics_loss(model, trajectory, mask, z[expand], zmask[expand]))
        h_samples.append((lang, trajectory[-1][mask]))
        if answer is None:
            # Natural-block continuation uses the corresponding causal prompt memory.
            _, prefix, labels = conditional_batch(codec, [(prompt, stages[0], lang, lang, False)], device)
            target = model.encode(prefix)
            continuations.append(nll(model.condition_decode(target, trajectory[-1], mask), labels))
    return torch.stack(losses).mean(), (
        torch.stack(continuations).mean() if continuations else None
    ), h_samples


def encode_stages(model, codec, stages, lang, device):
    states = model.encode(pad([codec.ids(text, lang) for text in stages], codec.pad, device))
    valid = states.content[-1]
    valid = torch.where(valid.any(1, keepdim=True), valid, states.valid)
    return states.latents[-1], valid


def batch_objectives(model, codec, rows, K, weights, rng, device, progress, lm_scale):
    ids, labels, texts = lm_batch(codec, rows, device)
    is_hnet = hasattr(model, "routers")
    terms, coverage, debug = {}, {}, {}
    if is_hnet:
        clean, states, aux_indices = rollout(model, ids, K, rng)
        terms["byte"] = nll(model.decode(states[-1], clean), labels)
        if aux_indices:
            terms["aux"] = torch.stack([nll(model.decode(states[i], clean), labels) for i in aux_indices]).mean()
        terms["ratio"], terms["anti"], debug = routing_loss(clean, model.config.get("ratio_goal", 36))
    else:
        terms["byte"] = nll(model(ids), labels)
    coverage["byte_targets"] = int(labels.ne(-100).sum())
    coverage["base_positions"] = int(ids.ne(codec.pad).sum())

    if is_hnet and model.recurrent and (weights["global"] or weights["local"]):
        cross, cross_coverage = cross_loss(model, clean, rows, bool(weights["local"]))
        terms.update(cross)
        coverage.update(cross_coverage)
    if is_hnet and model.recurrent and weights["ablation"] and rng.random() < .5:
        max_rate = interpolate(progress, (.0, .05, .15, .30, .50, .75, 1),
                               (.1, .1, .2, .35, .5, .65, .8))
        if max_rate > .65 and rng.random() >= .1:
            max_rate = .65
        fraction = rng.uniform(.1, max_rate)
        sides = [row[lang] for row in rows for lang in ("en", "ru")]
        kinds = ["words", "span", "latent"]
        if all(side.get("step_byte_spans") for side in sides):
            kinds.append("steps")
        kind = rng.choice(kinds)
        latent_mask = None
        if kind == "latent":
            latent_mask = (rng.randrange(3), rng.uniform(0, 1 - fraction), fraction)
            corrupted = model.encode(ids, latent_mask=latent_mask)
        else:
            corrupted = model.encode(corrupt_bytes(ids, texts, fraction, rng, kind, sides),
                                     input_content=ids.lt(256))
        loss = ablation_loss(model, clean, corrupted, latent_mask)
        if loss is not None:
            terms["ablation"] = loss
            coverage["ablation"] = len(sides)
    translations, tasks, dynamics = eligible_tasks(rows, rng)
    coverage.update(translation_eligible=len(translations), task_eligible=len(tasks), dynamics_eligible=len(dynamics))
    if weights["translation"] and translations and rng.random() < .5:
        terms["translation"], _ = conditional(model, codec, translations, K, device, rng)
        coverage["translation"] = len(translations)
    task_cache = {}
    if weights["task"] and tasks and rng.random() < .5:
        terms["task"], _ = conditional(model, codec, tasks, K, device, rng)
        coverage["task"] = len(tasks)
    if is_hnet and model.recurrent and weights["dynamics"] and dynamics and rng.random() < .25:
        terms["dynamics"], language_continuation, _ = dynamics_objective(
            model, codec, dynamics, K, device, task_cache
        )
        coverage["dynamics"] = len(dynamics)
        if language_continuation is not None:
            terms["continuation"] = language_continuation
    probabilities = {"ablation": .5, "dynamics": .25, "translation": .5, "task": .5, "continuation": .25}
    multipliers = {**weights, "byte": lm_scale, "aux": .25 * lm_scale, "continuation": 1.}
    weighted = {name: item * (multipliers[name] / probabilities.get(name, 1.))
                for name, item in terms.items()}
    return sum(weighted.values()), terms, weighted, coverage, debug


def sigreg_step(model, micro_inputs, step, device, minimum=256):
    """Replay the clean encodes with checkpointing, then jointly regularize the sketches."""
    buffers = {level: [[], []] for level in (0, 1)}

    def encode_for_sigreg(ids):
        with amp(device):
            state = model.encode(ids)
            output = []
            for level in (1, 2):
                for lang in (0, 1):
                    x = state.latents[level][lang::2]
                    valid = state.content[level][lang::2]
                    output.append(x[valid][:1024].float())
            return tuple(output)

    for ids in micro_inputs:
        z2_en, z2_ru, z3_en, z3_ru = checkpoint(encode_for_sigreg, ids, use_reentrant=False)
        for level, en, ru in ((0, z2_en, z2_ru), (1, z3_en, z3_ru)):
            buffers[level][0].append(en)
            buffers[level][1].append(ru)
    terms, counts = [], {}
    for level, languages in buffers.items():
        en, ru = [torch.cat(lang) for lang in languages]
        count = min(len(ru), len(en), 1024)
        counts[f"L{level+2}"] = {"en": len(en), "ru": len(ru), "used": 2 * count if 2 * count >= minimum else 0}
        if 2 * count >= minimum:
            terms.append(sigreg(torch.cat((en[:count], ru[:count])), step))
    return (torch.stack(terms).mean() if terms else None), counts


def lr_at(progress, config):
    peak, end = config.get("lr", 3e-4), config.get("min_lr", 3e-5)
    if progress < .02:
        return peak * max(.01, progress / .02)
    t = (progress - .02) / .98
    return end + (peak - end) * (1 + math.cos(math.pi * t)) * .5
