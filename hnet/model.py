"""Causal byte hierarchy, shared recurrent core, and BPE control."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .mamba import MambaBlock

PAD, BOS, EOS, MASK, SEP, EN, RU, TASK = range(256, 264)
CONTROL = {"en": EN, "ru": RU}


def gather(x, index):
    return x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))


def masked_mean(x, valid):
    return (x.float() * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


@dataclass
class Route:
    probs: torch.Tensor
    boundaries: torch.Tensor
    forced: torch.Tensor
    valid: torch.Tensor
    anchors: torch.Tensor
    index: torch.Tensor
    parents: torch.Tensor
    output_valid: torch.Tensor
    assignment: torch.Tensor
    steepness: torch.Tensor


@dataclass
class Encoded:
    skips: list
    routes: list
    latents: list
    content: list
    offsets: list
    valid: torch.Tensor


class Router(nn.Module):
    def __init__(self, width, threshold=.5):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            for layer in (self.query, self.key):
                layer.weight.mul_(.02).add_(torch.eye(width))
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    def probabilities(self, x):
        with torch.autocast(x.device.type, enabled=False):
            left = self.query(x.float())
            right = self.key(x.float())
            similarities = F.cosine_similarity(left[:, 1:], right[:, :-1], dim=-1, eps=1e-6)
            return F.pad((1 - similarities) * .5, (1, 0), value=1)

    def forward(self, x, valid, terminal):
        p = self.probabilities(x)
        forced = terminal.clone()
        forced[:, 0] = True
        b = ((p >= self.threshold) | forced) & valid
        parents = b.long().cumsum(1) - 1
        lengths = b.sum(1)
        slots = torch.arange(int(lengths.max()), device=x.device)
        index = slots.new_zeros((x.shape[0], slots.numel()))
        batch, pos = b.nonzero(as_tuple=True)
        index[batch, parents[batch, pos]] = pos
        output_valid = slots[None, :] < lengths[:, None]
        assignment = torch.where(b, p, 1 - p)
        assignment = torch.where(forced | ~valid, 1., assignment)
        anchors = torch.where(forced, 1., p).gather(1, index)
        return Route(p, b, forced, valid, anchors, index, parents.clamp_min(0),
                     output_valid, assignment, self.threshold)


def dechunk(x, route):
    """Parallel inclusive scan of affine maps for probability-weighted expansion."""
    dtype = x.dtype
    v = x.float()
    a = 1 - route.anchors.float()
    a = torch.cat((torch.zeros_like(a[:, :1]), a[:, 1:]), dim=1)
    c = (1 - a[..., None]) * v
    a = a[..., None]
    stride = 1
    while stride < c.shape[1]:
        c = torch.cat((c[:, :stride], c[:, stride:] + a[:, stride:] * c[:, :-stride]), dim=1)
        a = torch.cat((a[:, :stride], a[:, stride:] * a[:, :-stride]), dim=1)
        stride *= 2
    return gather(c.to(dtype), route.parents)


def rope(x):
    length, dim = x.shape[-2:]
    inv = 10000 ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    theta = torch.arange(length, device=x.device, dtype=torch.float32)[:, None] * inv
    cosine, sine = theta.cos().to(x.dtype), theta.sin().to(x.dtype)
    left, right = x[..., ::2], x[..., 1::2]
    return torch.stack((left * cosine - right * sine, right * cosine + left * sine), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, width=512, heads=8, kv_heads=2):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim = width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, kv_heads * self.dim, bias=False)
        self.v = nn.Linear(width, kv_heads * self.dim, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x, memory=None, memory_valid=None):
        causal = memory is None
        memory = x if causal else memory

        def heads(projection, value, count):
            return projection(value).unflatten(-1, (count, self.dim)).transpose(1, 2)

        q = heads(self.q, x, self.heads)
        k = heads(self.k, memory, self.kv_heads)
        v = heads(self.v, memory, self.kv_heads)
        if causal:
            q, k = rope(q), rope(k)
        mask = memory_valid[:, None, None, :] if memory_valid is not None else None
        # Explicit KV expansion supports the same paths on CPU and Spark.
        repeats = self.heads // self.kv_heads
        y = F.scaled_dot_product_attention(
            q, k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1),
            attn_mask=mask, is_causal=causal,
        )
        return self.out(y.transpose(1, 2).flatten(-2))


class TransformerBlock(nn.Module):
    def __init__(self, ff, width=512):
        super().__init__()
        self.norm1 = nn.RMSNorm(width, eps=1e-5)
        self.norm2 = nn.RMSNorm(width, eps=1e-5)
        self.attention = Attention(width)
        self.gate = nn.Linear(width, ff, bias=False)
        self.up = nn.Linear(width, ff, bias=False)
        self.down = nn.Linear(ff, width, bias=False)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


def mamba_stack(width, depth, backend):
    return nn.Sequential(*(MambaBlock(width, backend) for _ in range(depth)))


class HNet(nn.Module):
    def __init__(self, config, backend="reference"):
        super().__init__()
        self.config = config
        self.recurrent = config["model"] == "C"
        widths = config.get("widths", [128, 256, 384, 512])
        depths = config.get("depths", [6, 4, 4])
        self.widths = widths
        self.embed = nn.Embedding(264, widths[0])
        self.encoders = nn.ModuleList(mamba_stack(w, n, backend) for w, n in zip(widths, depths))
        self.decoders = nn.ModuleList(mamba_stack(w, n, backend) for w, n in zip(widths, depths))
        self.routers = nn.ModuleList(Router(w) for w in widths[:-1])
        self.up = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(widths, widths[1:]))
        self.down = nn.ModuleList(nn.Linear(b, a, bias=False) for a, b in zip(widths, widths[1:]))
        self.blocks = nn.ModuleList(
            TransformerBlock(config["ff"], widths[-1]) for _ in range(config.get("blocks", 8))
        )
        self.core_norm = nn.RMSNorm(widths[-1], eps=1e-5)
        self.out_norm = nn.RMSNorm(widths[0], eps=1e-5)
        self.cond_norm = nn.RMSNorm(widths[-1], eps=1e-5)
        self.condition = Attention(widths[-1])
        self.skip_drop = 0.
        self.checkpoint_core = config.get("checkpoint_core", True)
        if self.recurrent:
            self.alpha = nn.Parameter(torch.zeros(()))
            self.cross_predictors = nn.ModuleList(
                nn.Sequential(nn.Linear(w, w, bias=False), nn.SiLU(), nn.Linear(w, w, bias=False))
                for w in widths[2:]
            )
            self.abl_predictors = nn.ModuleList(nn.Linear(w, w, bias=False) for w in widths[2:])
            self.dyn_predictor = nn.Linear(widths[-1], widths[-1], bias=False)

    def encode(self, ids, *, terminal=False, latent_mask=None, input_content=None):
        valid = ids.ne(PAD)
        if not valid[:, 0].all():
            raise ValueError("Sequences must be nonempty and right padded")
        offsets = (ids < 256).long().cumsum(dim=1) - (ids < 256).long()
        content = ids.lt(256) if input_content is None else input_content
        end = torch.zeros_like(valid)
        if terminal:
            end.scatter_(1, (valid.sum(1) - 1)[:, None], True)
        x = self.embed(ids)
        masks, offsets_by_level, skips, routes, latents = [], [], [], [], []
        for level, (encoder, router, up) in enumerate(zip(self.encoders, self.routers, self.up)):
            s = encoder(x)
            route = router(s, valid, end)
            index = route.index
            values = gather(s, index)
            gate = route.assignment.gather(1, index)
            values = values * (1 + (gate - gate.detach())).to(values.dtype)[..., None]
            x = up(values)
            valid = route.output_valid
            end = end.gather(1, index) & valid
            content = content.gather(1, index) & valid
            offsets = offsets.gather(1, index)
            if latent_mask is not None and latent_mask[0] == level:
                start, fraction = latent_mask[1:]
                size = valid.sum(1, keepdim=True)
                positions = torch.arange(x.shape[1], device=x.device)[None, :]
                lower = (start * size).long()
                upper = lower + (fraction * size).ceil().long()
                removed = valid & (positions >= lower) & (positions < upper)
                x = x.masked_fill(removed[..., None], 0)
            skips.append(s)
            routes.append(route)
            latents.append(x)
            masks.append(content)
            offsets_by_level.append(offsets)
        return Encoded(skips, routes, latents, masks, offsets_by_level, valid)

    def reason(self, z, K=1, trajectory=False):
        if not 1 <= K <= 32 or (not self.recurrent and K != 1):
            raise ValueError("K must be in [1,32], or 1 for control B")
        h, result = z, []
        for index in range(K):
            if self.recurrent:
                scale = self.config.get("iteration_scale", .1)
                freq = torch.exp(
                    -math.log(10000.) * torch.arange(0, z.shape[-1], 2, device=z.device) / z.shape[-1]
                )
                phase = index * freq
                embedding = torch.stack((phase.sin(), phase.cos() - 1), dim=-1).flatten()
                h = h + self.alpha.tanh() * z + scale * embedding.to(h.dtype)
            for block in self.blocks:
                if self.training and self.checkpoint_core and torch.is_grad_enabled():
                    h = checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)
            h = self.core_norm(h)
            if trajectory:
                result.append(h)
        return result if trajectory else h

    def decode(self, h, encoded, *, skip=True):
        for level in (2, 1, 0):
            route = encoded.routes[level]
            expanded = dechunk(self.down[level](h), route)
            gate = (1 + (route.assignment - route.assignment.detach())).to(expanded.dtype)
            s = encoded.skips[level] if skip else 0
            if skip and self.training and self.skip_drop:
                keep = torch.rand(s.shape[0], 1, 1, device=s.device) >= self.skip_drop
                s = s * keep.to(s.dtype) / (1 - self.skip_drop)
            h = self.decoders[level](expanded * gate[..., None] + s)
        logits = F.linear(self.out_norm(h), self.embed.weight)
        forbidden = torch.arange(264, device=logits.device).ge(256)
        forbidden[EOS] = False
        return logits.masked_fill(forbidden, -torch.inf)

    def condition_decode(self, target, memory, memory_valid):
        q = target.latents[-1]
        return self.decode(q + self.condition(self.cond_norm(q), memory, memory_valid), target)

    def forward(self, ids, K=1):
        encoded = self.encode(ids)
        return self.decode(self.reason(encoded.latents[-1], K), encoded)


class BPETransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.get("width", 512)
        self.embed = nn.Embedding(config.get("vocab", 16384), width)
        self.blocks = nn.Sequential(*(
            TransformerBlock(config["ff"], width) for _ in range(config.get("blocks", 10))
        ))
        self.norm = nn.RMSNorm(width, eps=1e-5)

    def encode(self, ids):
        return self.norm(self.blocks(self.embed(ids)))

    def forward(self, ids, K=1):
        if K != 1:
            raise ValueError("Control A has no recurrence")
        return F.linear(self.encode(ids), self.embed.weight)


def build_model(config, device):
    if config["model"] == "A":
        return BPETransformer(config).to(device)
    return HNet(config, "cuda" if torch.device(device).type == "cuda" else "reference").to(device)


def count_parameters(model):
    return sum(p.numel() for p in {id(p): p for p in model.parameters() if p.requires_grad}.values())
