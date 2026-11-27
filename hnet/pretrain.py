"""Matched BPE Transformer and single-stage byte H-Net."""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from .model import Route, Router, TransformerBlock, dechunk, gather, mamba_stack


@dataclass
class FixedRoute:
    valid: torch.Tensor
    boundaries: torch.Tensor
    index: torch.Tensor
    parents: torch.Tensor


def fixed_route(valid, ratio):
    # Each boundary is a chunk start, as with the public dynamic router.
    # ceil(j * ratio) gives alternating integer spans for fractional ratios.
    batch, length = valid.shape
    chunks = math.floor((length - 1) / ratio) + 1
    positions = torch.arange(length, device=valid.device)
    starts = (torch.arange(chunks, device=valid.device, dtype=torch.float64) * ratio).ceil().long()
    parents = torch.searchsorted(starts, positions, right=True) - 1
    boundaries = torch.zeros(length, dtype=torch.bool, device=valid.device)
    boundaries[starts] = True
    return FixedRoute(valid, boundaries & valid, starts.expand(batch, -1), parents.expand(batch, -1))


@dataclass
class Prediction:
    logits: torch.Tensor
    route: Route | FixedRoute | None = None


def core(config):
    width = config["width"]
    return nn.Sequential(*(TransformerBlock(config["ff"], width) for _ in range(config["blocks"])))


def initialize(module):
    """Small initial output scores for both heads, including the tied byte head."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=.02)


def output_mask(vocab):
    mask = torch.zeros(vocab, dtype=torch.bool)
    mask[-3] = mask[-1] = True  # BOS and PAD, EOS remains a prediction target.
    return mask


class PretrainBPE(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        width = config["width"]
        self.embed = nn.Embedding(vocab, width)
        self.blocks = core(config)
        self.norm = nn.RMSNorm(width, eps=1e-5)
        self.head = nn.Linear(width, vocab, bias=False)
        self.register_buffer("forbidden", output_mask(vocab), persistent=False)
        self.apply(initialize)

    def forward(self, ids, valid):
        h = self.norm(self.blocks(self.embed(ids)))
        return Prediction(self.head(h).masked_fill(self.forbidden, -torch.inf))


class PretrainHNet(nn.Module):
    def __init__(self, config, backend, fixed_ratio=None):
        super().__init__()
        width, depth = config["outer_width"], config["outer_depth"]
        self.embed = nn.Embedding(259, width)
        self.encoder = mamba_stack(width, depth, backend)
        self.encoder_norm = nn.RMSNorm(width, eps=1e-5)
        # Preserve the shared modules' initialization for the same seed.
        # Only the dynamic model registers the router as a parameterized module.
        router = Router(width)
        self.router = router if fixed_ratio is None else None
        self.fixed_ratio = fixed_ratio
        self.up = nn.Linear(width, config["width"], bias=False)
        self.blocks = core(config)
        self.core_norm = nn.RMSNorm(config["width"], eps=1e-5)
        self.down = nn.Linear(config["width"], width, bias=False)
        self.decoder = mamba_stack(width, depth, backend)
        self.out_norm = nn.RMSNorm(width, eps=1e-5)
        self.register_buffer("forbidden", output_mask(259), persistent=False)
        for module in (self.embed, self.up, self.blocks, self.down):
            module.apply(initialize)

    def encode(self, ids):
        return self.encoder_norm(self.encoder(self.embed(ids)))

    def forward(self, ids, valid):
        h = self.encode(ids)
        if self.router is None:
            route = fixed_route(valid, self.fixed_ratio)
        else:
            route = self.router(h, valid, torch.zeros_like(valid))
        z = self.up(gather(h, route.index))
        c = self.down(self.core_norm(self.blocks(z)))
        if self.router is None:
            residual = gather(c, route.parents)
        else:
            expanded = dechunk(c, route)
            confidence = route.assignment
            gate = (1 + (confidence - confidence.detach())).to(expanded.dtype)
            residual = expanded * gate[..., None]
        decoded = self.decoder(h + residual)
        logits = F.linear(self.out_norm(decoded), self.embed.weight)
        return Prediction(logits.masked_fill(self.forbidden, -torch.inf), route)


def ratio_loss(route, target):
    """Equation 10 of the public H-Net paper. Only F (hard count) is detached."""
    positions = route.valid.sum()
    hard = route.boundaries.sum() / positions
    soft = (route.probs * route.valid).sum() / positions
    return target / (target - 1) * ((target - 1) * hard * soft + (1 - hard) * (1 - soft))


def build_pretrain_model(kind, config, vocab, device):
    if kind == "bpe":
        return PretrainBPE(config, vocab).to(device)
    if kind in ("hnet", "hnet-fixed"):
        backend = "cuda" if device.type == "cuda" else "reference"
        fixed_ratio = float(config["fixed_ratio"]) if kind == "hnet-fixed" else None
        if fixed_ratio is not None and (not math.isfinite(fixed_ratio) or fixed_ratio <= 1):
            raise ValueError("Fixed compression must be finite and greater than 1")
        return PretrainHNet(config, backend, fixed_ratio).to(device)
    raise ValueError(f"Unknown pretraining model: {kind}")
