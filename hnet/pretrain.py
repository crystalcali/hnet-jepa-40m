"""Matched BPE Transformer and single-stage byte H-Net."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import Route, Router, TransformerBlock, dechunk, gather, mamba_stack


@dataclass
class Prediction:
    logits: torch.Tensor
    route: Route | None = None


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
    def __init__(self, config, backend):
        super().__init__()
        width, depth = config["outer_width"], config["outer_depth"]
        self.embed = nn.Embedding(259, width)
        self.encoder = mamba_stack(width, depth, backend)
        self.encoder_norm = nn.RMSNorm(width, eps=1e-5)
        self.router = Router(width)
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
        route = self.router(h, valid, torch.zeros_like(valid))
        z = self.up(gather(h, route.index))
        c = self.down(self.core_norm(self.blocks(z)))
        expanded = dechunk(c, route)
        confidence = route.assignment
        gate = (1 + (confidence - confidence.detach())).to(expanded.dtype)
        decoded = self.decoder(h + expanded * gate[..., None])
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
    if kind == "hnet":
        backend = "cuda" if device.type == "cuda" else "reference"
        return PretrainHNet(config, backend).to(device)
    raise ValueError(f"Unknown pretraining model: {kind}")
