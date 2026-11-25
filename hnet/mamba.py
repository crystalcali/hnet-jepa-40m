"""Mamba-2 CUDA blocks and a small, differentiable PyTorch reference."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceMamba2(nn.Module):
    def __init__(self, width, state_size=64, head_dim=64):
        super().__init__()
        self.inner = 2 * width
        self.head_dim = head_dim
        self.heads = self.inner // head_dim
        self.state_size = state_size
        channels = self.inner + 2 * state_size
        self.in_proj = nn.Linear(width, self.inner + channels + self.heads, bias=False)
        self.conv1d = nn.Conv1d(channels, channels, 4, groups=channels, padding=3)
        dt = torch.exp(torch.empty(self.heads).uniform_(math.log(.001), math.log(.1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.heads))
        for p in (self.dt_bias, self.A_log, self.D):
            p._no_weight_decay = True
        self.norm = nn.RMSNorm(self.inner, eps=1e-5)
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def forward(self, inputs):
        batch, length, _ = inputs.shape
        gate, xbc, dt = self.in_proj(inputs).split(
            [self.inner, self.inner + 2 * self.state_size, self.heads], dim=-1
        )
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2))[..., :length].transpose(1, 2))
        x, b, c = xbc.float().split([self.inner, self.state_size, self.state_size], dim=-1)
        x = x.reshape(batch, length, self.heads, self.head_dim)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        decay = torch.exp(-torch.exp(self.A_log.float()) * dt)
        state = x.new_zeros(batch, self.heads, self.head_dim, self.state_size)
        output = []
        for i in range(length):
            u = x[:, i]
            state = (
                decay[:, i, :, None, None] * state
                + (dt[:, i, :, None] * u)[..., None] * b[:, i, None, None, :]
            )
            output.append((state * c[:, i, None, None, :]).sum(-1) + u * self.D[:, None])
        y = torch.stack(output, dim=1).reshape(batch, length, self.inner)
        y = self.norm(y * F.silu(gate.float()))
        return self.out_proj(y.to(inputs.dtype))


class MambaBlock(nn.Module):
    def __init__(self, width, backend):
        super().__init__()
        self.norm = nn.RMSNorm(width, eps=1e-5)
        if backend == "cuda":
            try:
                from mamba_ssm.modules.mamba2 import Mamba2
            except ImportError as error:
                raise RuntimeError("CUDA requires a working mamba-ssm and causal-conv1d build") from error
            self.mixer = Mamba2(
                d_model=width, d_state=64, d_conv=4, expand=2, headdim=64,
                ngroups=1, bias=False, conv_bias=True, rmsnorm=True,
                norm_before_gate=False, D_has_hdim=False, d_ssm=None,
            )
        else:
            self.mixer = ReferenceMamba2(width)

    def forward(self, x):
        return x + self.mixer(self.norm(x))
