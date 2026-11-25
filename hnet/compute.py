"""Shape-based, reproducible estimates with padded compute included."""
import math

from torch import nn

from .mamba import MambaBlock
from .model import Attention


class ComputeEstimate:
    def __init__(self, model):
        self.flops = 0
        self.model = model
        self.hooks = []

    def _linear(self, module, inputs):
        x, = inputs
        positions = math.prod(x.shape[:-1])
        self.flops += 2 * positions * module.in_features * module.out_features

    def _attention(self, module, inputs):
        x = inputs[0]
        memory = inputs[1] if len(inputs) > 1 else None
        length = x.shape[1]
        pairs = length * (length + 1) // 2 if memory is None else length * memory.shape[1]
        self.flops += 4 * x.shape[0] * pairs * module.heads * module.dim

    def _ssm(self, module, inputs):
        x, = inputs
        mixer = module.mixer
        positions = x.shape[0] * x.shape[1]
        inner = 2 * x.shape[-1]
        self.flops += positions * (2 * 4 * (inner + 128) + 6 * inner * 64)
        if getattr(mixer, "use_mem_eff_path", False):
            self.flops += 2 * positions * inner * x.shape[-1]

    def _head(self, module, inputs):
        x, = inputs
        self.flops += 2 * math.prod(x.shape) * self.model.embed.num_embeddings

    def __enter__(self):
        for module in self.model.modules():
            hook = None
            if isinstance(module, nn.Linear):
                hook = self._linear
            elif isinstance(module, Attention):
                hook = self._attention
            elif isinstance(module, MambaBlock):
                hook = self._ssm
            if hook:
                self.hooks.append(module.register_forward_pre_hook(hook))
        norm = self.model.out_norm if hasattr(self.model, "out_norm") else self.model.norm
        self.hooks.append(norm.register_forward_pre_hook(self._head))
        return self

    def __exit__(self, *_):
        for hook in self.hooks:
            hook.remove()
