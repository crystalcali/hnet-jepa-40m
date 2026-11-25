"""Probe host capabilities and run CUDA forward/backward parity when requested."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import platform

import torch

from hnet.mamba import MambaBlock
from hnet.model import HNet
from hnet.training import nll


def package(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-smoke", action="store_true")
    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    print(json.dumps({
        "machine": platform.machine(), "os": platform.platform(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": cuda,
        "device": torch.cuda.get_device_name() if cuda else None,
        "cuda_capability": torch.cuda.get_device_capability() if cuda else None,
        "bf16": torch.cuda.is_bf16_supported() if cuda else None,
        "mamba_ssm": package("mamba-ssm"), "causal_conv1d": package("causal-conv1d"),
        "triton": package("triton"),
    }, indent=2), flush=True)
    if not args.cuda_smoke:
        return
    if not cuda or not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 CUDA required")
    import causal_conv1d
    if not callable(causal_conv1d.causal_conv1d_fn):
        raise SystemExit("causal-conv1d is missing its CUDA path")
    torch.manual_seed(42)
    ref = MambaBlock(128, "reference").eval()
    accelerated = MambaBlock(128, "cuda").cuda().eval()
    accelerated.load_state_dict(ref.state_dict())
    inputs = torch.randn(2, 16, 128)
    expected = ref(inputs).detach()
    actual = accelerated(inputs.cuda()).cpu().detach()
    torch.testing.assert_close(actual, expected, atol=.002, rtol=.002)
    model = HNet({"model": "C", "ff": 256, "widths": [64, 128, 192, 256],
                  "depths": [1, 1, 1], "blocks": 1}, backend="cuda").cuda()
    ids = torch.tensor([[257, 261, 97, 98, 120]], device="cuda")
    labels = torch.tensor([[-100, 97, 98, 120, 258]], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = nll(model(ids, K=2), labels)
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    print(json.dumps({"cuda_smoke": "pass", "mamba_parity_max_error": float((expected-actual).abs().max()),
                      "loss": float(loss.detach())}))


if __name__ == "__main__":
    main()
