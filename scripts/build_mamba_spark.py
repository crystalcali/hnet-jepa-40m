"""Build the pinned Mamba release with only the DGX Spark GPU target."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen


VERSION = "2.2.6.post3"
URL = (
    "https://files.pythonhosted.org/packages/b6/0c/"
    "9373a469ff7a33bdd0644e55fa45165ba3900274dcf7fe9f10ccc232aef9/"
    "mamba_ssm-2.2.6.post3.tar.gz"
)
SHA256 = "826a3cdb651959f191dac64502f8a29627d9116fe6bb7c57e4f562da1aea7bf3"
MARKER = '            cc_flag.append("arch=compute_121,code=sm_121")\n'


def spark_setup(text):
    if text.count(MARKER) != 1:
        raise RuntimeError("Unexpected CUDA architecture section in Mamba setup.py")
    # Replace the accumulated multi-architecture flags before NVCC receives them.
    return text.replace(MARKER, '            cc_flag[:] = ["-gencode", "arch=compute_121,code=sm_121"]\n')


def main():
    import torch

    if torch.cuda.get_device_capability(0) != (12, 1):
        raise SystemExit("The fast Mamba build targets a DGX Spark GPU with compute capability 12.1.")
    try:
        installed = version("mamba-ssm")
    except PackageNotFoundError:
        installed = None
    if installed == VERSION:
        print(f"Mamba {VERSION} is already installed, attempting the smoke-test.")
        return

    print(f"Building Mamba {VERSION} for sm_121 only.", flush=True)
    with urlopen(URL, timeout=60) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != SHA256:
        raise SystemExit("Mamba source checksum mismatch.")

    with tempfile.TemporaryDirectory(prefix="hnet-mamba-") as work:
        with tarfile.open(fileobj=io.BytesIO(source), mode="r:gz") as archive:
            archive.extractall(work, filter="data")
        project = Path(work) / f"mamba_ssm-{VERSION}"
        setup = project / "setup.py"
        setup.write_text(spark_setup(setup.read_text()))
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-v", "--no-build-isolation",
             "--no-deps", str(project)],
            env=os.environ | {"MAMBA_FORCE_BUILD": "TRUE"},
            check=True,
        )


if __name__ == "__main__":
    main()
