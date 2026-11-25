# NVIDIA and Arm document this PyTorch image for DGX Spark (GB10).
FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace
COPY requirements*.txt /opt/hnet-reqs/

RUN python3 -c 'import torch; assert torch.version.cuda, torch.__version__; print(torch.__version__, torch.version.cuda)' \
    && nvcc --version \
    && python3 -m pip install --no-cache-dir -r /opt/hnet-reqs/requirements-data.txt \
        -r /opt/hnet-reqs/requirements-spark-build.txt

# Compile against the image's Torch ABI and CUDA toolkit. Keep the container's
# Torch/Triton stack and avoid fetching incompatible prebuilt extension wheels.
RUN MAX_JOBS=4 CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps \
        -r /opt/hnet-reqs/requirements-spark-cuda.txt \
    && python3 -c 'from causal_conv1d import causal_conv1d_fn; from mamba_ssm.modules.mamba2 import Mamba2'
