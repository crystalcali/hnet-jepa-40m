# English pre-training `bpe` and `hnet`

This experiment compares a small BPE Transformer with a byte H-Net. Both
use the same Transformer core, text windows and exposure schedule. The
one-stage H-Net uses the public 2025 H-Net router, a causal dechunker
and a skip connection. Cartesia's 2026 article motivates the matched
comparison. Its internal H-Net implementation and full training mix
are different from this experiment.

| Component | BPE Transformer | Byte H-Net |
| --- | --- | --- |
| Input | Train-only byte-level BPE, 8192 IDs including three controls | UTF-8, 259 IDs including three controls |
| Core | 10 blocks, width 512, GQA 8 Q / 2 KV, SwiGLU 1536 | Same |
| Outer layers | Separate embedding and output weights | 4 Mamba-2 blocks on each side, width 384, one dynamic router |
| Objective | Next-token cross-entropy | Next-byte cross-entropy and compression ratio loss |
| Trainable parameters | 38,545,920 | 38,498,976 |

The shared window limit is 2048 raw UTF-8 bytes, plus BOS on the input.
Each sequence comes from a single document window. BOS and PAD are
excluded from the output distribution. Both models train on EOS at the
end of the window. Validation BPB excludes EOS and divides total
negative log-likelihood by `ln(2) × UTF-8 bytes`. Compared models use the
same evaluation windows.

The target compression is the measured bytes per BPE token from up to
4096 fixed-seed train windows. The H-Net starts with a threshold
calibrated on a separate train-only sample and learns boundaries under
the public ratio loss (Eq. 10) at a starting coefficient of 0.10.
Report the actual fraction of retained positions separately. The two
models are matched in parameters and text exposure. FLOP estimates and
wall-clock time are measured independently. Peak CUDA allocation
refers to the current process. On a shared Spark, compare timings when
no other GPU workloads are running.

- [Cartesia 2026 comparison](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1)
- [Public H-Net architecture and ratio loss](https://arxiv.org/abs/2507.07955)

## Prepare on DGX Spark

From the repo root, use the native Spark environment. If
`cache/txt360-en-500mb` already contains the English JSONL files created
earlier, start with `tokenize`. Otherwise run `prepare` first. It downloads a
single pinned ~2 GB Parquet shard from TxT360-v2. It writes up to the first
whole document reaching 500 MB of unique English text overall, with
98%/1%/1% expected document split shares by SHA-256 buckets. Matching
documents are removed before splitting.

```bash
bash scripts/spark-native.sh run python3 -m data.pretrain_english prepare
bash scripts/spark-native.sh run python3 -m data.pretrain_english tokenize \
  --data cache/txt360-en-500mb
```

## Train and compare

Run the two loops sequentially in a Spark tmux session. The same
`--max-steps` and seed consume identical train windows. 1000 steps is a
first comparison, after which you can extend the step limit using the
same planned `--epochs` and the corresponding `--resume`.

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH="$TRITON_PTXAS_PATH"

for model in bpe hnet; do
  bash scripts/spark-native.sh run python3 pretrain_en.py train \
    --model "$model" --data cache/txt360-en-500mb \
    --run "runs/pretrain-en-$model" --max-steps 1000 || break
done
```

The dashboard can read both new `metrics.jsonl` files. Select the desired
run in the dropdown. In the log, `source_bytes_total` is the comparison
axis, `estimated_flops_total` is a shape-based approximation, and
`source_bytes_per_s` includes data preparation of each batch. Validation
defaults to the same 128 spaced windows, every 500 steps and at the end.

```bash
bash scripts/spark-native.sh dashboard --port 0
```

Once the setup is selected, score both checkpoints on the same complete
test split. Use `--windows 128` for a quick limited check.

```bash
for model in bpe hnet; do
  bash scripts/spark-native.sh run python3 pretrain_en.py eval \
    --data cache/txt360-en-500mb --checkpoint "runs/pretrain-en-$model/latest"
done
```

## Fixed chunk control

The `hnet-fixed` model uses the same byte embedding, Mamba-2 stacks,
Transformer core, residual merge and next-byte objective as `hnet`.
With a matching seed its shared parameters start identically. It uses a
deterministic grid of chunk starts, directly repeats the latest available
chunk output, and has no router or ratio loss. At a fractional rate of
3.75 the grid uses spans of three and four positions. No span can pass
information backward to positions before its start. The fixed arm has
38,204,064 parameters, 294,912 fewer than the dynamic arm.

`--fixed-ratio` defaults to the train-only BPE ratio. For comparison with
the existing dynamic checkpoint at step 1000, use `--fixed-ratio 3.75`,
rounded from its observed validation compression (3.7453). Both models
then read the same windows and start from the same shared weights.
Check actual validation compression, BPB, estimated FLOPs and elapsed
time. Differences include the dynamic path's confidence-weighted
smoothing and router loss. This control measures the effect of the full
dynamic routing path.

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH="$TRITON_PTXAS_PATH"

bash scripts/spark-native.sh run python3 pretrain_en.py train \
  --model hnet-fixed --fixed-ratio 3.75 --data cache/txt360-en-500mb \
  --run runs/pretrain-en-hnet-fixed --max-steps 1000

bash scripts/spark-native.sh run python3 pretrain_en.py eval \
  --data cache/txt360-en-500mb --split test \
  --checkpoint runs/pretrain-en-hnet-fixed/checkpoints/step-000001000
```

For a resume, pass the same `--fixed-ratio` with `--resume`. The dashboard
lists the fixed arm as a separate run.
