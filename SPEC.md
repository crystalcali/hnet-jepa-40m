# H-Net-JEPA-40M

Integrated implementation specification. The project starts from random weights,
with no pretrained language model, external tokenization of the H-Net arms, EMA
target encoder, or RL. The six implementation amendments are normative in
[PLAN.md](PLAN.md).

## Architecture and training

- Byte vocabulary 0–255 plus PAD, BOS, EOS, MASK, SEP, EN, RU and TASK. Add
  controls during collation and mask unused output classes. UTF-8 NFC, up to
  8192 positions including controls. Tied embedding and output projection.
- Three causal H-Net levels with widths 128/256/384/512, independent encoder and
  decoder stacks of 6/4/4 Mamba-2 blocks. Mamba-2 uses state size 64, expansion
  2, convolution width 4, head dimension 64, group count 1. Cosine boundary
  probabilities, hard chunk starts, straight-through routing and causal EMA
  expansion. Train-only quantile calibration and empirical hard rate target 36×.
- Model C: recurrent eight-block Transformer with RMSNorm, rotary positions,
  GQA 8/2 and SwiGLU hidden width 1536. Keep all eight weights shared across
  iterations. Deterministic iteration embeddings, learned reinjection, final
  and sampled intermediate decodes. K=1 until 20% exposure, then uniform 1…8.
  Evaluate K>8 as out of training depth. Model B: K=1, hidden width 1664,
  same hierarchy and conditional memory. Model A: width 512, hidden width
  1600, ten blocks, tied 16,384 BPE vocabulary, causal concatenation for
  conditional CE.
- Mode A predicts the next byte from the full hierarchy. Mode B computes
  prompt memory once with a terminal EOS propagated through all levels. The
  target prefix passes through the same hierarchy and queries that memory
  with a separate causal decoder path. Prompt and answer together fit in
  8192 raw positions. Test prompt/target independence before caching.
- Symmetric L2/L3 pooled and verified-span optimal-transport JEPA, ablation
  JEPA, fixed-horizon Dynamics JEPA, translations and supervised answer CE.
  Train both sides of representation losses. Apply Epps–Pulley SIGReg on
  balanced language samples across the optimizer step. Match byte and
  training exposure, log selected and eligible counts separately.
- AdamW (.9, .95), learning rate warmup and cosine decay, BF16 CUDA forward
  with FP32 statistics, gradient clipping, separate parameter grouping and
  atomic resumable checkpoints. Core gradient checkpointing. Preserve the
  independent CPU Mamba-2 reference and validate against the CUDA backend.
- Analytic counts A/B/C: 39,528,960 / 39,922,416 / 39,840,497. Include
  training-only heads in the 40M budget.

## Corpus and evaluation

- A 30M BPE-equivalent pilot precedes the 1B target. Counts sum both sides
  once per unique training pair. Freeze the train-only pilot BPE tokenizer
  and reuse for all arms and full measurement. Six exclusive corpus domains
  at 50/20/15/5/5/5%. Only existing published paired data with allowed
  licensing and verified provenance. Export split manifests, deficits and
  contamination coverage before research training.
- Keep benchmark data separate from train. Report grouped RU/EN BPB, the
  empirical compression rates, held-out cross-language retrieval and latent
  rank, and paired K comparisons on frozen bilingual problems. Comparisons
  by data exposure, estimated compute and elapsed time are independent.
- Pilot entrance requires verified CUDA kernels, quota coverage and objective
  eligibility. A genuine smaller corpus is diagnostic only. The 72-hour
  budget per full arm includes checkpoint and validation overhead. Full
  training is an explicit standalone command.

## Sources

- [H-Net paper](https://arxiv.org/abs/2507.07955)
- [H-Net reference](https://github.com/goombalab/hnet)
- [Mamba-2 source](https://github.com/state-spaces/mamba)
- [LeJEPA / SIGReg](https://arxiv.org/abs/2511.08544)
