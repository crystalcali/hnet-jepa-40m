# Implementation decisions

- The six approved amendments are in [PLAN.md](PLAN.md).
- CPU Mamba-2 keeps compatible parameter names and uses a direct causal scan.
  CUDA requires a working native Mamba-2 backend. Right-padded batches use a
  single document per sequence. No cross-document packing path is enabled.
- SDPA uses explicit KV expansion for a common CPU/CUDA attention path.
  Generation recomputes the target prefix while reusing mode-B prompt memory.
- SIGReg replays clean encoders with non-reentrant checkpointing to hold
  gradients across microbatches. Its clean L2/L3 objective logs per-language
  counts. Pooled and recurrence-state SIGReg are not part of this version.
- Corpus ingestion accepts pinned JSONL/Hugging Face field mappings. Exact
  per-side deduplication and overlap are automated. Near duplicates, semantic
  alignment and benchmark variants still require a reviewed upstream audit.
  The supplied catalog starts with no source enabled. It makes no licensing,
  quota or quality assertion.
- The runnable evaluator covers held-out BPB, basic retrieval/rank and the
  paired bilingual reasoning generator, and reads pinned normalized benchmark
  exports for continuation ranking, exact match, chrF++ and BLEU. Dataset-specific
  ingestion, code-execution benchmarks, language-ID probes and interactive internal
  trajectories are outstanding. Live metrics are read from the training journal.
