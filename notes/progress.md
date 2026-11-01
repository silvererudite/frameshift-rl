# Progress log

## 2026-11-01
- Kaggle auth OK (user `shamimahossain`). CLI 2.2.5.
- Tokenizer stats (CPU, Qwen3 tokenizer): raw BPE ≈ 0.52 tokens/base, kmer6 ≈ 0.55 (BPE still splits each 6-mer), perbase = 1.0.
  ~94% of bases in raw/kmer6 sit at a token edge (tokens are mostly 1–3 bases), so "interior" bases are a small population.
- `dna-grpo-smoke` v1: 2× Tesla T4 confirmed (`machine_shape: NvidiaTeslaT4` gives T4 x2). torch 2.11.0+cu128,
  transformers 5.16.1, trl 1.17.1, Python 3.13. Base model eval + reward parsing work (answers use `<answer>` tags).
  Failed at LoRA injection: Kaggle's preinstalled torchao 0.10 is rejected by current PEFT → kernel header now uninstalls torchao.
- Kaggle limit: max 2 concurrent batch GPU sessions. Main run split into `dna-grpo-main-a` (raw + perbase, one GPU each)
  and `dna-grpo-main-b` (kmer6). Each training run has a 30-min wall-clock cap (callback stops training and records the step).
