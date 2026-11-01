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
- smoke v2: torchao fix confirmed (LoRA injected). Failed because the smoke ran GRPO in a process seeing 2 GPUs
  (TRL ≥1.x refuses nn.DataParallel). Main kernels launch one subprocess per format with `CUDA_VISIBLE_DEVICES` set, so they avoid this.
- main-b v1 failed on a missing results dir (fixed). main-a v1 and main-b v2 started ~21:47 local.
- main-b (kmer6, exact-match reward) complete: 100 steps in 16.5 min (~10 s/step), post-eval 213 s.
  Result: revcomp 0/1,600 correct training rollouts, 0% eval before and after; 76/100 steps had all-zero reward
  (grad_norm 0). Base model: 93% of revcomp answers have the wrong length; per-base agreement ≈ 25% (chance).
  GC count 1.7% → 2.1%. → Exact-match GRPO cannot bootstrap a skill at 0% pass@8.
- Human approved a second run with a partial-credit reward (revcomp: aligned matches / max length; GC: 1/(1+|err|)).
  Kernels `dna-grpo-dense-a` (raw, perbase) and `dna-grpo-dense-b` (kmer6); pre-RL baseline reused from the exact run.
