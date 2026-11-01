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
- main-a (exact, raw + perbase): both stopped by the 30-min cap at step 90. revcomp 0% pre/post in every format.
- dense-a / dense-b (partial credit): 100 steps each (raw 13.1 min, perbase 24.7, kmer6 17.7). Policy collapses to copying the
  input: 88% / 98% / 74% verbatim copies (raw / perbase / kmer6). No answers are reverse-only or complement-only.
- Figure QA (all opened and checked):
  - fig1: tokenization of one 40-bp eval sequence in each format (6-mer boundaries marked).
  - fig2: partial-credit reward curves + zero-std groups (dotted = exact run) + completion length.
  - fig3: partial-credit score by length bucket, pre vs post (exact match is 0 everywhere, noted).
  - fig4: per-base error at token edges vs inside tokens with a 75% chance line; 6-mer position; length errors.
  - fig5: probe grid GIF (121 KB) + final frame. fig6: before/after examples picked by rule (first id per bucket).
  - fig7: tokens per base vs post-RL partial credit. fig8: exact-match null (0/4,568). fig9: answer-type breakdown (hero).
  - social_card.png 1200×675.
