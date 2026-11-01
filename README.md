# frameshift-rl · Does RL fix what the tokenizer breaks?

A small, controlled reinforcement-learning experiment on DNA. **Same model, same tasks, same sequences, same reward; only
the way the DNA is cut into tokens changes** (run twice: with an exact-match and a partial-credit reward). Inspired by FineDNA / Carbon-X (Hugging Face Bio), whose 6-mer
tokenization makes base-level operations such as reverse complement cut *across* token boundaries.

![How the model sees DNA](figures/fig1_how_the_model_sees_dna.png)

## Key results

**Question.** If you train a small LM with GRPO on exact, checkable DNA tasks, does RL teach it to work around how its
input is segmented, or does the segmentation cap what RL can reach?

**Answer for Qwen3-0.6B: neither. RL never got to the question, and the reason is the interesting part.**

1. **Exact-match reward → no learning signal.** 0 of 4,568 reverse-complement rollouts were correct across all three
   formats. Every prompt group had zero reward variance, so GRPO's gradient was exactly zero ([Fig 8](figures/fig8_sparse_reward_null.png)).
2. **Partial-credit reward → reward hacking.** Rewarding the fraction of correct bases taught the policy to **copy the input
   verbatim** (88% raw · 98% per-base · 74% 6-mer of answers after RL, up from 8% · 17% · 5%). Copying always gets the
   length right and scores ~25% by chance. Per-base accuracy stayed at chance.
3. **The format effect showed up in the copying, not in the skill.** One-token-per-base input produced the most
   length-faithful outputs, and the 6-mer view the least. Errors are at chance whether a base sits at a token edge or
   inside a token, so there is no boundary effect without an underlying skill.

![Partial-credit RL taught the model to copy](figures/fig9_what_the_answers_are.png)

**Takeaway:** RL with verifiable rewards amplifies behaviour the model already has. When pass@k is 0, exact rewards give
nothing to learn from, and naive shaped rewards get gamed. A curriculum or an anti-copy reward comes before any
tokenizer comparison.

Full write-up with every number, CI and caveat: **[FINDINGS.md](FINDINGS.md)** · interactive page: [`site/index.html`](site/index.html)

## The experiment in one table

| | |
|---|---|
| Model | Qwen3-0.6B (fp32) + LoRA r=16 on all linear projections, non-thinking chat template |
| RL | GRPO via TRL 1.17: 8 completions per prompt, 4 prompts per step, β = 0, lr 2e-5 |
| Tasks | reverse complement and GC count on random DNA, 12–90 bp (task family from FineDNA `rl_mix`) |
| Input formats | `raw` `ACGTAC…` (BPE ≈ 2 bases/token) · `perbase` `A C G T …` (1 base/token) · `kmer6` `ACGTAC GATTCA …` (Carbon-X-style view) |
| Rewards | **exact match** (1/0) and **partial credit** (fraction of aligned correct bases; 1/(1+\|error\|) for counts) |
| Eval | 240 held-out prompts (2 tasks × 3 length buckets × 40) × 4 samples, identical sequences in every format, 95% bootstrap CIs |
| Compute | Kaggle, 2× Tesla T4, ~2.5 GPU-hours in total |

## Figures

| | |
|---|---|
| [Fig 1](figures/fig1_how_the_model_sees_dna.png) | How each format tokenizes the same sequence |
| [Fig 2](figures/fig2_learning_curves.png) | Reward curves and GRPO health signals (zero-variance groups, completion length) |
| [Fig 3](figures/fig3_length_cliff.png) | Partial-credit score by sequence length, before vs after RL |
| [Fig 4](figures/fig4_error_positions.png) | Where errors happen relative to token boundaries |
| [Fig 5](figures/fig5_learning.gif) | Animated: per-base correctness on 24 probe prompts across training |
| [Fig 6](figures/fig6_examples.png) | Before/after examples (picked by rule, not by outcome) |
| [Fig 7](figures/fig7_compression_vs_accuracy.png) | Input compression vs accuracy |
| [Fig 8](figures/fig8_sparse_reward_null.png) | Why exact-match GRPO learns nothing here |
| [Fig 9](figures/fig9_what_the_answers_are.png) | What the answers actually are: partial-credit RL converges on copying |

## Repo layout

```
dna_grpo.py              single-file library: data, formats, rewards, eval, GRPO training (runs as a Kaggle kernel)
tests/test_rewards.py    reward/parsing/data tests (pytest, CPU)
scripts/build_kernel.py  writes kernels/<name>/ (entry.py + kernel-metadata.json)
scripts/push_and_wait.sh push a kernel, poll, download outputs
scripts/tokenizer_stats.py  tokens/base and token-boundary offsets per example (CPU)
scripts/make_figures.py  every figure + figures/summary.json (all quoted numbers)
scripts/build_site.py    site/index.html (self-contained interactive results page)
results/                 raw eval completions, per-step rewards, train logs, probe snapshots (both runs)
notes/                   progress log, FineDNA schema notes, post draft
```

## Reproduce

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install kaggle six pytest matplotlib numpy pandas scipy transformers tokenizers datasets imageio imageio-ffmpeg
pytest -q tests
python scripts/tokenizer_stats.py
# GPU runs on Kaggle (private kernels; set USER in scripts/build_kernel.py to your Kaggle username)
python scripts/build_kernel.py main --name main-a  --formats raw,perbase
python scripts/build_kernel.py main --name main-b  --formats kmer6
python scripts/build_kernel.py main --name dense-a --formats raw,perbase --budget-min 25 --env REWARD=dense --env SKIP_PRE=1
python scripts/build_kernel.py main --name dense-b --formats kmer6        --budget-min 25 --env REWARD=dense --env SKIP_PRE=1
for k in main-a main-b dense-a dense-b; do scripts/push_and_wait.sh $k; done   # Kaggle allows 2 GPU sessions at once
python scripts/make_figures.py && python scripts/build_site.py
```

Versions inside the kernels: torch 2.11.0+cu128, transformers 5.16.1, TRL 1.17.1, Python 3.13.

---
Shamima Hossain · 2026
