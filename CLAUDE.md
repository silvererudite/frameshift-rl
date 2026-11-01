# DNA-GRPO: Does RL fix what the tokenizer breaks?

A small, sharp RL experiment on DNA tasks inspired by FineDNA / Carbon-X (Hugging Face Bio, Oct 2026).
This file is for **both** the human (Elma) and the coding agent (Claude Code). Humans: read Part A. Agent: read everything, follow Part B step by step.

---

## The research question (read this first, both of you)

Carbon-X encodes DNA as non-overlapping 6-mer tokens, and its model card admits it miscounts bases and confuses coordinates. Base-level operations (reverse complement, GC counting) cut *across* token boundaries, whether those tokens are Qwen BPE chunks or 6-mers.

**Question:** When we train a small LM with GRPO on exact-match DNA tasks, does RL teach the policy to work around its input segmentation, or does the input representation set a ceiling RL cannot lift?

**Design:** same model, same tasks, same reward, same sequences, three input formats:

| Format id | Example input | What it simulates |
|---|---|---|
| `raw` | `ACGTACGATTCA` | vanilla BPE (irregular chunks) |
| `perbase` | `A C G T A C G A T T C A` | one token per base |
| `kmer6` | `ACGTAC GATTCA` | Carbon-X's 6-mer view, in a vanilla model |

**Core result we need:** reward curves, accuracy-vs-length before/after RL, and an error analysis that shows *where* the model fails relative to token boundaries.

**This project doubles as a CV/portfolio piece.** Figures are a first-class deliverable, not an afterthought. See Part C.

---

# Part A: Human setup (≈10 minutes, do once)

1. **Kaggle account ready for GPUs**
   - Phone-verify your Kaggle account (Settings → Phone verification). Without this, kernels silently run without a GPU.
   - Open any notebook in the Kaggle web UI once and switch Accelerator to **GPU T4 x2**, then stop it. This unlocks GPU for API-pushed kernels.
   - Check your remaining weekly GPU quota on kaggle.com (top of the notebook editor). This project should use roughly 2–3 GPU-hours.

2. **Kaggle API token**
   - kaggle.com → Settings → API → **Generate New Token** (KGAT_… token).
   - Save it on the machine where Claude Code runs:
     ```bash
     mkdir -p ~/.kaggle
     printf '%s' 'KGAT_your_token_here' > ~/.kaggle/access_token
     chmod 600 ~/.kaggle/access_token
     ```
   - (A legacy `~/.kaggle/kaggle.json` also works if you already have one.)
   - Install/upgrade the CLI and test:
     ```bash
     pip install -U "kaggle>=1.8"
     kaggle kernels list --mine --page-size 1
     ```
   - Tell the agent your Kaggle username (it needs it for kernel ids).

3. **Start Claude Code** in this repo folder and say:
   > Read CLAUDE.md and run the project. You have my approval to push private Kaggle GPU kernels for this project.

4. **Your checkpoints** (the agent will stop and ask you at these):
   - after the GPU smoke test (to confirm T4s were assigned),
   - before the main training kernel (≈60–90 min of GPU),
   - when figures are ready (to pick the hero figure and title for posting).

**What you get at the end:** `FINDINGS.md`, `figures/` (PNG + SVG + a social card), `site/index.html` (a self-contained interactive results page you can drop into your GitHub Pages site), and a draft post.

---

# Part B: Agent runbook

## B0. Ground rules (non-negotiable)

- **Credentials:** never print, log, echo, or commit the Kaggle token or any file under `~/.kaggle/`. To check auth, run a harmless command (`kaggle kernels list --mine --page-size 1`) and only report success/failure.
- **GPU work runs on Kaggle, not locally.** Local machine = CPU only: data generation, unit tests, tokenizer stats, plotting, building kernels, pulling outputs. Do not try to train or run model inference locally beyond a few CPU tokenizer calls.
- **Kernels are always private** (`"is_private": true`). Never make anything public without asking.
- **Ask the human before:** the first GPU push (smoke test), the main training push, deleting any Kaggle kernel or dataset, or changing the experimental design.
- **Do not guess library APIs.** TRL changes fast. Inside the kernel, log `trl.__version__`, and when unsure, inspect `GRPOConfig` / `GRPOTrainer` signatures (`inspect.signature`) in the smoke test and adapt.
- Keep a running log in `notes/progress.md`: what ran, kernel ids, wall time, anything surprising.
- Use a task list; tick items as they finish.

## B1. Repo layout (create this)

```
.
├── CLAUDE.md
├── dna_grpo.py            # single-file library: data, formats, rewards, eval, train (all stages)
├── tests/test_rewards.py  # pytest, CPU
├── scripts/
│   ├── build_kernel.py    # writes kernels/<stage>/ (entry.py + kernel-metadata.json)
│   ├── push_and_wait.sh   # push, poll status, download outputs
│   └── make_figures.py    # all figures from outputs/ (CPU)
├── kernels/               # generated, gitignored
├── outputs/               # downloaded from Kaggle, gitignored
├── figures/               # final figures (PNG 300dpi + SVG)
├── site/index.html        # interactive results page
├── notes/progress.md
└── FINDINGS.md
```

`.gitignore` must include: `kernels/`, `outputs/`, `.kaggle/`, `*.token`, `access_token`, `kaggle.json`, `__pycache__/`.

**Why single-file:** Kaggle script kernels run one file. `build_kernel.py <stage>` copies `dna_grpo.py` into `kernels/<stage>/entry.py` and appends a footer that calls the stage (e.g. `main(stage="train")`). Data is generated in-kernel from a fixed seed, so nothing needs uploading.

## B2. Inspect FineDNA `rl_mix` (CPU, ≤10 min)

- Stream `HuggingFaceBio/FineDNA`, config `rl_mix` (`datasets.load_dataset(..., streaming=True)`), print columns and ~10 examples, and count task types if a task field exists.
- Write findings to `notes/rl_mix_schema.md`.
- **Decision rule:** if rl_mix has basic-DNA-operation rows (reverse complement / counting) with exact, code-checkable answers and lengths ≤ ~90 bp, sample our train/eval prompts from them and record that. Otherwise generate procedurally (below) and say so in FINDINGS.md. Either way, *mention FineDNA as the source of the task family*.

## B3. Tasks, formats, data (in `dna_grpo.py`)

- **Tasks:**
  - `revcomp`: return the reverse complement.
  - `gc_count`: return the number of G + C bases (integer).
- **Lengths:** uniform over 12–90 bp. **Buckets** for analysis: `12–30`, `31–60`, `61–90`.
- **Splits (fixed seed 1234):** 2,000 train prompts, 240 eval prompts (balanced: 2 tasks × 3 buckets × 40). The **same underlying sequences** are rendered in all three formats.
- **Prompt template** (plain text, rendered through the chat template with `enable_thinking=False`, then passed to TRL as a *string* prompt, not a conversation):
  ```
  Here is a DNA sequence: {seq_formatted}
  Task: {task_instruction}
  Respond with only the answer inside <answer></answer>.
  ```
  For `kmer6`, chunk from the left into groups of 6 separated by single spaces (last chunk may be shorter).
- **Every example record carries metadata:** `id, task, format, length, bucket, seq, target, prompt`.

## B4. Reward + metrics (in `dna_grpo.py`) and tests

- Parse the **last** `<answer>…</answer>`; if missing → reward 0, flag `parse_fail`.
- Normalize: uppercase, strip all whitespace (so the model may answer in any spacing).
- **Reward (training): exact match, 1.0 / 0.0.** No partial credit.
- **Metrics (logged only, never rewarded):**
  - `exact`, `parse_fail`, `len_mismatch` (revcomp only),
  - `per_position_correct`: for revcomp when lengths match, a boolean list per output position,
  - `hamming_sim`: fraction of matching positions when lengths match.
- `tests/test_rewards.py` (pytest, must pass before any GPU push): correct revcomp, spaced answer still correct, missing tag, two answer tags (last wins), wrong length, lowercase, GC count with whitespace.

## B5. Tokenizer statistics (CPU, before GPU)

Using the `Qwen/Qwen3-0.6B` tokenizer only (no model weights):
- tokens-per-base for each format and length bucket,
- for each eval example and format, **character offsets of token boundaries in the DNA span** (needed for the boundary-error analysis in Part C).
Save to `outputs/tokenizer_stats.jsonl`.

## B6. Kaggle plumbing

**kernel-metadata.json template** (`build_kernel.py` fills it):
```json
{
  "id": "<KAGGLE_USERNAME>/dna-grpo-<stage>",
  "title": "dna-grpo-<stage>",
  "code_file": "entry.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
```
- **Always set `machine_shape` to a T4 value.** The default GPU can be a P100 (compute capability 6.0), which current PyTorch does not support.
- **Verify, don't assume, the GPU count.** The smoke test prints `nvidia-smi` and `torch.cuda.device_count()`. If you get 1 T4, run formats sequentially; if 2, run two formats in parallel. If the T4-x2 shape needs a different `machine_shape` string, try it in the smoke test and record which value worked in `notes/progress.md`.
- Internet must be on (pip + Hugging Face Hub).
- At the top of every kernel: `pip install -q -U trl peft datasets accelerate` (add `vllm` only if you decide to use it), then print versions.
- Write all results to `/kaggle/working/results/…`. That directory is what `kaggle kernels output` downloads.

**`push_and_wait.sh <stage>`:**
1. `kaggle kernels push -p kernels/<stage>`
2. Poll `kaggle kernels status <user>/dna-grpo-<stage>` every 60 s, in the background (don't block on foreground sleep). Statuses: queued, running, complete, error, cancel*.
3. On `complete` or `error`: `kaggle kernels output <user>/dna-grpo-<stage> -p outputs/<stage>`, then show the tail of the log.
4. On `error`: read the log, fix, rebuild, and re-push. Max 3 retries per stage, then stop and report to the human.

## B7. Stages

**Stage `smoke` (≈5 min GPU; ask the human first)**
- Print GPU info and library versions, plus `inspect.signature(GRPOConfig)` (save to file).
- Load Qwen3-0.6B, generate 4 answers for 4 eval prompts per format, run the reward on them.
- Run **3 GRPO steps** with the real config on `perbase`, to catch API and memory errors.
- **Done when:** a T4 is confirmed, rewards are computed, 3 training steps complete, and outputs are downloaded. Report the GPU type/count to the human.

**Stage `main` (≈60–90 min GPU; ask the human first)**
One kernel does everything, so the model downloads once per process:
1. **Baseline eval** of the base model on all 240 eval prompts × 3 formats, 4 samples each (temperature 0.7, top-p 0.8, max 192 new tokens). Save every completion with full metadata to `results/eval_pre_<format>.jsonl`.
2. **GRPO per format.** Order: `raw` and `perbase` first (in parallel if 2 GPUs, via subprocesses with `CUDA_VISIBLE_DEVICES=0` and `1`), then `kmer6`.
3. **Post-RL eval**, identical to the baseline → `results/eval_post_<format>.jsonl`.
4. Save `trainer.state.log_history` → `results/train_log_<format>.json`.
5. Save LoRA adapters → `results/adapter_<format>/` (small; useful for the demo).

**GRPO config** (verify names against the logged signature; adapt if TRL renamed things):
- model `Qwen/Qwen3-0.6B`, **fp32** (T4 has no bf16; fp16 Qwen can overflow), LoRA r=16, alpha=32, target all linear projections
- `learning_rate=2e-5`, `max_steps=100`, `num_generations=8`, 4 prompts/step (so the per-step batch is 32 completions; respect TRL's divisibility rule)
- `max_completion_length=192`, `temperature=1.0`, `beta=0.0`
- `logging_steps=1`, `save_strategy="no"`, `report_to="none"`, `seed=1234`
- If a step takes >25 s, drop to `max_steps=60` and note it. If OOM: halve the prompts/step first, then `num_generations` → 6.

**Probe callback (needed for the animated figure, see C4):** every 10 steps, greedy-decode a fixed set of **24 revcomp probe prompts** (8 per bucket) for the current format and save `results/probe_<format>_step<N>.jsonl` with `per_position_correct`. Keep it cheap (greedy, batch 24).

**Logged training signals** to keep (print available `log_history` keys and keep whichever of these exist): reward mean/std, fraction of groups with zero reward std (all-correct or all-wrong), mean completion length, clipped-completion ratio, entropy, grad norm, step time.

## B8. After `main`

1. Download outputs, then run `python scripts/make_figures.py` (Part C).
2. Fill in `FINDINGS.md` (template in Part D).
3. Build `site/index.html` (C6).
4. Show the human the figure gallery and ask which one is the hero figure.

## B9. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available() == False` | Account not phone-verified, or GPU never enabled in the UI once (Part A1) |
| P100 assigned / "no kernel image" CUDA error | Set `machine_shape` to a T4 value, rebuild, push |
| NaN loss | Make sure the model is fp32, not fp16 |
| Reward stuck at exactly 0 everywhere | Check parsing: print 5 raw completions. The model may skip `<answer>` tags → check chat template / `enable_thinking=False` |
| Reward ≈ 1 from step 0 on gc_count | Fine for that task; report it per task. Revcomp carries the story |
| Kernel killed at time limit | Lower `max_steps`; run `kmer6` in a second kernel |
| pip resolver conflicts | Pin to versions printed in the smoke test |

---

# Part C: Figures (first-class deliverable)

These figures are going on a CV, a personal site, and social media. Each one must make **one claim** that is readable in 5 seconds.

## C0. Style rules (apply to every figure)

- **One fixed colour per format everywhere** (Okabe-Ito, colourblind-safe): `raw` = #E69F00 (orange), `perbase` = #0072B2 (blue), `kmer6` = #009E73 (green). Pre-RL = dashed/hollow, post-RL = solid/filled.
- **Titles state the finding, not the variables.** Good: "Per-base input lifts revcomp accuracy at every length; RL doesn't close the gap for BPE". Bad: "Accuracy vs length". Write the title *after* looking at the data. If the data doesn't support a strong claim, say what it does show.
- A subtitle line with setup: `Qwen3-0.6B · GRPO (TRL) · 100 steps · n=240 eval prompts × 4 samples`.
- Error bars: 95% bootstrap CIs (1,000 resamples over prompts) on every accuracy number.
- Minimal chrome: no top/right spines, light grid on y only, direct labels on lines instead of a legend where possible.
- Font ≥ 11 pt at final size. Export **PNG (300 dpi) and SVG** for each, named `figures/fig<N>_<slug>.{png,svg}`.
- Never show anything you can't defend: label sample sizes and don't smooth away the raw signal (show the raw reward faintly, with a rolling mean on top).

## C1. Fig 1: "How the model sees DNA" (the hook)

One real eval sequence (~40 bp) drawn three times, one row per format, as coloured blocks where **each block is one token** (alternate two tints of the format colour so boundaries pop). Underneath, the reverse-complement target with arrows showing that output base *i* depends on input base *L−1−i*. Annotate tokens-per-base for each row.
This is the figure that makes a non-specialist get the idea instantly. Make it beautiful.

## C2. Fig 2: Learning curves

Reward vs step for each format (raw faint, 10-step rolling mean bold), revcomp and gc_count as two panels. Add a second row with small multiples of **fraction of zero-std groups** and **mean completion length**. Those are the RL-health signals that show you understand GRPO, not just the headline number.

## C3. Fig 3: The length cliff, before vs after RL

x = length bucket, y = revcomp exact-match accuracy. Per format: dashed line = pre-RL, solid = post-RL, with a shaded band between them ("what RL bought you"). Annotate the post-RL gap between the `perbase` and `raw` lines at 61–90 bp.

## C4. Fig 4: Where the errors happen (the key analysis)

For revcomp completions with the correct length, map each output position back to its input position (*L−1−i*) and classify that input base as **token-boundary** (first or last char of its input token) or **token-interior**, using the offsets from B5.
- Panel A: error rate, boundary vs interior, per format, pre and post RL, with CIs. (For `perbase` every base is its own token, so it's the control.)
- Panel B: for `kmer6`, error rate by **position within the 6-mer (0–5)**, pre vs post. A U-shape or edge spike is the visual proof that segmentation drives errors.
- Also report the **length-mismatch rate** pre vs post per format as a small inset or table. Does RL fix length errors before substitution errors?

If there is no boundary effect, keep the figure and say so honestly. A clean null is still a finding.

## C5. Fig 5 (animated): Watching RL learn, base by base

From the probe files: a grid per format (rows = the 24 probe sequences sorted by length, columns = output positions; green = correct, red = wrong, grey = length mismatch / missing), one frame per probe checkpoint (step 0, 10, …, 100). Put the three formats side by side in each frame.
Export `figures/fig5_learning.gif` (≤ 8 MB, ~2 fps, hold the last frame for 2 s) and an `.mp4`. This one is for LinkedIn/X.

## C6. Extras

- **Fig 6: Before/after examples.** 3 hand-picked revcomp examples (one per bucket): target vs pre-RL vs post-RL output, with wrong bases highlighted red in a monospace layout. Pick examples that are typical, not cherry-picked wins; state how you picked them.
- **Fig 7: Compression vs accuracy.** One dot per eval prompt (post-RL), x = tokens-per-base of the input, y = mean exact-match over 4 samples, coloured by format, with a logistic fit per format.
- **Social card:** `figures/social_card.png`, 1200×675. Fig 1 on the left, the single most striking number on the right (e.g. "+23 pts from changing the input format, +4 pts from 100 steps of RL"), a one-line question as title, and a footer with the name and repo link placeholder.
- **Interactive page `site/index.html`:** single self-contained file (Plotly from CDN, data inlined as JSON). Sections: the question → Fig 1 → interactive learning curves (toggle formats) → length cliff → error-position analysis → an **example browser** (dropdown to pick an eval prompt and see target vs pre/post completions per format with red-highlighted errors) → setup and limitations → links. Must read well on mobile and support dark mode. It should drop straight into a GitHub Pages site as `/projects/dna-grpo/`.

## C7. Figure QA (do before showing the human)

For each figure: open the PNG and look at it. Check that labels aren't clipped, colours are consistent with C0, every number in the title matches the data, CIs are present, n is stated, and it's readable at 50% zoom. Fix, then list the figures with one line each in `notes/progress.md`.

---

# Part D: FINDINGS.md template

```markdown
# Does RL fix what the tokenizer breaks? (DNA edition)

**TL;DR (3 bullets, numbers included)**

## Setup
Model, tasks, formats, data source (FineDNA rl_mix or procedural, and why), GRPO config, compute (Kaggle T4 ×N, wall time).

## Results
1. Baseline: how the format alone changes accuracy (Fig 3, dashed).
2. What RL changes, per format (Fig 2, Fig 3).
3. Where errors live: boundary vs interior (Fig 4).
4. Learning dynamics (Fig 5, zero-std groups, length).

## What this means for DNA LMs like Carbon-X
2–4 sentences. Careful: we tested a 0.6B vanilla model with a simulated 6-mer view, not Carbon-X itself.

## Limitations
Single seed (say so), 100 steps, small model, LoRA, two synthetic tasks, no thinking mode.

## Next steps
e.g. the same probe on Carbon-X directly; a tool-augmented policy; partial-credit reward vs exact-match.

## Reproduce
Commands, versions, kernel ids.
```

Also draft `notes/post_draft.md`: a 5-sentence LinkedIn/X post (hook question → one-line method → one striking number → the GIF → link), in plain first-person voice, no hype words.

---

## Definition of done

- [ ] `pytest` passes
- [ ] smoke + main kernels complete; outputs in `outputs/`
- [ ] Figs 1–5 (+ extras) in `figures/` as PNG + SVG, GIF ≤ 8 MB, QA'd
- [ ] `site/index.html` works offline except for the CDN script, and on mobile
- [ ] `FINDINGS.md` filled with real numbers and honest limitations
- [ ] `notes/progress.md` has kernel ids, versions, wall times
- [ ] No credential anywhere in the repo (`git grep -i KGAT` returns nothing)
