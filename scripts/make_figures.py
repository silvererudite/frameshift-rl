"""All figures from outputs/ (CPU). Writes figures/fig<N>_<slug>.{png,svg}, the GIF/MP4,
the social card, and figures/summary.json (every number quoted in titles/FINDINGS).

usage: python scripts/make_figures.py
"""
import glob
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dna_grpo import FORMATS, dense_reward, format_seq, revcomp  # noqa: E402

import os
OUT = pathlib.Path(os.environ.get("OUTPUTS_DIR", ROOT / "outputs"))
FIG = pathlib.Path(os.environ.get("FIGURES_DIR", ROOT / "figures"))
FIG.mkdir(exist_ok=True)
COL = {"raw": "#E69F00", "perbase": "#0072B2", "kmer6": "#009E73"}
TINT = {"raw": ("#E69F00", "#F7D58C"), "perbase": ("#0072B2", "#8FC1E3"), "kmer6": ("#009E73", "#8ED6BF")}
LABEL = {"raw": "raw (BPE)", "perbase": "per-base", "kmer6": "6-mer"}
BUCKETS = ["12-30", "31-60", "61-90"]
INK, INK2, GRID = "#1f2328", "#59636e", "#e6e8eb"
RNG = np.random.default_rng(0)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "svg.fonttype": "none",
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})


RUN = os.environ.get("RUN", "dense")  # which RL run supplies post-RL results: "dense" or "exact"


def find(pattern, run=None):
    """Output files of one run. Dense-reward kernels write under results/dense/; the exact-match run does not.
    The pre-RL baseline always comes from the exact run (same base model, same eval settings)."""
    run = run or RUN
    hits = sorted(glob.glob(str(OUT / "**" / pattern), recursive=True))
    return [h for h in hits if ("/dense/" in h) == (run == "dense")]


def jl(path):
    with open(path) as f:
        return [json.loads(x) for x in f]


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print("saved", name)


def suptitle(fig, title, sub, y=1.02):
    fig.suptitle(title, x=0.01, y=y, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, y - 0.045, sub, ha="left", va="top", fontsize=10.5, color=INK2)


def boot_ci(df, col="exact", by="id", n=1000):
    """Mean of `col` with a 95% bootstrap CI, resampling prompts (samples within a prompt stay together)."""
    per = df.groupby(by)[col].mean().to_numpy(dtype=float)
    if len(per) == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(per), (n, len(per)))
    bs = per[idx].mean(1)
    return per.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ----------------------------------------------------------------------------- load
EV = {}
for phase in ("pre", "post"):
    for fmt in FORMATS:
        p = find(f"eval_{phase}_{fmt}.jsonl", run="exact" if phase == "pre" else None)
        if p:
            d = pd.DataFrame(jl(p[0]))
            d["phase"] = phase
            d["pc"] = [dense_reward(c, t, k) for c, t, k in zip(d.completion, d.target, d.task)]
            EV[(phase, fmt)] = d
ev = pd.concat(EV.values(), ignore_index=True) if EV else pd.DataFrame()
TOK = {(r["id"], r["format"]): r for r in jl(OUT / "tokenizer_stats.jsonl")}
LOGS = {f: json.load(open(find(f"train_log_{f}.json")[0])) for f in FORMATS if find(f"train_log_{f}.json")}
RW = {f: jl(find(f"rewards_{f}.jsonl")[0]) for f in FORMATS if find(f"rewards_{f}.jsonl")}
RW_EXACT = {f: jl(find(f"rewards_{f}.jsonl", "exact")[0]) for f in FORMATS if find(f"rewards_{f}.jsonl", "exact")}
EV_EXACT_POST = {f: pd.DataFrame(jl(find(f"eval_post_{f}.jsonl", "exact")[0])) for f in FORMATS
                 if find(f"eval_post_{f}.jsonl", "exact")}
STEPS = {f: LOGS[f]["steps"] for f in LOGS}
N_SAMPLES = int(ev["sample"].max() + 1) if len(ev) else 4
N_PROMPTS = ev["id"].nunique() if len(ev) else 240
steps_txt = "/".join(sorted({str(s) for s in STEPS.values()})) or "100"
SUB = f"Qwen3-0.6B + LoRA · GRPO (TRL) · {steps_txt} steps · n={N_PROMPTS} eval prompts × {N_SAMPLES} samples · 95% bootstrap CIs"
S = {"steps": STEPS, "n_prompts": int(N_PROMPTS), "n_samples": N_SAMPLES}


def acc(phase, fmt, task="revcomp", bucket=None, col="exact"):
    d = EV.get((phase, fmt))
    if d is None:
        return (np.nan,) * 3
    d = d[d.task == task]
    if bucket:
        d = d[d.bucket == bucket]
    return boot_ci(d, col)


for phase in ("pre", "post"):
    for fmt in FORMATS:
        for task in ("revcomp", "gc_count"):
            S[f"{task}_{phase}_{fmt}"] = acc(phase, fmt, task)[0]
            for b in BUCKETS:
                S[f"{task}_{phase}_{fmt}_{b}"] = acc(phase, fmt, task, b)[0]


for fmt, d in EV_EXACT_POST.items():
    for task in ("revcomp", "gc_count"):
        S[f"{task}_postexact_{fmt}"] = boot_ci(d[d.task == task])[0]
for (phase, fmt), d in EV.items():
    rc = d[d.task == "revcomp"]
    S[f"revcomp_hamming_{phase}_{fmt}"] = boot_ci(rc.assign(h=rc.hamming_sim.fillna(0.0)), "h")[0]
    S[f"revcomp_lenok_{phase}_{fmt}"] = 1 - rc.len_mismatch.fillna(True).astype(float).mean()


def pts(x):
    return f"{100 * x:.0f}"


# ----------------------------------------------------------------------------- Fig 1
def fig1():
    rec = min((r for r in TOK.values() if r["task"] == "revcomp" and r["id"].startswith("ev")),
              key=lambda r: (abs(r["length"] - 40), r["id"]))
    from dna_grpo import make_splits
    _, ev_items, _ = make_splits()
    seq = next(it["seq"] for it in ev_items if it["id"] == rec["id"])
    L = len(seq)
    tgt = revcomp(seq)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(-9.5, L + 4.5)
    ax.set_ylim(-4.6, 3.9)
    ax.axis("off")
    ax.grid(False)
    w = 0.92
    for row, fmt in enumerate(FORMATS):
        y = 3 - row * 1.25
        bt = TOK[(rec["id"], fmt)]["base_token"]
        ntok = len(set(bt))
        for i, b in enumerate(seq):
            # one block per token: alternate two tints; neighbouring bases of the same token share a block
            same_prev = i > 0 and bt[i - 1] == bt[i]
            same_next = i < L - 1 and bt[i + 1] == bt[i]
            k = sorted(set(bt)).index(bt[i]) % 2
            x0 = i - (0.5 if same_prev else w / 2)
            x1 = i + (0.5 if same_next else w / 2)
            ax.add_patch(Rectangle((x0, y - 0.42), x1 - x0, 0.84, color=TINT[fmt][k], lw=0))
            ax.text(i, y, b, ha="center", va="center", family="DejaVu Sans Mono", fontsize=10.5,
                    color="white" if k == 0 else INK, fontweight="bold")
        if fmt == "kmer6":
            for g in range(6, L, 6):
                ax.plot([g - 0.5, g - 0.5], [y - 0.62, y + 0.62], color=INK, lw=2.2, solid_capstyle="butt")
            ax.text(L + 0.3, y, "│ = 6-mer\n    boundary", fontsize=8.5, color=INK2, va="center")
        ax.text(-1.2, y + 0.12, LABEL[fmt], ha="right", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(-1.2, y - 0.25, f"{ntok} tokens · {ntok / L:.2f} tok/base", ha="right", va="center",
                fontsize=9.5, color=INK2)
    # target row
    y = -2.6
    for i, b in enumerate(tgt):
        ax.add_patch(Rectangle((i - w / 2, y - 0.42), w, 0.84, color="#eceef0", lw=0))
        ax.text(i, y, b, ha="center", va="center", family="DejaVu Sans Mono", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(-1.2, y + 0.12, "reverse\ncomplement", ha="right", va="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(-1.2, y - 0.62, "(the answer)", ha="right", va="center", fontsize=9.5, color=INK2)
    yb = 3 - 2 * 1.25 - 0.45
    for i in (0, 9, L // 2, L - 6):
        j = L - 1 - i
        ax.add_patch(FancyArrowPatch((j, yb), (i, y + 0.45), arrowstyle="-|>", mutation_scale=11,
                                     color=INK2, lw=1.1, connectionstyle="arc3,rad=0.0", alpha=0.85))
    ax.text(L / 2, -3.75, f"output base i = complement of input base L−1−i  (L = {L}): every answer base reaches "
                          "across the sequence, and across token boundaries", ha="center", fontsize=10.5, color=INK2)
    suptitle(fig, "Same DNA, three ways to cut it: the answer needs single bases, but BPE hands the model ~2-base chunks",
             f"One real eval sequence ({L} bp) through the Qwen3 tokenizer. Each coloured block is one input token.",
             y=1.0)
    save(fig, "fig1_how_the_model_sees_dna")
    S["fig1"] = dict(id=rec["id"], length=L, **{f"ntok_{f}": len(set(TOK[(rec["id"], f)]["base_token"])) for f in FORMATS})


# ----------------------------------------------------------------------------- Fig 2
def reward_by_step(fmt):
    rows = []
    for r in RW[fmt]:
        for t, v in zip(r["task"], r["reward"]):
            rows.append((r["call"], t, v))
    d = pd.DataFrame(rows, columns=["step", "task", "reward"])
    return d.groupby(["task", "step"]).reward.mean()


def zero_std_by_step(fmt, src=None):
    out = []
    for r in (src or RW)[fmt]:
        g = pd.DataFrame(dict(p=r["prompt_idx"], r=r["reward"], t=r["task"]))
        s = g.groupby("p").agg(std=("r", "std"), t=("t", "first"))
        out.append(dict(step=r["call"], all=(s["std"] == 0).mean(),
                        clen=np.mean(r["clen"])))
    return pd.DataFrame(out).set_index("step")


def fig2():
    if not RW:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw=dict(height_ratios=[1.4, 1]))
    for ci, task in enumerate(("revcomp", "gc_count")):
        ax = axes[0, ci]
        ends = []
        for fmt in FORMATS:
            if fmt not in RW:
                continue
            r = reward_by_step(fmt).loc[task]
            ax.plot(r.index, r.values, color=COL[fmt], alpha=0.22, lw=1)
            sm = r.rolling(10, min_periods=1).mean()
            ax.plot(sm.index, sm.values, color=COL[fmt], lw=2.4)
            ax.plot([sm.index[-1]], [sm.values[-1]], "o", color=COL[fmt], ms=6)
            ends.append((sm.values[-1], fmt, sm.index[-1]))
        # direct labels, nudged apart so they never overlap
        ends.sort()
        ly = -1
        for v, fmt, x in ends:
            ly = max(v, ly + 0.035)
            ax.text(x + 2.5, ly, LABEL[fmt], color=INK, fontsize=10, va="center", fontweight="bold")
        ax.set_title({"revcomp": "Reverse complement", "gc_count": "GC count"}[task], loc="left")
        ax.set_ylabel("training reward (partial credit)" if RUN == "dense" else "training reward (exact match)")
        ax.set_ylim(0, 0.6 if RUN == "dense" else 1.02)
        ax.set_xlim(0, max(STEPS.values()) * 1.18)
    for ci, (key, lab) in enumerate((("all", "groups with zero reward std\n(no learning signal)"),
                                     ("clen", "mean completion length (chars)"))):
        ax = axes[1, ci]
        for fmt in FORMATS:
            if fmt not in RW:
                continue
            z = zero_std_by_step(fmt)[key]
            ax.plot(z.index, z.values, color=COL[fmt], alpha=0.2, lw=1)
            ax.plot(z.index, z.rolling(10, min_periods=1).mean(), color=COL[fmt], lw=2, label=LABEL[fmt])
            if key == "all" and fmt in RW_EXACT and RUN == "dense":
                ze = zero_std_by_step(fmt, RW_EXACT)[key].rolling(10, min_periods=1).mean()
                ax.plot(ze.index, ze.values, color=COL[fmt], lw=1.6, ls=":",
                        label=f"{LABEL[fmt]} · exact-match reward")
        ax.set_ylabel(lab, fontsize=10)
        ax.set_xlabel("GRPO step")
        ax.set_xlim(0, max(STEPS.values()) * 1.18)
        if key == "all":
            ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best", fontsize=9.5)
    final = {f: reward_by_step(f).loc["revcomp"].tail(10).mean() for f in RW}
    first = {f: reward_by_step(f).loc["revcomp"].head(10).mean() for f in RW}
    S["fig2"] = dict(revcomp_first10=first, revcomp_last10=final)
    best = max(final, key=final.get)
    title = S.get("fig2_title") or (f"Revcomp reward climbs most for {LABEL[best]} "
                                    f"({pts(first[best])}% → {pts(final[best])}% over training); "
                                    f"GC count rises for every format")
    suptitle(fig, title, "Faint: per-step reward over 32 completions. Bold: 10-step rolling mean. "
                         "Bottom row: GRPO health signals. " + SUB.split(" · n=")[0])
    fig.tight_layout()
    save(fig, "fig2_learning_curves")


# ----------------------------------------------------------------------------- Fig 3
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    x = np.arange(3)
    for ai, task in enumerate(("revcomp", "gc_count")):
        ax = axes[ai]
        for k, fmt in enumerate(FORMATS):
            off = (k - 1) * 0.07
            pre = np.array([acc("pre", fmt, task, b, "pc") for b in BUCKETS])
            post = np.array([acc("post", fmt, task, b, "pc") for b in BUCKETS])
            S.setdefault("fig3_pc", {})[f"{task}|{fmt}"] = dict(pre=pre[:, 0].tolist(), post=post[:, 0].tolist())
            if np.isnan(post[:, 0]).all():
                continue
            ax.fill_between(x + off, pre[:, 0], post[:, 0], color=COL[fmt], alpha=0.12, lw=0)
            ax.errorbar(x + off, pre[:, 0], yerr=[pre[:, 0] - pre[:, 1], pre[:, 2] - pre[:, 0]], color=COL[fmt],
                        ls="--", lw=1.6, marker="o", mfc="white", ms=7, capsize=0, elinewidth=1)
            ax.errorbar(x + off, post[:, 0], yerr=[post[:, 0] - post[:, 1], post[:, 2] - post[:, 0]], color=COL[fmt],
                        ls="-", lw=2.4, marker="o", ms=7, capsize=0, elinewidth=1.2)
            ax.text(2.18, post[-1, 0], LABEL[fmt], color=INK, fontweight="bold", va="center", fontsize=10)
        ax.set_xticks(x, [f"{b} bp" for b in BUCKETS])
        ax.set_xlim(-0.25, 2.65)
        ax.set_ylim(0, 0.5)
        ax.set_yticks(np.linspace(0, 0.5, 6), [f"{v:.0%}" for v in np.linspace(0, 0.5, 6)])
        ax.set_xlabel("sequence length")
        ax.set_ylabel("partial-credit score (eval)")
        if task == "revcomp":
            ax.axhline(0.25, color=INK2, ls=":", lw=1.2)
            ax.text(-0.2, 0.258, "≈ score of copying the input (25%)", fontsize=9, color=INK2)
        ax.set_title({"revcomp": "Reverse complement", "gc_count": "GC count"}[task], loc="left")
    axes[1].plot([], [], "k--", marker="o", mfc="white", label="before RL")
    axes[1].plot([], [], "k-", marker="o", label="after RL")
    axes[1].legend(loc="upper left")
    S["fig3_note"] = "exact match is 0% for revcomp in every cell; partial-credit score plotted instead"
    suptitle(fig, S.get("fig3_title", "Accuracy falls with length; shaded band = what RL bought"), SUB)
    fig.tight_layout()
    save(fig, "fig3_length_cliff")


# ----------------------------------------------------------------------------- Fig 4
def position_errors(phase, fmt):
    d = EV.get((phase, fmt))
    if d is None:
        return pd.DataFrame()
    d = d[(d.task == "revcomp") & d.per_position_correct.notna()]
    rows = []
    for r in d.itertuples():
        t = TOK[(r.id, fmt)]
        L = r.length
        bt = t["base_token"]
        for i, ok in enumerate(r.per_position_correct):
            j = L - 1 - i
            # position of input base j within its own token
            tok_pos = j - bt.index(bt[j])
            rows.append((r.id, r.sample, j, t["is_boundary"][j], j % 6, tok_pos, not ok))
    return pd.DataFrame(rows, columns=["id", "sample", "j", "boundary", "pos6", "tokpos", "err"])


def fig4():
    PE = {(p, f): position_errors(p, f) for p in ("pre", "post") for f in FORMATS}
    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 0.8], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    xs, labs = [], []
    S["fig4"] = {}
    for k, fmt in enumerate(FORMATS):
        for m, (bnd, name) in enumerate(((True, "boundary"), (False, "interior"))):
            for q, phase in enumerate(("pre", "post")):
                d = PE[(phase, fmt)]
                if d.empty:
                    continue
                d = d[d.boundary == bnd]
                if len(d) == 0:
                    continue
                d = d.assign(key=d.id + "_" + d["sample"].astype(str))
                mu, lo, hi = boot_ci(d, "err", by="id")
                xpos = k * 3 + m * 1.1 + q * 0.42
                ax.bar(xpos, mu, width=0.38, color=COL[fmt] if phase == "post" else "white",
                       edgecolor=COL[fmt], lw=1.6, hatch=None if bnd else "///")
                ax.errorbar(xpos, mu, yerr=[[mu - lo], [hi - mu]], color=INK, lw=1, capsize=0)
                S["fig4"][f"{fmt}_{name}_{phase}"] = dict(err=mu, n_positions=int(len(d)))
            xs.append(k * 3 + m * 1.1 + 0.21)
            labs.append({"boundary": "edge", "interior": "inside"}[name])
        ax.text(k * 3 + 0.76, -0.16, LABEL[fmt], transform=ax.get_xaxis_transform(), ha="center",
                fontweight="bold", color=INK)
    ax.set_xticks(xs, labs, fontsize=9.5)
    ax.set_ylabel("per-base error rate\n(length-correct revcomp answers)")
    ax.set_title("A · Error rate by where the input base sits", loc="left", fontsize=12)
    ax.bar([0], [0], color="white", edgecolor=INK2, label="before RL")
    ax.bar([0], [0], color=INK2, edgecolor=INK2, label="after RL")
    ax.bar([0], [0], color="white", edgecolor=INK2, hatch="///", label="token-interior base")
    ax.axhline(0.75, color=INK2, ls="--", lw=1)
    ax.text(8.6, 0.765, "chance (random base) = 75%", ha="right", fontsize=9, color=INK2)
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.legend(loc="upper left", fontsize=9, ncol=3, columnspacing=1)
    ax.text(0.5, -0.24, "edge = first/last base of its token · per-base has no interior bases (control)",
            transform=ax.transAxes, fontsize=9, color=INK2, ha="center")

    ax = fig.add_subplot(gs[1])
    for phase, ls, mfc in (("pre", "--", "white"), ("post", "-", COL["kmer6"])):
        d = PE[("pre" if phase == "pre" else "post", "kmer6")]
        if d.empty:
            continue
        vals = np.array([boot_ci(d[d.pos6 == p], "err") for p in range(6)])
        ax.errorbar(range(6), vals[:, 0], yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], color=COL["kmer6"],
                    ls=ls, marker="o", mfc=mfc, ms=7, lw=2, label="before RL" if phase == "pre" else "after RL")
        S["fig4"][f"kmer6_pos6_{phase}"] = vals[:, 0].tolist()
    ax.set_xticks(range(6), [str(i) for i in range(6)])
    ax.set_xlabel("position of input base inside its 6-mer")
    ax.set_ylabel("per-base error rate")
    ax.set_title("B · 6-mer: error by k-mer position", loc="left", fontsize=12)
    ax.axhline(0.75, color=INK2, ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.legend(fontsize=9.5, loc="lower right")

    ax = fig.add_subplot(gs[2])
    for k, fmt in enumerate(FORMATS):
        for q, phase in enumerate(("pre", "post")):
            d = EV.get((phase, fmt))
            if d is None:
                continue
            d = d[(d.task == "revcomp")].assign(lm=lambda z: z.len_mismatch.fillna(True).astype(float))
            mu, lo, hi = boot_ci(d, "lm")
            ax.bar(k + (q - 0.5) * 0.36, mu, width=0.34, color=COL[fmt] if phase == "post" else "white",
                   edgecolor=COL[fmt], lw=1.6)
            ax.errorbar(k + (q - 0.5) * 0.36, mu, yerr=[[mu - lo], [hi - mu]], color=INK, lw=1)
            S["fig4"][f"{fmt}_lenmismatch_{phase}"] = mu
    ax.set_xticks(range(3), [LABEL[f] for f in FORMATS], fontsize=9.5)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel("wrong length or no answer")
    ax.set_title("C · Length errors", loc="left", fontsize=12)
    suptitle(fig, S.get("fig4_title", "Where revcomp errors happen relative to token boundaries"),
             "Each output base i is mapped to input base L−1−i; boundary = first/last base of its input token. "
             + SUB.split(" · n=")[0], y=1.05)
    save(fig, "fig4_error_positions")


# ----------------------------------------------------------------------------- Fig 5 (animated)
def fig5():
    import imageio.v2 as imageio
    probes = {}
    for f in FORMATS:
        for p in find(f"probe_{f}_step*.jsonl"):
            step = int(pathlib.Path(p).stem.split("step")[1])
            probes[(f, step)] = sorted(jl(p), key=lambda r: (r["length"], r["id"]))
    if not probes:
        return
    steps = sorted({s for (_, s) in probes})
    frames = []
    cmap = {"ok": (0.0, 0.62, 0.45), "bad": (0.84, 0.24, 0.18), "miss": (0.78, 0.79, 0.80), "none": (1, 1, 1)}
    for step in steps:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        for ax, f in zip(axes, FORMATS):
            # use the latest probe at or before this step (formats may stop at different steps)
            avail = [s for (ff, s) in probes if ff == f and s <= step]
            rows = probes[(f, max(avail))] if avail else []
            grid = np.ones((24, 90, 3))
            for ri, r in enumerate(rows):
                L = r["length"]
                ppc = r.get("per_position_correct")
                for c in range(L):
                    grid[ri, c] = cmap["miss"] if ppc is None else (cmap["ok"] if ppc[c] else cmap["bad"])
            ax.imshow(grid, aspect="auto", interpolation="nearest")
            ax.grid(False)
            ax.set_yticks([])
            ax.set_xticks([0, 29, 59, 89], ["1", "30", "60", "90"])
            for s in ax.spines.values():
                s.set_visible(False)
            nex = sum(r["exact"] for r in rows)
            ax.set_title(f"{LABEL[f]}   {nex}/24 exact", loc="left", fontsize=12, color=INK)
            ax.set_xlabel("output position")
        axes[0].set_ylabel("24 probe sequences (short → long)")
        fig.suptitle(f"Partial-credit GRPO, step {step}: right lengths appear, right bases do not (greedy, 24 probe prompts)", x=0.01, ha="left",
                     fontsize=14, fontweight="bold", color=INK)
        fig.text(0.01, 0.885, "green = correct base · red = wrong base · grey = wrong length or no answer · ~25% green is what a copy of the input scores",
                 fontsize=10, color=INK2)
        fig.subplots_adjust(top=0.80, bottom=0.13, left=0.04, right=0.99, wspace=0.08)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if step == steps[-1]:
            save(fig, "fig5_learning_final")
        else:
            plt.close(fig)
    dur = [500] * (len(frames) - 1) + [2500]
    imageio.mimsave(FIG / "fig5_learning.gif", frames, duration=dur, loop=0)
    try:
        imageio.mimsave(FIG / "fig5_learning.mp4", [fr for fr, d in zip(frames, dur) for _ in range(d // 500)],
                        fps=2, macro_block_size=1)
    except Exception as e:
        print("mp4 failed:", e)
    print("saved fig5 gif", (FIG / "fig5_learning.gif").stat().st_size // 1024, "KB")


# ----------------------------------------------------------------------------- Fig 6
def fig6():
    if not EV:
        return
    pre_pb = EV[("pre", "raw")]
    picks = []
    for b in BUCKETS:
        # rule: the first revcomp eval prompt (by id) in each bucket -- sequences are random, so this is unbiased
        picks.append(sorted(pre_pb[(pre_pb.task == "revcomp") & (pre_pb.bucket == b)].id.unique())[0])
    lines = []
    for pid in picks:
        base = EV[("pre", "raw")]
        r0 = base[base.id == pid].iloc[0]
        lines.append(("title", f"{pid} · {r0.length} bp", None))
        lines.append(("target", "target", r0.target))
        for f in FORMATS:
            for phase in ("pre", "post"):
                d = EV.get((phase, f))
                if d is None:
                    continue
                r = d[(d.id == pid) & (d["sample"] == 0)].iloc[0]
                lines.append((f, f"{LABEL[f]} {'before' if phase == 'pre' else 'after'}",
                              r.answer if isinstance(r.answer, str) else "∅ (no <answer>)"))
        lines.append(("gap", "", None))
    fig, ax = plt.subplots(figsize=(14, 0.27 * len(lines) + 0.6))
    ax.axis("off")
    ax.set_xlim(0, 112)
    ax.set_ylim(len(lines), -0.5)
    tgt = None
    cw = 1.0
    for y, (kind, lab, s) in enumerate(lines):
        if kind == "title":
            ax.text(0, y, lab, fontsize=11, fontweight="bold", color=INK, va="center")
            continue
        if kind == "gap":
            continue
        if kind == "target":
            tgt = s
        ax.text(19, y, lab, ha="right", va="center", fontsize=9.5, color=COL.get(kind, INK),
                fontweight="bold")
        for i, ch in enumerate(s[:92]):
            bad = kind != "target" and (i >= len(tgt) or ch != tgt[i])
            if bad:
                ax.add_patch(Rectangle((20 + i * cw - 0.45, y - 0.42), 0.9, 0.84, color="#f6c9c4", lw=0))
            ax.text(20 + i * cw, y, ch, ha="center", va="center", family="DejaVu Sans Mono", fontsize=8.6,
                    color="#b42318" if bad else INK)
    suptitle(fig, "Before vs after RL on three typical reverse-complement prompts",
             "Sample 0 of 4 for each model. Red = base differs from the target at that position. Picked by rule: "
             "the first eval prompt (by id) in each length bucket, not chosen by outcome.", y=1.0)
    save(fig, "fig6_examples")


# ----------------------------------------------------------------------------- Fig 7
def fig7():
    d = EV.get(("post", "raw"))
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for fmt in FORMATS:
        e = EV.get(("post", fmt))
        if e is None:
            continue
        e = e[e.task == "revcomp"].groupby("id").agg(acc=("pc", "mean"), L=("length", "first")).reset_index()
        e["tpb"] = [TOK[(i, fmt)]["tokens_per_base"] for i in e.id]
        jit = RNG.normal(0, 0.006, len(e)) if fmt == "perbase" else 0
        ax.scatter(e.tpb + jit, e.acc + RNG.normal(0, 0.012, len(e)), s=22, color=COL[fmt], alpha=0.55,
                   edgecolor="white", lw=0.5, label=LABEL[fmt])
        ax.errorbar(e.tpb.mean(), e.acc.mean(), xerr=e.tpb.std(), fmt="D", color=COL[fmt], ms=10, mec=INK, lw=1.4)
    ax.set_xlabel("input tokens per base (lower = more compressed)")
    ax.set_ylabel("post-RL partial-credit score\n(mean of 4 samples)")
    ax.set_ylim(-0.02, 0.6)
    ax.legend(loc="center left")
    suptitle(fig, S.get("fig7_title", "More compression, less accuracy: revcomp vs tokens per base"),
             "One dot per revcomp eval prompt (n=120 per format), jittered. Diamond = format mean ± SD of tokens/base.")
    save(fig, "fig7_compression_vs_accuracy")


# ----------------------------------------------------------------------------- Fig 8 (exact-match null)
def fig8():
    if not RW_EXACT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    S["fig8"] = {}
    for ax, task in zip(axes, ("revcomp", "gc_count")):
        for k, fmt in enumerate(FORMATS):
            if fmt not in RW_EXACT:
                continue
            rs = [v for r in RW_EXACT[fmt] for t, v in zip(r["task"], r["reward"]) if t == task]
            hit = float(np.mean(np.array(rs) > 0))
            # groups (8 completions of one prompt) where at least one completion was right
            groups = []
            for r in RW_EXACT[fmt]:
                g = pd.DataFrame(dict(p=r["prompt_idx"], t=r["task"], r=r["reward"]))
                groups.extend(g[g.t == task].groupby("p").r.max())
            ghit = float(np.mean(np.array(groups) > 0))
            ax.bar(k - 0.18, hit, 0.34, color=COL[fmt])
            ax.bar(k + 0.18, ghit, 0.34, color="white", edgecolor=COL[fmt], lw=1.6)
            ax.text(k - 0.18, hit + 0.004, f"{sum(np.array(rs) > 0)}/{len(rs)}", ha="center", va="bottom", fontsize=9.5, color=INK)
            ax.text(k + 0.18, ghit + 0.004, f"{sum(np.array(groups) > 0)}/{len(groups)}", ha="center", va="bottom",
                    fontsize=9.5, color=INK)
            S["fig8"][f"{fmt}_{task}"] = dict(rollouts_correct=int(sum(np.array(rs) > 0)), rollouts=len(rs),
                                              groups_with_signal=int(sum(np.array(groups) > 0)), groups=len(groups))
        ax.set_xticks(range(3), [LABEL[f] for f in FORMATS])
        ax.set_title({"revcomp": "Reverse complement", "gc_count": "GC count"}[task], loc="left")
        ax.set_ylabel("fraction")
        ax.set_ylim(0, max(0.1, ax.get_ylim()[1]))
    axes[0].bar([0], [0], color=INK2, label="training rollouts with reward 1")
    axes[0].bar([0], [0], color="white", edgecolor=INK2, label="prompt groups with ≥1 correct (any gradient)")
    axes[0].legend(loc="upper left", fontsize=9.5)
    suptitle(fig, S.get("fig8_title", "With an exact-match reward, GRPO almost never sees a correct answer"),
             f"Exact-match (1/0) GRPO run, {steps_txt} steps × 32 rollouts per format. A group of 8 with no correct answer has zero "
             "advantage and contributes no gradient.")
    fig.tight_layout()
    save(fig, "fig8_sparse_reward_null")


# ----------------------------------------------------------------------------- Fig 9 (what the answers are)
CATS = [("exact", "correct reverse complement", "#1f2328"), ("copy", "verbatim copy of the input", "#CC79A7"),
        ("other", "right length, other content", "#9aa4b2"), ("len", "wrong length", "#d9dde2"),
        ("none", "no <answer> tag", "#ffffff")]


def answer_kinds(d):
    rc = d[d.task == "revcomp"]
    a = rc.answer.fillna("")
    exact = a == rc.target
    copy = (a == rc.seq) & ~exact
    none = rc.answer.isna()
    other = (a.str.len() == rc.length) & ~exact & ~copy & ~none
    wl = ~exact & ~copy & ~none & ~other
    return {"exact": exact.mean(), "copy": copy.mean(), "other": other.mean(), "len": wl.mean(), "none": none.mean()}


def fig9():
    rows = []
    for fmt in FORMATS:
        for tag, d in (("before RL", EV.get(("pre", fmt))), ("after exact-match RL", EV_EXACT_POST.get(fmt)),
                       ("after partial-credit RL", EV.get(("post", fmt)) if RUN == "dense" else None)):
            if d is not None:
                rows.append((fmt, tag, answer_kinds(d)))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(13, 0.52 * len(rows) + 2.2))
    ax.grid(False)
    ys = []
    y = 0
    prev = None
    for fmt, tag, k in rows:
        if prev is not None and fmt != prev:
            y += 0.6
        left = 0
        for key, lab, c in CATS:
            ax.barh(y, k[key], left=left, color=c, edgecolor="#9aa4b2" if key == "none" else "white", lw=1,
                    hatch="///" if key == "none" else None, height=0.78)
            if k[key] >= 0.06:
                ax.text(left + k[key] / 2, y, f"{k[key]:.0%}", ha="center", va="center", fontsize=10,
                        color="white" if key in ("exact", "copy") else INK, fontweight="bold")
            left += k[key]
        ax.text(-0.01, y, tag, ha="right", va="center", fontsize=10.5, color=INK)
        if fmt != prev:
            ax.text(-0.215, y + (0.0), LABEL[fmt], ha="right", va="center", fontsize=11.5, fontweight="bold", color=COL[fmt])
        S.setdefault("fig9", {})[f"{fmt}|{tag}"] = {kk: float(v) for kk, v in k.items()}
        ys.append(y)
        prev = fmt
        y += 1
    ax.set_ylim(y - 0.4, -1.4)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.spines["left"].set_visible(False)
    for i, (key, lab, c) in enumerate(CATS):
        ax.barh([0], [0], color=c, edgecolor="#9aa4b2", hatch="///" if key == "none" else None, label=lab)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=9.5, handlelength=1.2, columnspacing=1.2)
    ax.set_xlabel(f"share of reverse-complement answers (120 eval prompts × {N_SAMPLES} samples per row)")
    suptitle(fig, S.get("fig9_title", "Partial-credit RL taught the model to copy the input"),
             "Qwen3-0.6B + LoRA · GRPO (TRL) · same eval prompts in every row. A copy always has the right length and "
             "matches the reverse complement at ~25% of positions by chance.", y=1.08)
    save(fig, "fig9_what_the_answers_are")


# ----------------------------------------------------------------------------- Fig 10 (copy takes over)
def probe_series(run):
    out = {}
    for f in FORMATS:
        for p in find(f"probe_{f}_step*.jsonl", run=run):
            step = int(pathlib.Path(p).stem.split("step")[1])
            rows = jl(p)
            out.setdefault(f, {})[step] = dict(
                copy=np.mean([r.get("answer") == r["seq"] for r in rows]),
                lenok=np.mean([r.get("len_mismatch") is False for r in rows]))
    return out


def fig10():
    dense, exact = probe_series("dense"), probe_series("exact")
    if not dense:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, lab in ((axes[0], "copy", "answers that are a verbatim copy of the input"),
                         (axes[1], "lenok", "answers with the right length")):
        ends = []
        for f in FORMATS:
            for src, ls, lw in ((exact, ":", 1.6), (dense, "-", 2.6)):
                if f not in src:
                    continue
                st = sorted(src[f])
                ax.plot(st, [src[f][k][key] for k in st], ls=ls, lw=lw, color=COL[f], marker="o" if ls == "-" else None, ms=5)
            st = sorted(dense[f])
            ends.append((dense[f][st[-1]][key], f, st[-1]))
        ends.sort()
        ly = -1
        for v, f, x in ends:
            ly = max(v, ly + 0.06)
            ax.text(x + 3, ly, LABEL[f], color=INK, fontweight="bold", va="center", fontsize=10)
        ax.set_xlim(-2, 122)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
        ax.set_xlabel("GRPO step")
        ax.set_title(lab, loc="left", fontsize=12)
    axes[0].plot([], [], "k-", marker="o", label="partial-credit reward")
    axes[0].plot([], [], "k:", label="exact-match reward")
    axes[0].legend(loc="upper left")
    S["fig10"] = {f: {k: dense[f][max(dense[f])][k] for k in ("copy", "lenok")} for f in dense}
    suptitle(fig, S.get("fig10_title", "Copying takes over within the first ~30 steps of partial-credit training"),
             "24 fixed reverse-complement probe prompts, greedy decoding, checked every 10 GRPO steps "
             "(small n: each point is out of 24).")
    fig.tight_layout()
    save(fig, "fig10_copying_takes_over")


# ----------------------------------------------------------------------------- Figs 11-12 (Carbon-X)
CARBON = {}
for _f in ("raw", "perbase"):
    _p = sorted(glob.glob(str(OUT / "**" / "carbon" / f"eval_carbon_{_f}.jsonl"), recursive=True))
    if _p:
        CARBON[_f] = pd.DataFrame(jl(_p[0]))
CLAB = {"raw": "Carbon-X · plain DNA", "perbase": "Carbon-X · spaced letters (re-tokenized to 6-mers)"}


def fig11():
    if not CARBON:
        return
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.55, wspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    ax.grid(False)
    rows = [("Qwen3-0.6B · plain DNA (BPE)", EV[("pre", "raw")]), ("Qwen3-0.6B · spaced letters", EV[("pre", "perbase")])]
    rows += [(CLAB[f], CARBON[f]) for f in ("raw", "perbase") if f in CARBON]
    S["fig11"] = {}
    for y, (lab, d) in enumerate(rows):
        k = answer_kinds(d)
        S["fig11"][lab] = {kk: float(v) for kk, v in k.items()}
        left = 0
        for key, _, c in CATS:
            ax.barh(y, k[key], left=left, color=c, edgecolor="#9aa4b2" if key == "none" else "white",
                    hatch="///" if key == "none" else None, height=0.7)
            if k[key] >= 0.06:
                ax.text(left + k[key] / 2, y, f"{k[key]:.0%}", ha="center", va="center", fontsize=10, fontweight="bold",
                        color="white" if key in ("exact", "copy") else INK)
            left += k[key]
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("share of reverse-complement answers (before any of our RL)")
    for key, lab, c in CATS:
        ax.barh([0], [0], color=c, edgecolor="#9aa4b2", hatch="///" if key == "none" else None, label=lab)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=9)
    ax.set_title("A · What each base model answers", loc="left", fontsize=12)
    x = np.arange(3)
    for i, task in enumerate(("revcomp", "gc_count")):
        ax = fig.add_subplot(gs[i, 1])
        for lab, d, c, ls in (("Qwen3-0.6B", EV[("pre", "raw")], INK2, "--"),
                              ("Carbon-X (6-mer)", CARBON.get("raw"), "#CC79A7", "-"),
                              ("Carbon-X (letters)", CARBON.get("perbase"), "#CC79A7", ":")):
            if d is None:
                continue
            v = np.array([boot_ci(d[(d.task == task) & (d.bucket == b)]) for b in BUCKETS])
            ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=c, ls=ls, marker="o", ms=6,
                        lw=2, label=lab, elinewidth=1)
            S["fig11"][f"{lab}|{task}"] = v[:, 0].tolist()
        ax.set_xticks(x, [f"{b} bp" for b in BUCKETS])
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_title(("B · " if i == 0 else "C · ") + {"revcomp": "Reverse complement", "gc_count": "GC count"}[task]
                     + ": exact match", loc="left", fontsize=12)
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")
    suptitle(fig, S.get("fig11_title", "Does a DNA-trained model have the skill our small model lacked?"),
             f"Same 240 eval prompts × 4 samples, T=0.7, top-p 0.8. Carbon-X (Qwen3-14B DNA fine-tune) in 4-bit NF4 on a T4. "
             "95% bootstrap CIs.", y=1.04)
    save(fig, "fig11_carbon_x_vs_qwen")


def fig12():
    d = CARBON.get("raw")
    if d is None:
        return
    d = d[(d.task == "revcomp") & d.per_position_correct.notna()]
    if len(d) < 10:
        S["fig12"] = dict(n_length_correct=int(len(d)))
        return
    rows = []
    for r in d.itertuples():
        L = r.length
        for i, ok in enumerate(r.per_position_correct):
            j = L - 1 - i
            rows.append((r.id, j % 6, i / max(1, L - 1), not ok, (j % 6) in (0, 5)))
    e = pd.DataFrame(rows, columns=["id", "pos6", "relpos", "err", "edge"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    v = np.array([boot_ci(e[e.pos6 == p], "err") for p in range(6)])
    axes[0].errorbar(range(6), v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color="#CC79A7", marker="o", ms=8, lw=2.4)
    axes[0].set_xticks(range(6), ["0\n(edge)", "1", "2", "3", "4", "5\n(edge)"])
    axes[0].set_xlabel("position of the input base inside its 6-mer token")
    axes[0].set_ylabel("per-base error rate")
    axes[0].set_title("A · Error by position inside the 6-mer token", loc="left", fontsize=12)
    bins = np.linspace(0, 1, 11)
    e["bin"] = np.clip(np.digitize(e.relpos, bins) - 1, 0, 9)
    w = np.array([boot_ci(e[e.bin == b], "err") for b in range(10)])
    xc = (bins[:-1] + bins[1:]) / 2
    axes[1].errorbar(xc, w[:, 0], yerr=[w[:, 0] - w[:, 1], w[:, 2] - w[:, 0]], color="#CC79A7", marker="o", ms=6, lw=2)
    axes[1].set_xlabel("position in the answer (start → end)")
    axes[1].set_title("B · Error along the answer", loc="left", fontsize=12)
    for ax in axes:
        ax.axhline(0.75, color=INK2, ls="--", lw=1)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{t:.0%}" for t in np.linspace(0, 1, 6)])
    axes[1].text(1.0, 0.765, "chance", ha="right", fontsize=9, color=INK2)
    S["fig12"] = dict(n_length_correct=int(d.shape[0]), pos6=v[:, 0].tolist(), along=w[:, 0].tolist(),
                      edge=boot_ci(e[e.edge], "err")[0], interior=boot_ci(e[~e.edge], "err")[0])
    suptitle(fig, S.get("fig12_title", "Carbon-X reverse-complement errors vs its own 6-mer token boundaries"),
             f"Length-correct Carbon-X answers only (n = {d.shape[0]} samples). Output base i traced to input base L−1−i; "
             "6-mers are counted from the start of the sequence. 95% bootstrap CIs.")
    fig.tight_layout()
    save(fig, "fig12_carbon_x_error_positions")


# ----------------------------------------------------------------------------- Fig 13 (Carbon-X: the leftover 6-mer)
def fig13():
    if not CARBON:
        return
    d = pd.concat([CARBON[f].assign(run=f) for f in CARBON], ignore_index=True)
    d = d[d.task == "revcomp"].copy()
    d["rem"] = d.length % 6
    d["lenerr"] = d.len_mismatch.fillna(True).astype(float)
    d["key"] = d.id + "_" + d.run
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    S["fig13"] = {}
    for ax, col, lab in ((axes[0], "exact", "exact-match accuracy"), (axes[1], "lenerr", "wrong-length (or missing) answers")):
        v = np.array([boot_ci(d[d.rem == t], col, by="key") for t in range(6)])
        n = [d[d.rem == t].key.nunique() for t in range(6)]
        ax.bar(range(6), v[:, 0], color=["#9aa4b2"] + ["#CC79A7"] * 5, width=0.62)
        ax.errorbar(range(6), v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", color=INK, lw=1.2)
        for t in range(6):
            ax.text(t, v[t, 2] + 0.02, f"{v[t, 0]:.0%}", ha="center", fontsize=10, color=INK, fontweight="bold")
        ax.set_xticks(range(6), [f"{t}\n(n={n[t]})" for t in range(6)])
        ax.set_xlabel("letters left over in the final, incomplete 6-mer  (sequence length mod 6)")
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])
        ax.set_title(lab, loc="left", fontsize=12)
        S["fig13"][col] = v[:, 0].tolist()
    diff = (d.answer.str.len() - d.length).dropna()
    bad = diff[diff != 0]
    S["fig13"].update(n_len_errors=int(len(bad)), small_len_errors=float((bad.abs() <= 2).mean()),
                      mult6_len_errors=float((bad % 6 == 0).mean()))
    suptitle(fig, S.get("fig13_title", "Carbon-X stumbles on the leftover chunk: accuracy depends on sequence length mod 6"),
             "Carbon-X pads the final partial 6-mer with N (e.g. AANNNN). Reverse complement starts from that chunk. "
             "Both Carbon-X runs pooled (identical tokens), 95% bootstrap CIs over prompt × run.")
    fig.tight_layout()
    save(fig, "fig13_carbon_x_leftover_chunk")


# ----------------------------------------------------------------------------- social card
def social():
    import matplotlib.image as mpimg
    img = mpimg.imread(FIG / "fig1_how_the_model_sees_dna.png")
    fig = plt.figure(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.02, 0.12, 0.58, 0.66])
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.03, 0.90, "Does RL fix what the tokenizer breaks?", fontsize=28, fontweight="bold", color=INK)
    fig.text(0.03, 0.83, "GRPO on DNA reverse complement, three input segmentations, one small LM", fontsize=14, color=INK2)
    lines = S.get("social_lines", [])
    yy = 0.66
    for big, small in lines:
        fig.text(0.63, yy, big, fontsize=36, fontweight="bold", color=INK)
        fig.text(0.63, yy - 0.035, small, fontsize=13, color=INK2, va="top", linespacing=1.4)
        yy -= 0.30
    fig.text(0.03, 0.04, "Shamima Hossain · github.com/silvererudite/frameshift-rl", fontsize=12, color=INK2)
    fig.savefig(FIG / "social_card.png", dpi=100, facecolor="white")
    plt.close(fig)
    print("saved social_card")


if __name__ == "__main__":
    titles = ROOT / "figures" / "titles.json"
    if titles.exists():
        S.update(json.load(open(titles)))
    fig1()
    if EV:
        fig2()
        fig3()
        fig4()
        fig6()
        fig7()
    fig8()
    fig9()
    fig10()
    fig11()
    fig12()
    fig13()
    fig5()
    social()
    json.dump({k: v for k, v in S.items() if k not in ("social_lines",)}, open(FIG / "summary.json", "w"),
              indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
