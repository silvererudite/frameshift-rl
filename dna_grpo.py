"""DNA-GRPO: does RL fix what the tokenizer breaks?

Single-file library: data, input formats, rewards/metrics, eval, GRPO training.
Kaggle script kernels run one file, so `scripts/build_kernel.py` copies this file
into `kernels/<stage>/entry.py` and appends a footer calling `main(stage=...)`.
"""

import json
import os
import random
import re
import sys
import time

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
SEED = 1234
MODEL_ID = "Qwen/Qwen3-0.6B"
FORMATS = ["raw", "perbase", "kmer6"]
TASKS = ["revcomp", "gc_count"]
BUCKETS = [(12, 30), (31, 60), (61, 90)]
N_TRAIN = 2000
N_EVAL_PER_CELL = 40  # 2 tasks x 3 buckets x 40 = 240
N_PROBE_PER_BUCKET = 8

TASK_INSTRUCTION = {
    "revcomp": "Give the reverse complement of this sequence.",
    "gc_count": "Count the number of G and C bases in this sequence. Answer with a single integer.",
}
PROMPT_TEMPLATE = (
    "Here is a DNA sequence: {seq_formatted}\n"
    "Task: {task_instruction}\n"
    "Respond with only the answer inside <answer></answer>."
)
_COMP = str.maketrans("ACGT", "TGCA")


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def gc_count(seq: str) -> int:
    return sum(c in "GC" for c in seq)


def bucket_of(length: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= length <= hi:
            return f"{lo}-{hi}"
    raise ValueError(length)


def format_seq(seq: str, fmt: str) -> str:
    if fmt == "raw":
        return seq
    if fmt == "perbase":
        return " ".join(seq)
    if fmt == "kmer6":
        return " ".join(seq[i:i + 6] for i in range(0, len(seq), 6))
    raise ValueError(fmt)


def base_char_positions(seq: str, fmt: str) -> list[int]:
    """Character offset of each base inside format_seq(seq, fmt)."""
    s = format_seq(seq, fmt)
    return [i for i, c in enumerate(s) if c in "ACGT"]


def target_of(seq: str, task: str) -> str:
    return revcomp(seq) if task == "revcomp" else str(gc_count(seq))


def _rand_seq(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(length))


def make_splits(seed: int = SEED):
    """Underlying (task, seq) items; identical across formats."""
    rng = random.Random(seed)
    eval_items, probe_items, train_items = [], [], []
    k = 0
    for task in TASKS:
        for lo, hi in BUCKETS:
            for _ in range(N_EVAL_PER_CELL):
                L = rng.randint(lo, hi)
                eval_items.append(dict(id=f"ev{k:04d}", task=task, seq=_rand_seq(rng, L)))
                k += 1
    for j, (lo, hi) in enumerate(BUCKETS):
        for i in range(N_PROBE_PER_BUCKET):
            L = rng.randint(lo, hi)
            probe_items.append(dict(id=f"pr{j}{i}", task="revcomp", seq=_rand_seq(rng, L)))
    for i in range(N_TRAIN):
        L = rng.randint(12, 90)
        train_items.append(dict(id=f"tr{i:05d}", task=TASKS[i % 2], seq=_rand_seq(rng, L)))
    rng.shuffle(train_items)
    return train_items, eval_items, probe_items


def user_text(seq: str, task: str, fmt: str) -> str:
    return PROMPT_TEMPLATE.format(seq_formatted=format_seq(seq, fmt), task_instruction=TASK_INSTRUCTION[task])


def render(items, fmt: str, tokenizer=None):
    out = []
    for it in items:
        text = user_text(it["seq"], it["task"], fmt)
        prompt = text
        if tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        out.append(dict(id=it["id"], task=it["task"], format=fmt, length=len(it["seq"]),
                        bucket=bucket_of(len(it["seq"])), seq=it["seq"],
                        target=target_of(it["seq"], it["task"]), prompt=prompt))
    return out


# ----------------------------------------------------------------------------
# Reward + metrics
# ----------------------------------------------------------------------------
_ANS = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def parse_answer(completion: str):
    m = _ANS.findall(completion or "")
    if not m:
        return None
    return re.sub(r"\s+", "", m[-1]).upper()


def score(completion: str, target: str, task: str) -> dict:
    ans = parse_answer(completion)
    tgt = re.sub(r"\s+", "", target).upper()
    r = dict(answer=ans, exact=False, parse_fail=ans is None, len_mismatch=None,
             per_position_correct=None, hamming_sim=None)
    if ans is None:
        return r
    r["exact"] = ans == tgt
    if task == "revcomp":
        r["len_mismatch"] = len(ans) != len(tgt)
        if not r["len_mismatch"]:
            ppc = [a == b for a, b in zip(ans, tgt)]
            r["per_position_correct"] = ppc
            r["hamming_sim"] = sum(ppc) / len(ppc)
    return r


def reward(completion: str, target: str, task: str) -> float:
    return 1.0 if score(completion, target, task)["exact"] else 0.0


def _completion_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[-1].get("content", "")
    return str(c)


def exact_match_reward(prompts=None, completions=None, target=None, task=None, **kw):
    """TRL reward function: dataset columns `target`, `task` arrive as kwargs."""
    return [reward(_completion_text(c), t, k) for c, t, k in zip(completions, target, task)]


# ----------------------------------------------------------------------------
# GPU side (imports deferred so the CPU side needs no torch)
# ----------------------------------------------------------------------------
RESULTS = os.environ.get("RESULTS_DIR", "/kaggle/working/results")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_model(device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    return model, tok


def generate(model, tok, prompts, n=4, greedy=False, max_new_tokens=192, batch_prompts=24,
             temperature=0.7, top_p=0.8):
    import torch
    outs = []
    was_training = model.training
    model.eval()
    for i in range(0, len(prompts), batch_prompts):
        chunk = prompts[i:i + batch_prompts]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if greedy:
            kw.update(do_sample=False, num_return_sequences=1)
        else:
            kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0, num_return_sequences=n)
        with torch.no_grad():
            g = model.generate(**enc, **kw)
        texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        k = 1 if greedy else n
        outs.extend(texts[j * k:(j + 1) * k] for j in range(len(chunk)))
    if was_training:
        model.train()
    return outs


def run_eval(model, tok, records, path, n=4, greedy=False):
    t0 = time.time()
    comps = generate(model, tok, [r["prompt"] for r in records], n=n, greedy=greedy)
    rows = []
    with open(path, "w") as f:
        for r, cs in zip(records, comps):
            for si, c in enumerate(cs):
                row = {k: v for k, v in r.items() if k != "prompt"}
                row.update(sample=si, completion=c, **score(c, r["target"], r["task"]))
                rows.append(row)
                f.write(json.dumps(row) + "\n")
    acc = {t: sum(x["exact"] for x in rows if x["task"] == t) / max(1, sum(x["task"] == t for x in rows))
           for t in TASKS}
    log(f"eval -> {os.path.basename(path)}: {acc} ({time.time() - t0:.0f}s)")
    return rows


def grpo_kwargs(GRPOConfig, out_dir, max_steps):
    import inspect
    want = dict(
        output_dir=out_dir, learning_rate=2e-5, max_steps=max_steps, num_generations=8,
        per_device_train_batch_size=8, gradient_accumulation_steps=4, generation_batch_size=32,
        max_completion_length=192, temperature=1.0, beta=0.0, logging_steps=1,
        save_strategy="no", report_to="none", seed=SEED, bf16=False, fp16=False,
        remove_unused_columns=False, log_completions=False, top_p=1.0,
        gradient_checkpointing=False,
    )
    params = set(inspect.signature(GRPOConfig).parameters)
    dropped = [k for k in want if k not in params]
    if dropped:
        log("GRPOConfig: dropping unsupported kwargs", dropped)
    return {k: v for k, v in want.items() if k in params}


def train_format(fmt, max_steps=100, time_budget_min=40.0, do_eval=True, n_eval=None):
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    os.makedirs(RESULTS, exist_ok=True)
    log(f"[{fmt}] torch {torch.__version__} transformers {transformers.__version__} trl {trl.__version__}"
        f" device {torch.cuda.get_device_name(0)}")
    model, tok = load_model()
    train_items, eval_items, probe_items = make_splits()
    if n_eval:
        eval_items = eval_items[::max(1, len(eval_items) // n_eval)][:n_eval]
    ev = render(eval_items, fmt, tok)
    probes = render(probe_items, fmt, tok)
    probe_prompts = [p["prompt"] for p in probes]

    if do_eval:
        run_eval(model, tok, ev, f"{RESULTS}/eval_pre_{fmt}.jsonl")

    def save_probe(m, step):
        comps = generate(m, tok, probe_prompts, greedy=True)
        with open(f"{RESULTS}/probe_{fmt}_step{step:03d}.jsonl", "w") as f:
            for p, c in zip(probes, comps):
                row = {k: v for k, v in p.items() if k != "prompt"}
                row.update(step=step, completion=c[0], **score(c[0], p["target"], p["task"]))
                f.write(json.dumps(row) + "\n")

    t_start = time.time()

    class ProbeAndBudget(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kw):
            save_probe(model, 0)

        def on_step_end(self, args, state, control, model=None, **kw):
            if state.global_step % 10 == 0:
                save_probe(model, state.global_step)
            if (time.time() - t_start) / 60 > time_budget_min:
                log(f"[{fmt}] time budget hit at step {state.global_step}; stopping")
                save_probe(model, state.global_step)
                control.should_training_stop = True
            return control

    # per-task reward log: TRL only logs the pooled reward; one reward call == one generation batch
    calls = {"n": 0}
    rlog = open(f"{RESULTS}/rewards_{fmt}.jsonl", "w")

    def logged_reward(prompts=None, completions=None, target=None, task=None, **kw):
        rs = exact_match_reward(prompts, completions, target=target, task=task)
        texts = [_completion_text(c) for c in completions]
        rlog.write(json.dumps(dict(call=calls["n"], task=list(task), reward=rs,
                                   clen=[len(t) for t in texts], parse_fail=[parse_answer(t) is None for t in texts],
                                   prompt_idx=[hash(p) % 10**8 for p in prompts])) + "\n")
        rlog.flush()
        calls["n"] += 1
        return rs
    logged_reward.__name__ = "exact_match"

    train = render(train_items, fmt, tok)
    ds = Dataset.from_list([{"prompt": r["prompt"], "target": r["target"], "task": r["task"]} for r in train])
    cfg = GRPOConfig(**grpo_kwargs(GRPOConfig, f"/tmp/grpo_{fmt}", max_steps))
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = GRPOTrainer(model=model, reward_funcs=logged_reward, args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=peft_cfg, callbacks=[ProbeAndBudget()])
    trainer.train()
    wall = time.time() - t_start
    hist = trainer.state.log_history
    log(f"[{fmt}] trained {trainer.state.global_step} steps in {wall / 60:.1f} min; log keys:",
        sorted({k for h in hist for k in h}))
    with open(f"{RESULTS}/train_log_{fmt}.json", "w") as f:
        json.dump(dict(log_history=hist, steps=trainer.state.global_step, wall_s=wall,
                       versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                     trl=trl.__version__)), f)
    pm = trainer.model
    pm.save_pretrained(f"{RESULTS}/adapter_{fmt}")
    if do_eval:
        run_eval(pm, tok, ev, f"{RESULTS}/eval_post_{fmt}.jsonl")
    log(f"[{fmt}] done")


def smoke():
    import inspect
    import subprocess
    import torch
    os.makedirs(RESULTS, exist_ok=True)
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    log("cuda devices:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    from trl import GRPOConfig
    with open(f"{RESULTS}/grpo_config_signature.txt", "w") as f:
        f.write(str(inspect.signature(GRPOConfig)))
    if os.environ.get("SMOKE_TRAIN_ONLY"):
        t0 = time.time()
        train_format("perbase", max_steps=3, do_eval=False)
        log(f"3 GRPO steps took {time.time() - t0:.0f}s (incl. model load + probes)")
        return
    model, tok = load_model()
    _, eval_items, _ = make_splits()
    sel = [eval_items[0], eval_items[45], eval_items[130], eval_items[200]]
    for fmt in FORMATS:
        recs = render(sel, fmt, tok)
        run_eval(model, tok, recs, f"{RESULTS}/smoke_eval_{fmt}.jsonl")
        print(recs[0]["prompt"])
        with open(f"{RESULTS}/smoke_eval_{fmt}.jsonl") as f:
            for line in list(f)[:2]:
                print("  ->", json.loads(line)["completion"][:200])
    del model
    torch.cuda.empty_cache()
    t0 = time.time()
    train_format("perbase", max_steps=3, do_eval=False)
    log(f"3 GRPO steps took {time.time() - t0:.0f}s (incl. model load + probes)")


def main(stage="smoke"):
    log("stage", stage, "python", sys.version.split()[0])
    if stage == "smoke":
        smoke()
    elif stage.startswith("train_"):
        fmt = stage[len("train_"):]
        train_format(fmt, max_steps=int(os.environ.get("MAX_STEPS", 100)),
                     time_budget_min=float(os.environ.get("TIME_BUDGET_MIN", 40)))
    elif stage.startswith("main"):
        # run one subprocess per format, each pinned to a GPU
        import subprocess
        import torch
        os.makedirs(RESULTS, exist_ok=True)
        fmts = os.environ["FORMATS"].split(",")
        ngpu = max(1, torch.cuda.device_count())
        log("GPUs:", ngpu, "formats:", fmts)
        procs = []
        for i, fmt in enumerate(fmts):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % ngpu))
            logf = open(f"{RESULTS}/stdout_{fmt}.log", "w")
            procs.append((fmt, subprocess.Popen([sys.executable, __file__, f"train_{fmt}"], env=env,
                                                stdout=logf, stderr=subprocess.STDOUT)))
            if ngpu == 1:
                procs[-1][1].wait()
        for fmt, p in procs:
            p.wait()
            log(fmt, "exit", p.returncode)
            print(open(f"{RESULTS}/stdout_{fmt}.log").read()[-3000:])


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1].startswith("train_"):
    main(sys.argv[1])
