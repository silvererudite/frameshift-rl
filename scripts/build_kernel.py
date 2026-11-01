"""Write kernels/<stage>/ (entry.py + kernel-metadata.json) from dna_grpo.py.

usage: python scripts/build_kernel.py <stage> [--formats raw,perbase] [--name suffix]
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
USER = "shamimahossain"

HEADER = '''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "trl", "peft", "datasets", "accelerate", "bitsandbytes"], check=False)
# Kaggle ships torchao 0.10, which recent PEFT rejects at LoRA injection time
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
'''

ap = argparse.ArgumentParser()
ap.add_argument("stage")
ap.add_argument("--formats", default="raw,perbase,kmer6")
ap.add_argument("--name", default=None)
ap.add_argument("--max-steps", default="100")
ap.add_argument("--budget-min", default="40")
ap.add_argument("--env", action="append", default=[])
a = ap.parse_args()

name = a.name or a.stage
a.extra_env = "".join(f'    os.environ["{e.split("=")[0]}"] = "{e.split("=",1)[1]}"\n' for e in a.env)
out = ROOT / "kernels" / name
out.mkdir(parents=True, exist_ok=True)
footer = f'''

if not (len(sys.argv) > 1 and sys.argv[1].startswith("train_")):
    os.environ.setdefault("FORMATS", "{a.formats}")
{a.extra_env}
    os.environ.setdefault("MAX_STEPS", "{a.max_steps}")
    os.environ.setdefault("TIME_BUDGET_MIN", "{a.budget_min}")
    main(stage="{a.stage}")
'''
lib = (ROOT / "dna_grpo.py").read_text()
# the header must run only in the parent process, before torch/trl are imported
head = HEADER if a.stage != "train" else ""
(out / "entry.py").write_text(
    'import sys as _s\nif not (len(_s.argv) > 1 and _s.argv[1].startswith("train_")):\n'
    + "".join("    " + l + "\n" for l in head.strip().splitlines()) + "\n" + lib + footer)
meta = {
    "id": f"{USER}/dna-grpo-{name}", "title": f"dna-grpo-{name}", "code_file": "entry.py",
    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
    "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built", out)
