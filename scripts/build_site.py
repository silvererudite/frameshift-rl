"""Build site/index.html: one self-contained page (Plotly from CDN, data inlined as JSON).

usage: python scripts/build_site.py   (run after make_figures.py)
"""
import base64
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (loads all outputs on import)

ROOT = mf.ROOT
FIG = mf.FIG
SITE = ROOT / "site"
SITE.mkdir(exist_ok=True)
summary = json.load(open(FIG / "summary.json"))
findings = json.load(open(FIG / "titles.json")) if (FIG / "titles.json").exists() else {}


def f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


data = {"formats": mf.FORMATS, "label": mf.LABEL, "color": mf.COL, "buckets": mf.BUCKETS}
# learning curves (per task, 10-step rolling mean + raw)
data["curves"] = {}
for fmt in mf.RW:
    r = mf.reward_by_step(fmt)
    data["curves"][fmt] = {t: {"step": r.loc[t].index.tolist(), "raw": [f(v) for v in r.loc[t].values],
                               "smooth": [f(v) for v in r.loc[t].rolling(10, min_periods=1).mean().values]}
                           for t in ("revcomp", "gc_count")}
# accuracy by bucket with CIs
data["acc"] = {}
for task in ("revcomp", "gc_count"):
    for fmt in mf.FORMATS:
        for ph in ("pre", "post"):
            data["acc"][f"{task}|{fmt}|{ph}"] = [[f(v) for v in mf.acc(ph, fmt, task, b, "pc")] for b in mf.BUCKETS]
data["fig4"] = summary.get("fig4", {})
# example browser: revcomp prompts, sample 0 answer per format/phase
ex = {}
for (ph, fmt), d in mf.EV.items():
    for r in d[(d["sample"] == 0)].itertuples():
        e = ex.setdefault(r.id, {"id": r.id, "task": r.task, "length": int(r.length), "seq": r.seq,
                                 "target": r.target, "out": {}})
        e["out"][f"{fmt}|{ph}"] = r.answer if isinstance(r.answer, str) else None
data["examples"] = sorted(ex.values(), key=lambda e: (e["task"] != "revcomp", e["length"]))
data["summary"] = {k: v for k, v in summary.items() if not isinstance(v, dict)}
fig1 = base64.b64encode((FIG / "fig1_how_the_model_sees_dna.png").read_bytes()).decode()

tldr = "".join(f"<li>{x}</li>" for x in findings.get("tldr", []))
html = (pathlib.Path(__file__).parent / "site_template.html").read_text()
html = (html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__FIG1__", fig1).replace("__TLDR__", tldr)
        .replace("__SUB__", mf.SUB))
(SITE / "index.html").write_text(html)
print("wrote", SITE / "index.html", len(html) // 1024, "KB")
