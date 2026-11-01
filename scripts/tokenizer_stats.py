"""B5: tokens-per-base and token-boundary offsets per eval/probe example (CPU, tokenizer only).

For each example x format: the input-token index of every base, and whether the base sits on
a token boundary (first or last base of its token). Written to outputs/tokenizer_stats.jsonl.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer

from dna_grpo import FORMATS, MODEL_ID, base_char_positions, bucket_of, format_seq, make_splits, user_text

tok = AutoTokenizer.from_pretrained(MODEL_ID)
_, ev, pr = make_splits()
out = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "tokenizer_stats.jsonl"
with open(out, "w") as f:
    for it in ev + pr:
        for fmt in FORMATS:
            text = user_text(it["seq"], it["task"], fmt)
            start = text.index(format_seq(it["seq"], fmt))
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            char_tok = {}
            for ti, (a, b) in enumerate(enc["offset_mapping"]):
                for c in range(a, b):
                    char_tok[c] = ti
            base_tok = [char_tok[start + c] for c in base_char_positions(it["seq"], fmt)]
            n_tok = len(set(base_tok))
            boundary = [(i == 0 or base_tok[i - 1] != t) or (i == len(base_tok) - 1 or base_tok[i + 1] != t)
                        for i, t in enumerate(base_tok)]
            f.write(json.dumps(dict(id=it["id"], task=it["task"], format=fmt, length=len(it["seq"]),
                                    bucket=bucket_of(len(it["seq"])), n_tokens=n_tok,
                                    tokens_per_base=n_tok / len(it["seq"]), base_token=base_tok,
                                    is_boundary=boundary)) + "\n")
print("wrote", out)
