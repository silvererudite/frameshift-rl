# FineDNA `rl_mix` schema (streamed, first 2–5k rows)

Columns: `prompt` (chat list), `answer_text`, `task_group`, `task_type`, `verifier_metadata`, `gold_answer_json`, `domain`.

task_group counts (first 2,000 rows): basic_dna 633, bioreason 409, genomeqa 271, bio_mcq 219, vepqa 148,
dna_skills 120, seqrecqa_position 112, if 88.

`basic_dna` holds ~60 code-checkable task types at ~20–50 rows each per 5k rows, including
`reverse_complement` (27), `gc_content` (37), `pyrimidine_count`, `purine_count`, `cpg_count`, `telomere_repeat_count`, …

**Decision:** FineDNA is the source of the task family (reverse complement, base counting). We generate
prompts procedurally instead of sampling them because:
- we need *identical sequences* rendered in three input formats, with controlled lengths (12–90 bp, balanced buckets);
- `reverse_complement` rows are rare (~0.5% of the mix), so 2,000 train + 240 eval items would need a large scan,
  and their prompt wording and lengths vary.
