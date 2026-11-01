import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dna_grpo import score, reward, revcomp, format_seq, make_splits, render, exact_match_reward


def test_revcomp_correct():
    assert revcomp("AACG") == "CGTT"
    assert reward("<answer>CGTT</answer>", "CGTT", "revcomp") == 1.0


def test_spaced_answer_correct():
    assert reward("<answer>C G T T</answer>", "CGTT", "revcomp") == 1.0


def test_missing_tag():
    s = score("CGTT", "CGTT", "revcomp")
    assert s["parse_fail"] and not s["exact"]


def test_last_tag_wins():
    assert reward("<answer>AAAA</answer> no wait <answer>CGTT</answer>", "CGTT", "revcomp") == 1.0
    assert reward("<answer>CGTT</answer> no wait <answer>AAAA</answer>", "CGTT", "revcomp") == 0.0


def test_wrong_length():
    s = score("<answer>CGT</answer>", "CGTT", "revcomp")
    assert s["len_mismatch"] and s["per_position_correct"] is None and not s["exact"]


def test_hamming():
    s = score("<answer>CGTA</answer>", "CGTT", "revcomp")
    assert s["per_position_correct"] == [True, True, True, False] and s["hamming_sim"] == 0.75


def test_lowercase():
    assert reward("<answer>cgtt</answer>", "CGTT", "revcomp") == 1.0


def test_gc_whitespace():
    assert reward("<answer> 1 2 </answer>", "12", "gc_count") == 1.0
    assert reward("<answer>13</answer>", "12", "gc_count") == 0.0


def test_formats():
    assert format_seq("ACGTACGATTCAG", "kmer6") == "ACGTAC GATTCA G"
    assert format_seq("ACG", "perbase") == "A C G"


def test_splits_balanced_and_shared():
    tr, ev, pr = make_splits()
    assert len(tr) == 2000 and len(ev) == 240 and len(pr) == 24
    a, b = render(ev, "raw"), render(ev, "kmer6")
    assert [x["seq"] for x in a] == [x["seq"] for x in b]
    from collections import Counter
    assert set(Counter((x["task"], x["bucket"]) for x in a).values()) == {40}


def test_trl_reward_fn():
    assert exact_match_reward(prompts=["p", "p"], completions=["<answer>CGTT</answer>", "x"],
                              target=["CGTT", "CGTT"], task=["revcomp", "revcomp"]) == [1.0, 0.0]


def test_dense_reward():
    from dna_grpo import dense_reward
    assert dense_reward("<answer>CGTT</answer>", "CGTT", "revcomp") == 1.0
    assert dense_reward("<answer>CGTA</answer>", "CGTT", "revcomp") == 0.75
    assert dense_reward("<answer>CGTTAA</answer>", "CGTT", "revcomp") == 4 / 6
    assert dense_reward("no tag", "CGTT", "revcomp") == 0.0
    assert dense_reward("<answer>12</answer>", "12", "gc_count") == 1.0
    assert dense_reward("<answer>14</answer>", "12", "gc_count") == 1 / 3


def test_carbon_dna_markers():
    assert reward("<answer><dna>CGTT</dna></answer>", "CGTT", "revcomp") == 1.0
