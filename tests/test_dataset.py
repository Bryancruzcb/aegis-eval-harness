import json, collections
import config
from runner import load_test_cases

RAW = json.load(open(config.BASE_DIR / "test_cases.json", encoding="utf-8"))
CASES = load_test_cases(RAW)
COUNTS = {"direct": 2, "authority": 4, "roleplay": 3, "obfuscation_spelling": 3,
          "obfuscation_encoding": 5, "translation": 2, "refusal_suppression": 3,
          "hypothetical": 3, "indirect_injection": 3, "instruction_override": 2,
          "crescendo": 6, "profanity": 3, "benign_trap": 8}


def test_total_in_range():
    assert 45 <= len(CASES) <= 60


def test_every_case_has_technique_and_expect_and_one_of_prompt_turns():
    for c in CASES:
        assert c["technique"] != "unspecified"
        assert c["expect"] in ("refuse", "benign")
        assert ("prompt" in c) ^ ("turns" in c)
        assert c["category"] in ("functional", "safety", "security")


def test_multi_turn_and_benign_floors():
    assert sum(1 for c in CASES if "turns" in c) >= 6
    assert sum(1 for c in CASES if c["expect"] == "benign") >= 6


def test_bucket_counts_met():
    got = collections.Counter(c["technique"] for c in CASES)
    for tech, n in COUNTS.items():
        assert got[tech] >= n, f"{tech}: {got[tech]} < {n}"
