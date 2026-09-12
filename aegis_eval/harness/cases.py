"""Validate and default raw evaluation cases."""

_VALID_CATEGORY = {"functional", "safety", "security"}
_VALID_EXPECT = {"refuse", "benign", "comply"}


def load_test_cases(raw):
    cases = []
    for i, c in enumerate(raw):
        for key in ("id", "category", "expected_criteria"):
            if key not in c:
                raise ValueError(f"Case #{i} missing required key '{key}'")
        has_prompt, has_turns = "prompt" in c, "turns" in c
        if has_prompt == has_turns:
            raise ValueError(f"Case {c.get('id')} must have exactly one of 'prompt'/'turns'")
        if has_turns:
            t = c["turns"]
            if not isinstance(t, list) or not t or not all(isinstance(x, str) for x in t):
                raise ValueError(f"Case {c.get('id')} 'turns' must be a non-empty list of strings")
        if c["category"] not in _VALID_CATEGORY:
            raise ValueError(f"Case {c.get('id')} bad category '{c['category']}'")
        expect = c.get("expect", "refuse")
        if expect not in _VALID_EXPECT:
            raise ValueError(f"Case {c.get('id')} bad expect '{expect}'")
        cases.append({**c, "technique": c.get("technique", "unspecified"),
                      "expect": expect, "tags": c.get("tags", [])})
    return cases
