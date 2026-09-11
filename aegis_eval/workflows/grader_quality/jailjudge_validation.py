"""Load a pinned, local-only JAILJUDGE human-annotated test subset."""
import hashlib
import json
import random

COMMIT = "12c5739bf53fbe01709fcd875b949314f391c0a0"
SHA256 = "9d19d2db791b7f3a19993365421c9cdc40b1f241ae0e6dcacbddbcf86aa654da"
SEED = 20260907


def normalize(text):
    return " ".join(text.casefold().split())


def select_rows(rows, previous, *, per_class=100):
    old_prompts = {normalize(row["goal"]) for row in previous}
    old_responses = {normalize(row["target_response"]) for row in previous}
    groups = set()
    pools = {False: [], True: []}
    counts = {"source_rows": len(rows), "overlap_excluded": 0,
              "duplicate_requests_excluded": 0, "non_english_excluded": 0,
              "empty_text_excluded": 0}
    for row in rows:
        if type(row.get("is_jailbroken")) is not bool:
            raise ValueError("JAILJUDGE label must be a JSON boolean")
        if any(not isinstance(row.get(key), str)
               for key in ("user_prompt", "model_response", "language")):
            raise ValueError("JAILJUDGE text fields must be strings")
        if any(not row[key].strip() for key in ("user_prompt", "model_response", "language")):
            counts["empty_text_excluded"] += 1
            continue
        if row["language"] != "en":
            counts["non_english_excluded"] += 1
            continue
        prompt = normalize(row["user_prompt"])
        if prompt in old_prompts or normalize(row["model_response"]) in old_responses:
            counts["overlap_excluded"] += 1
            continue
        if prompt in groups:
            counts["duplicate_requests_excluded"] += 1
            continue
        groups.add(prompt)
        pools[row["is_jailbroken"]].append({"goal": row["user_prompt"],
            "target_response": row["model_response"],
            "human_majority": "1" if row["is_jailbroken"] else "0"})
    rng = random.Random(SEED)
    chosen = []
    for label, pool in pools.items():
        counts[f"eligible_{str(label).lower()}"] = len(pool)
        if len(pool) < per_class:
            raise ValueError("Insufficient independent rows for requested class balance")
        chosen.extend(rng.sample(pool, per_class))
    rng.shuffle(chosen)
    return chosen, counts


def load(path, previous):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("JAILJUDGE file differs from the pinned ID dataset")
    rows, counts = select_rows(json.loads(raw), previous)
    return rows, {"split": "jailjudge-id-validation", "seed": SEED,
                  "dataset_commit": COMMIT, "dataset_sha256": SHA256,
                  "dataset_selection": counts,
                  "label_provenance": "Public human-annotated ID test labels; GPT-4-assisted review",
                  "overlap_check": "Normalized exact request/response exclusion against all JBB rows; semantic overlap not ruled out"}
