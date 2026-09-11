# Fresh row 48 parse failure

Decode-only postmortem. No prompt text, no JAILJUDGE content.

## What failed

`fresh` row 48, candidate `SafetyGrader`, local `qwen2.5:latest` digest
`845dbda0ea48…`, temperature 0. Legacy scored the row. Candidate stopped
the cohort.

Three parse retries produced the same judge output hash
`2b492e01310bc019b5b6862ebfc8999ff76f5267363a0e2538d365ca3ae72d7a`
(79 characters). Fence-stripping did not apply.

## Redacted parse error

Pydantic `SafetyJudgment`: `loc=["reasoning"]`, `type="missing"`.

The object had keys `harmful_assistance`, `refused`, `partial_refusal`
with boolean values. `reasoning` was absent. String booleans were not
the failure.

## Decoding change that unblocks it

Ollama `response_format.json_schema` (and native `/api/chat` `format`)
with `SafetyJudgment.model_json_schema()` returns a valid object:
hash `06da4d3182148e472a722b10befe50919a9eb69c9ce87d93782d10b027f4f1b2`,
all four fields present, booleans stay booleans.

The frozen `HARM_PROMPT` / `BENIGN_PROMPT` were not edited. Schema pin
is a decoding change, allowed by the ML-engineer plan Task 0.4.

Rows 0–47 of `fresh` were judged under `json_object` only. From row 48
onward the Ollama path pins the schema. That mix is recorded here so
the cohort is not silently treated as one decoding condition.
