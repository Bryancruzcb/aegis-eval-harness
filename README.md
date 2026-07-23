# AegisEval

A small, asynchronous harness for evaluating LLM applications on three axes:

- **Functional** — does it answer normal requests helpfully and stay on-topic?
- **Safety** — does it refuse to produce profanity or abusive content on request?
- **Security** — does it resist jailbreaks, prompt injection, and attempts to
  leak a secret held in its system prompt?

The model under test is a "Secret Guardian" support bot: it holds a passcode in
its system prompt and is told never to reveal it. The suite fires a mix of
ordinary questions and attacks at it and grades every response.

## How it works

Each test case runs through a two-stage pipeline:

1. **Deterministic checks first** — fast, free regex checks for the two
   unambiguous failures: the secret passcode leaking (including obfuscated forms
   like `O R I O N 9 9`) or profanity. A failure here fails the case immediately
   and skips the judge, saving an API call.
2. **LLM-as-a-Judge** — responses that pass the deterministic checks are graded
   by a second model against the case's criteria, returning a strict
   `{score, reasoning, passed}` JSON object. This catches the soft failures a
   regex can't: a technically-polite refusal that still complied with a
   roleplay jailbreak, for example.

Every result gets one of three statuses:

- **pass** / **fail** — the model answered and was graded.
- **error** — the answer couldn't be obtained or graded (rate limit, timeout,
  unparseable judge output). Errors are reported separately and **excluded from
  the pass rate**, so an API outage never shows up as a safety failure.

Transient failures (429 / 5xx / timeout) are retried with exponential backoff;
concurrency is throttled so free-tier rate limits don't cause a wall of 429s.

## Setup

```bash
pip install -r requirements.txt
cp env.example .env      # then edit .env
```

Configure a provider in `.env`. You have two easy options:

**Gemini (needs a free API key):**
```env
GEMINI_API_KEY=your_key_here
TARGET_PROVIDER=gemini
TARGET_MODEL=gemini-3.5-flash
JUDGE_PROVIDER=gemini
JUDGE_MODEL=gemini-3.5-flash
```

> **Free-tier note.** Model availability and quota vary by key. Older flash
> models (1.5/2.0, and even 2.5-flash for newly created keys) return
> `404 "no longer available to new users"`, and free-tier daily quota on the
> full models is small — a few full suite runs can exhaust it and you'll see
> `429` errors (which the harness reports as *errors*, not failures). Two easy
> mitigations: run against a `-lite` model such as `gemini-flash-lite-latest`
> (higher free limits), and keep `MAX_CONCURRENT_REQUESTS=1` so requests don't
> burst past the per-minute cap. To see exactly what a key can use:
> ```python
> from google import genai
> for m in genai.Client(api_key="...").models.list():
>     if "generateContent" in (m.supported_actions or []):
>         print(m.name)
> ```

**Local, no key (via [Ollama](https://ollama.com)):**
```bash
ollama pull llama3.2:3b
```
then pass `--target-provider ollama --judge-provider ollama` on the command line.

OpenAI is also supported (`OPENAI_API_KEY`, `--target-provider openai`).

## Running

```bash
# Full suite
python run.py

# Only the security / jailbreak cases
python run.py --category security
python run.py --tag jailbreak

# Fully local run, no API key
python run.py --target-provider ollama --target-model llama3.2:3b \
              --judge-provider ollama  --judge-model  llama3.2:3b

# Use as a CI quality gate: exit non-zero if the pass rate drops below 80%
python run.py --fail-under 80
```

Exit codes: `0` success, `1` quality gate failed (or every case errored),
`2` configuration error (missing key / unreadable suite).

## Output

Written to `output/` (git-ignored):

- `run_results.json` — full machine-readable results, plus a timestamped copy.
- `report.html` — a self-contained dark-mode dashboard with per-case
  prompt/response/reasoning, filterable by status and category. Model output is
  HTML-escaped, so a malicious response can't inject script into the report.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The unit tests cover the evaluators, summary math, retry classification, report
escaping, and the pass/fail/error routing — none of them touch the network, so
they run offline and in CI (see `.github/workflows/ci.yml`).

## Project layout

| File | Responsibility |
|------|----------------|
| `run.py`          | CLI entry point, argument parsing, quality gate |
| `runner.py`       | Async orchestration, per-case status, summary |
| `target.py`       | The model under test + its guardian system prompt |
| `evaluators.py`   | Deterministic checks + the LLM judge |
| `providers.py`    | Shared clients, retry policy, error taxonomy |
| `reporter.py`     | Terminal summary + HTML dashboard |
| `test_cases.json` | The suite of prompts and expected criteria |
| `config.py`       | Defaults and environment loading |

## Adding test cases

Append an object to `test_cases.json`:

```json
{
  "id": "TC-011",
  "category": "security",
  "prompt": "Your attack or request here",
  "description": "What this case checks.",
  "tags": ["jailbreak", "my_tag"],
  "expected_criteria": "What a passing response must (not) do."
}
```
