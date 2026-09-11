"""Historical Gemini 3.7 comparison on JBB-dev under the old refusal contract.

Not the SafetyGrader experiment. Do not mix its MCC with
`docs/grader-quality-results.json`.
"""
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from google import genai
from google.genai import types

import calibrate
import compare_graders as comparison
import config
from evaluators import _judge_prompt

MODEL = "gemini-3.7-flash"
CAP = 5.0
MAX_OUTPUT = 2048
INPUT_RATE = 0.75 / 1_000_000
OUTPUT_RATE = 3.75 / 1_000_000
OUTPUT = config.BASE_DIR / "output/hosted-comparison/gemini-3.7-flash-dev.json"
SOURCE = config.BASE_DIR / "output/false-positive-ablation/refusal_grader_baseline.py"
NAMES = ["local_two_stage", "hosted_two_stage"]


@contextmanager
def exclusive_run(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def checkpoint_record(payload, record, *, terminal=False):
    failed = any("error" in v for v in record["variants"].values())
    payload.setdefault("failed_records" if failed and not terminal else "records", []).append(record)
    payload["summary"] = comparison.summarize(payload["records"], NAMES)
    return not failed


def accounted_cost(attempt):
    estimate = attempt.get("estimated_usd")
    if isinstance(estimate, (int, float)) and math.isfinite(estimate) and estimate > attempt["reserved_usd"]:
        return estimate
    usage = attempt.get("usage") or {}
    counters = [usage.get("prompt_token_count"), usage.get("total_token_count")]
    outputs = [usage.get("candidates_token_count", 0), usage.get("thoughts_token_count", 0)]
    complete = (all(type(n) is int and n >= 0 for n in counters + outputs)
                and counters[1] == counters[0] + sum(outputs))
    if complete and isinstance(estimate, (int, float)) and math.isfinite(estimate) and estimate >= 0:
        return estimate
    return attempt["reserved_usd"]


def reserve(payload, row, dollars):
    if dollars <= 0 or sum(accounted_cost(a) for a in payload["attempts"]) + dollars > CAP:
        raise ValueError("Authorized budget would be exceeded")
    attempt = {"row": row, "reserved_usd": dollars, "status": "reserved"}
    payload["attempts"].append(attempt)
    return attempt


def usage_cost(usage):
    return ((usage.get("prompt_token_count") or 0) * INPUT_RATE +
            ((usage.get("candidates_token_count") or 0) +
             (usage.get("thoughts_token_count") or 0)) * OUTPUT_RATE)


def retryable_attempt(attempt):
    return attempt.get("code") in (429, 500, 502, 503, 504) and not any(
        "PerDay" in name for name in attempt.get("quota_ids", []))


async def run():
    if datetime.now(timezone.utc).year != 2026:
        raise ValueError("Reverify introductory pricing before running")
    grader = comparison.load_variant("hosted_baseline", SOURCE)
    rows, _ = calibrate._split(calibrate._fetch_rows(refresh=False), seed=calibrate.SPLIT_SEED)
    baseline = json.loads((config.BASE_DIR / "output/false-positive-ablation/qwen-dev.json").read_text())
    if baseline["status"] != "complete" or len(baseline["records"]) != len(rows):
        raise ValueError("Complete matching local baseline is required")
    if any(r["row_sha256"] != comparison.row_hash(row) for r, row in zip(baseline["records"], rows)):
        raise ValueError("Local baseline rows differ")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_hash != baseline["sources"]["two_stage"]["sha256"]:
        raise ValueError("Frozen rubric differs from local baseline")
    identity = {"model": MODEL, "source_sha256": source_hash,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "dependencies": comparison.execution_identity(), "budget_usd": CAP,
                "input_rate_per_million": INPUT_RATE * 1_000_000, "output_rate_per_million": OUTPUT_RATE * 1_000_000,
                "thinking_level": "low", "max_output_tokens": MAX_OUTPUT,
                "temperature": 0,
                "local_baseline_sha256": hashlib.sha256((config.BASE_DIR / "output/false-positive-ablation/qwen-dev.json").read_bytes()).hexdigest()}
    payload = {"identity": identity, "started_utc": datetime.now(timezone.utc).isoformat(),
               "status": "running", "attempts": [], "records": []}
    previous_path = OUTPUT.with_name("hosted-judge-dev.json")
    if not OUTPUT.exists():
        if not previous_path.exists():
            raise ValueError("Prior spending ledger is required")
        previous = json.loads(previous_path.read_text())
        if previous["status"] not in ("complete", "complete_with_errors"):
            raise ValueError("Prior comparison must be finished")
        payload["attempts"] = previous["attempts"]
        payload["prior_trial"] = {"model": "gemini-3.5-flash", "path": str(previous_path),
                                  "sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                                  "reason": "Continued authorized comparison; prior ledger includes 3.8. Known token costs settle reservations; unknown usage keeps full reservation. Original records preserved."}
    if OUTPUT.exists():
        payload = json.loads(OUTPUT.read_text())
        if hashlib.sha256(previous_path.read_bytes()).hexdigest() != payload["prior_trial"]["sha256"]:
            raise ValueError("Prior spending ledger changed; reconcile before resuming")
        if payload["identity"] != identity:
            raise ValueError("Checkpoint configuration changed; do not reset the spending ledger")
        if payload["status"] in ("complete", "complete_with_errors"):
            return 0 if payload["status"] == "complete" else 1
        for i, record in enumerate(payload["records"]):
            if record["row_sha256"] != comparison.row_hash(rows[i]):
                raise ValueError("Checkpoint rows changed")
    client = genai.Client(api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=90000, retry_options=types.HttpRetryOptions(attempts=1)))
    payload["status"] = "running"
    payload.pop("finished_utc", None)
    payload["summary"] = comparison.summarize(payload["records"], NAMES)
    comparison.save(OUTPUT, payload)
    current_row = len(payload["records"])

    async def judge(messages, criteria, *, system_instruction, parse, response_schema, **kwargs):
        contents = _judge_prompt(messages, criteria)
        # Reserve generous input overhead and the full output cap, including thinking.
        input_allowance = len((contents + system_instruction).encode("utf-8")) + 8192
        dollars = input_allowance * INPUT_RATE + MAX_OUTPUT * OUTPUT_RATE
        attempt = reserve(payload, current_row, dollars)
        attempt["model"] = MODEL
        comparison.save(OUTPUT, payload)
        try:
            response = await client.aio.models.generate_content(model=MODEL, contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_instruction,
                    temperature=0, max_output_tokens=MAX_OUTPUT, thinking_config=types.ThinkingConfig(thinking_level="low"),
                    response_mime_type="application/json", response_schema=response_schema))
            usage = response.usage_metadata
            attempt["usage"] = usage.model_dump(mode="json", exclude_none=True) if usage else None
            attempt["estimated_usd"] = usage_cost(attempt["usage"]) if usage else None
            attempt["model_version"] = response.model_version
            attempt["finish_reasons"] = [str(c.finish_reason) for c in (response.candidates or [])]
            attempt["prompt_block_reason"] = str(response.prompt_feedback.block_reason) if response.prompt_feedback else None
            attempt["status"] = "received"
            comparison.save(OUTPUT, payload)
            return parse(response.text)
        except BaseException as exc:
            attempt["status"] = "error"
            attempt["error"] = type(exc).__name__
            attempt["code"] = getattr(exc, "code", None)
            details = getattr(exc, "details", {})
            if isinstance(details, dict):
                attempt["quota_ids"] = [v["quotaId"] for item in details.get("error", {}).get("details", [])
                                        for v in item.get("violations", []) if "quotaId" in v]
            comparison.save(OUTPUT, payload)
            raise

    comparison.run_llm_judge_eval_conversation = judge
    for current_row in range(len(payload["records"]), len(rows)):
        row = rows[current_row]
        results = {"two_stage": await comparison.evaluate(grader, row["goal"], row["target_response"],
                                                           "gemini", MODEL, 95)}
        for retry in range(2):
            if ("error" not in results["two_stage"] or results["two_stage"]["error"] == "ValueError"
                    or not payload["attempts"] or not retryable_attempt(payload["attempts"][-1])):
                break
            await asyncio.sleep(20 * (retry + 1))
            results = {"two_stage": await comparison.evaluate(grader, row["goal"], row["target_response"],
                                                               "gemini", MODEL, 95)}
        record = {"row": current_row, "row_sha256": comparison.row_hash(row),
                  "actual": calibrate._actual_jailbroken(row), "variants": {
                      "local_two_stage": baseline["records"][current_row]["variants"]["two_stage"],
                      "hosted_two_stage": results["two_stage"]}}
        terminal = results["two_stage"].get("error") == "JudgeParseError"
        succeeded = checkpoint_record(payload, record, terminal=terminal)
        comparison.save(OUTPUT, payload)
        print(f"Hosted {current_row + 1}/{len(rows)}", flush=True)
        if not succeeded and not terminal:
            payload["status"] = "stopped_on_error"
            break
        await asyncio.sleep(1)
    else:
        payload["status"] = ("complete_with_errors" if any(
            "error" in r["variants"]["hosted_two_stage"] for r in payload["records"]) else "complete")
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    comparison.save(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "summary": payload["summary"],
                      "accounted_usd": sum(accounted_cost(a) for a in payload["attempts"]),
                      "reserved_usd": sum(a["reserved_usd"] for a in payload["attempts"]),
                      "usage_estimated_usd": sum(a.get("estimated_usd") or 0 for a in payload["attempts"])}, indent=2))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    with exclusive_run(OUTPUT.with_name("hosted-judge-dev.lock")):
        with exclusive_run(OUTPUT.with_suffix(".lock")):
            raise SystemExit(asyncio.run(run()))
