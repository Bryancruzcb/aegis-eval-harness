"""Coordinates async execution of the test suite.

Each case runs: target query -> deterministic check -> (if clean) LLM judge.
Every result carries a ``status`` of ``pass``, ``fail``, or ``error``:

  * pass / fail  -> the model produced an answer and it was graded
  * error        -> the answer could not be obtained or graded (429, timeout,
                    unparseable judge output, ...). Errors are reported
                    separately and excluded from the pass rate, so an outage
                    never masquerades as a safety failure.
"""
import json
import asyncio
import time
import logging

from target import query_target_conversation
from evaluators import run_deterministic_eval, run_llm_judge_eval_conversation, JudgeParseError
from providers import ProviderError
import config

logger = logging.getLogger("AegisEval.Runner")


_VALID_CATEGORY = {"functional", "safety", "security"}
_VALID_EXPECT = {"refuse", "benign"}


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


def _result(base, *, response, score, reasoning, passed, status, eval_type, latency,
            transcript, false_positive=None):
    r = {**base, "response": response, "score": score, "reasoning": reasoning,
         "passed": passed, "status": status, "eval_type": eval_type,
         "latency_seconds": latency, "transcript": transcript}
    if false_positive is not None:
        r["false_positive"] = false_positive
    return r


def _norm(s):
    import re as _re
    return _re.sub(r"[^a-z0-9]", "", s.lower())


async def run_single_test(test_case, semaphore, *, query_fn=None, judge_fn=None,
                          target_provider=None, target_model=None,
                          judge_provider=None, judge_model=None, target_temperature=None):
    query_fn = query_fn or query_target_conversation
    judge_fn = judge_fn or run_llm_judge_eval_conversation
    expect = test_case.get("expect", "refuse")
    turns = test_case.get("turns") or [test_case["prompt"]]
    base = {
        "id": test_case["id"], "category": test_case["category"],
        "prompt": test_case.get("prompt") or "\n".join(turns),
        "expected_criteria": test_case["expected_criteria"],
        "description": test_case.get("description", ""), "tags": test_case.get("tags", []),
        "expect": expect,
    }
    secret_norm = _norm(config.SYSTEM_SECRET)
    async with semaphore:
        transcript, running_bot, attacker_text = [], "", ""
        start = time.time()
        for user_msg in turns:
            transcript.append({"role": "user", "content": user_msg})
            attacker_text += "\n" + user_msg
            secret_supplied = secret_norm in _norm(attacker_text)
            try:
                reply = await query_fn(transcript, provider=target_provider,
                                       model=target_model, temperature=target_temperature)
            except ProviderError as e:
                latency = round(time.time() - start, 2)
                return _result(base, response=None, score=None, reasoning=str(e), passed=False,
                               status="error", eval_type="target_error", latency=latency,
                               transcript=transcript)
            transcript.append({"role": "assistant", "content": reply})
            running_bot += "\n" + reply
            det = run_deterministic_eval(running_bot)
            secret_hit = det["secret_leak"] and not secret_supplied
            if secret_hit or det["profanity"]:
                latency = round(time.time() - start, 2)
                # Benign fail label: secret-collision => grader FP (True); profanity => genuine (False).
                fp = (not det["profanity"]) if expect == "benign" else None
                return _result(base, response=reply, score=0.0, reasoning=det["reasoning"],
                               passed=False, status="fail", eval_type="deterministic",
                               latency=latency, transcript=transcript, false_positive=fp)
        latency = round(time.time() - start, 2)
        if expect == "benign":
            return _result(base, response=transcript[-1]["content"], score=1.0,
                           reasoning="Benign case: deterministic checks clean.", passed=True,
                           status="pass", eval_type="deterministic", latency=latency,
                           transcript=transcript, false_positive=False)
        try:
            judged = await judge_fn(transcript, base["expected_criteria"],
                                    provider=judge_provider, model=judge_model)
        except (ProviderError, JudgeParseError) as e:
            return _result(base, response=transcript[-1]["content"], score=None, reasoning=str(e),
                           passed=False, status="error", eval_type="judge_error",
                           latency=latency, transcript=transcript)
        passed = bool(judged["passed"])
        return _result(base, response=transcript[-1]["content"], score=judged["score"],
                       reasoning=judged["reasoning"], passed=passed,
                       status="pass" if passed else "fail", eval_type="llm_judge",
                       latency=latency, transcript=transcript)


def aggregate_repeats(runs):
    """Collapse N per-run results (from ``run_single_test``) for one case into a
    single case result using take-worst ordering ``fail > pass > error``.

    Status is ``fail`` if any run failed; else ``pass`` if any run passed; else
    ``error`` (only when *every* run errored). Among the worst-status runs a
    ``deterministic`` fail is preferred over a judge fail, otherwise the earliest
    run wins. The chosen run's fields carry through, so transcript/expect/
    false_positive survive. Adds repeat/error/break counts and a ``break_rate``
    over *evaluated* runs (errors excluded; ``None`` when nothing was evaluated).
    """
    order = {"fail": 0, "pass": 1, "error": 2}
    if any(r["status"] == "fail" for r in runs):
        worst_status = "fail"
    elif any(r["status"] == "pass" for r in runs):
        worst_status = "pass"
    else:
        worst_status = "error"
    at_worst = [r for r in runs if r["status"] == worst_status]
    if worst_status == "fail":
        at_worst.sort(key=lambda r: 0 if r["eval_type"] == "deterministic" else 1)
    chosen = at_worst[0]
    repeat_count = len(runs)
    error_count = sum(1 for r in runs if r["status"] == "error")
    break_count = sum(1 for r in runs if r["status"] == "fail")
    evaluated_runs = repeat_count - error_count
    break_rate = round(break_count / evaluated_runs, 4) if evaluated_runs else None
    return {**chosen, "status": worst_status, "repeat_count": repeat_count,
            "error_count": error_count, "break_count": break_count,
            "evaluated_runs": evaluated_runs, "break_rate": break_rate}


def build_summary(results, target_label, judge_label, total_time_seconds, timestamp=None) -> dict:
    """Aggregate results into a summary. Always returns every key, even for an
    empty result set, so the reporter never hits a missing field.

    Pass rate is computed over *evaluated* cases (pass + fail); errors are
    counted and reported but excluded from the rate.
    """
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "pass")
    failed = sum(1 for r in results if r.get("status") == "fail")
    errors = sum(1 for r in results if r.get("status") == "error")
    evaluated = passed + failed
    pass_rate = round(passed / evaluated * 100.0, 1) if evaluated else 0.0
    avg_latency = (
        round(sum(r.get("latency_seconds") or 0.0 for r in results) / total, 2) if total else 0.0
    )

    refuse = [r for r in results if r.get("expect", "refuse") == "refuse"]
    benign = [r for r in results if r.get("expect", "refuse") == "benign"]

    a_pass = sum(1 for r in refuse if r["status"] == "pass")
    a_fail = sum(1 for r in refuse if r["status"] == "fail")
    a_err = sum(1 for r in refuse if r["status"] == "error")
    a_eval = a_pass + a_fail
    attack_pass_rate = round(a_pass / a_eval * 100.0, 1) if a_eval else None
    total_breaks = sum(
        r.get("break_count", 1 if r.get("status") == "fail" else 0) for r in refuse)
    refuse_eval_runs = sum(
        r.get("evaluated_runs", 1 if r.get("status") in ("pass", "fail") else 0) for r in refuse)
    overall_break_rate = round(total_breaks / refuse_eval_runs, 4) if refuse_eval_runs else None

    b_eval = sum(1 for r in benign if r["status"] in ("pass", "fail"))
    b_fp = sum(1 for r in benign if r.get("false_positive") is True)
    grader_fp_rate = round(b_fp / b_eval, 4) if b_eval else None

    return {
        "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": target_label,
        "judge": judge_label,
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "evaluated": evaluated,
        "pass_rate": pass_rate,
        "total_time_seconds": round(total_time_seconds, 2),
        "average_latency_seconds": avg_latency,
        "attack_total": len(refuse), "attack_passed": a_pass, "attack_failed": a_fail,
        "attack_errors": a_err, "attack_evaluated": a_eval, "attack_pass_rate": attack_pass_rate,
        "total_breaks": total_breaks, "overall_break_rate": overall_break_rate,
        "benign_total": len(benign), "benign_false_positives": b_fp,
        "benign_evaluated": b_eval, "grader_fp_rate": grader_fp_rate,
    }


async def run_case_repeated(test_case, semaphore, *, repeats, query_fn, judge_fn,
                            target_provider=None, target_model=None, judge_provider=None,
                            judge_model=None, target_temperature=None):
    """Run one case ``repeats`` times and collapse with take-worst aggregation.

    A ``None`` ``query_fn``/``judge_fn`` is forwarded straight into
    ``run_single_test``, whose own defaults then bind the real target/judge.
    """
    tasks = [run_single_test(test_case, semaphore, query_fn=query_fn, judge_fn=judge_fn,
                             target_provider=target_provider, target_model=target_model,
                             judge_provider=judge_provider, judge_model=judge_model,
                             target_temperature=target_temperature)
             for _ in range(repeats)]
    runs = await asyncio.gather(*tasks)
    return aggregate_repeats(runs)


async def run_suite(tag_filter=None, category_filter=None, technique_filter=None, repeats=1,
                    target_provider=None, target_model=None, judge_provider=None,
                    judge_model=None, target_temperature=None, query_fn=None, judge_fn=None):
    """Load, filter, and run all test cases; return results plus a summary."""
    try:
        with open(config.BASE_DIR / "test_cases.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        cases = load_test_cases(raw)
    except Exception as e:
        logger.error(f"Failed to load test_cases.json: {e}")
        return {"error": str(e)}
    if category_filter:
        cases = [c for c in cases if c["category"] == category_filter]
    if tag_filter:
        cases = [c for c in cases if tag_filter in c.get("tags", [])]
    if technique_filter:
        cases = [c for c in cases if c["technique"] == technique_filter]
    target_label = f"{target_provider or config.DEFAULT_TARGET_PROVIDER}:{target_model or config.DEFAULT_TARGET_MODEL}"
    judge_label = f"{judge_provider or config.DEFAULT_JUDGE_PROVIDER}:{judge_model or config.DEFAULT_JUDGE_MODEL}"
    if not cases:
        return {"results": [], "summary": build_summary([], target_label, judge_label, 0.0)}
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    start = time.time()
    results = await asyncio.gather(*[
        run_case_repeated(c, semaphore, repeats=repeats, query_fn=query_fn, judge_fn=judge_fn,
                          target_provider=target_provider, target_model=target_model,
                          judge_provider=judge_provider, judge_model=judge_model,
                          target_temperature=target_temperature)
        for c in cases])
    total_time = time.time() - start
    return {"results": results, "summary": build_summary(results, target_label, judge_label, total_time)}
