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
from evaluators import (
    run_deterministic_eval,
    run_llm_judge_eval_conversation,
    JudgeParseError,
    _alphanumeric,
)
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


def _base_fields(test_case):
    """The identity fields every result for a case carries, however it ends.

    Used by both the normal path and the unexpected-exception fallback so an
    internally-errored result has the same shape as its siblings.
    """
    turns = test_case.get("turns") or [test_case.get("prompt", "")]
    return {
        "id": test_case.get("id"), "category": test_case.get("category"),
        "prompt": test_case.get("prompt") or "\n".join(turns),
        "expected_criteria": test_case.get("expected_criteria"),
        "description": test_case.get("description", ""), "tags": test_case.get("tags", []),
        "expect": test_case.get("expect", "refuse"),
    }


def _log_done(tc_id, status, score=None):
    """One completion line per run, so a long suite shows progress."""
    detail = f" (score {score})" if score is not None else ""
    logger.info(f"Completed {tc_id}: {status}{detail}")


def _internal_error_result(test_case, exc):
    """Turn an unexpected (non-ProviderError) exception into an error result so
    one bad case degrades instead of sinking the whole run."""
    return _result(_base_fields(test_case), response=None, score=None, reasoning=str(exc),
                   passed=False, status="error", eval_type="internal_error", latency=0.0,
                   transcript=[])


async def run_single_test(test_case, semaphore, *, query_fn=None, judge_fn=None,
                          target_provider=None, target_model=None,
                          judge_provider=None, judge_model=None, target_temperature=None):
    query_fn = query_fn or query_target_conversation
    judge_fn = judge_fn or run_llm_judge_eval_conversation
    expect = test_case.get("expect", "refuse")
    turns = test_case.get("turns") or [test_case["prompt"]]
    base = _base_fields(test_case)
    tc_id = base["id"]
    secret_norm = _alphanumeric(config.SYSTEM_SECRET)
    async with semaphore:
        logger.info(f"Starting {tc_id} ({base['category']})")
        transcript, running_bot, attacker_text = [], "", ""
        start = time.time()
        for user_msg in turns:
            transcript.append({"role": "user", "content": user_msg})
            attacker_text += "\n" + user_msg
            secret_supplied = secret_norm in _alphanumeric(attacker_text)
            try:
                reply = await query_fn(transcript, provider=target_provider,
                                       model=target_model, temperature=target_temperature)
            except ProviderError as e:
                latency = round(time.time() - start, 2)
                logger.warning(f"{tc_id}: target error: {e}")
                _log_done(tc_id, "error")
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
                _log_done(tc_id, "fail (deterministic)", 0.0)
                return _result(base, response=reply, score=0.0, reasoning=det["reasoning"],
                               passed=False, status="fail", eval_type="deterministic",
                               latency=latency, transcript=transcript, false_positive=fp)
        latency = round(time.time() - start, 2)
        if expect == "benign":
            _log_done(tc_id, "pass (benign, deterministic-only)", 1.0)
            return _result(base, response=transcript[-1]["content"], score=1.0,
                           reasoning="Benign case: deterministic checks clean.", passed=True,
                           status="pass", eval_type="deterministic", latency=latency,
                           transcript=transcript, false_positive=False)
        logger.info(f"Deterministic checks passed for {tc_id}. Invoking LLM Judge...")
        try:
            judged = await judge_fn(transcript, base["expected_criteria"],
                                    provider=judge_provider, model=judge_model)
        except (ProviderError, JudgeParseError) as e:
            logger.warning(f"{tc_id}: judge error: {e}")
            _log_done(tc_id, "error")
            return _result(base, response=transcript[-1]["content"], score=None, reasoning=str(e),
                           passed=False, status="error", eval_type="judge_error",
                           latency=latency, transcript=transcript)
        passed = bool(judged["passed"])
        _log_done(tc_id, "pass" if passed else "fail", judged["score"])
        return _result(base, response=transcript[-1]["content"], score=judged["score"],
                       reasoning=judged["reasoning"], passed=passed,
                       status="pass" if passed else "fail", eval_type="llm_judge",
                       latency=latency, transcript=transcript)


def aggregate_repeats(runs):
    """Collapse N per-run results (from ``run_single_test``) for one case into a
    single case result using take-worst ordering ``fail > pass > error``.

    Requires a non-empty ``runs`` list (i.e. ``repeats >= 1``); raises
    ``ValueError`` otherwise rather than indexing an empty selection.

    Status is ``fail`` if any run failed; else ``pass`` if any run passed; else
    ``error`` (only when *every* run errored). Among the worst-status runs a
    ``deterministic`` fail is preferred over a judge fail, otherwise the earliest
    run wins. The chosen run's fields carry through, so transcript/expect/
    false_positive survive. Adds repeat/error/break counts and a ``break_rate``
    over *evaluated* runs (errors excluded; ``None`` when nothing was evaluated).
    """
    if not runs:
        raise ValueError(
            "aggregate_repeats requires at least one run "
            "(precondition: repeats >= 1); got an empty run list."
        )
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

    a_pass = sum(1 for r in refuse if r.get("status") == "pass")
    a_fail = sum(1 for r in refuse if r.get("status") == "fail")
    a_err = sum(1 for r in refuse if r.get("status") == "error")
    a_eval = a_pass + a_fail
    attack_pass_rate = round(a_pass / a_eval * 100.0, 1) if a_eval else None
    total_breaks = sum(
        r.get("break_count", 1 if r.get("status") == "fail" else 0) for r in refuse)
    refuse_eval_runs = sum(
        r.get("evaluated_runs", 1 if r.get("status") in ("pass", "fail") else 0) for r in refuse)
    overall_break_rate = round(total_breaks / refuse_eval_runs, 4) if refuse_eval_runs else None

    b_eval = sum(1 for r in benign if r.get("status") in ("pass", "fail"))
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
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    # An unexpected exception degrades that repeat to an error run rather than
    # killing the case (and, via the suite gather, the whole run).
    runs = []
    for r in raw:
        if isinstance(r, Exception):
            logger.error(f"{test_case.get('id')}: unexpected error in repeat: {r}")
            runs.append(_internal_error_result(test_case, r))
        else:
            runs.append(r)
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
    # Taken from the data as loaded, so the --technique warning below still lists
    # something useful when --tag/--category already emptied the selection.
    available_techniques = sorted({c["technique"] for c in cases})
    if category_filter:
        cases = [c for c in cases if c["category"] == category_filter]
    if tag_filter:
        cases = [c for c in cases if tag_filter in c.get("tags", [])]
    if technique_filter:
        cases = [c for c in cases if c["technique"] == technique_filter]
        if not cases:
            # spec 4.8: warn with the valid techniques, then proceed empty.
            logger.warning(
                f"--technique '{technique_filter}' matched no test cases. "
                f"Valid techniques in the selected data: {', '.join(available_techniques)}. "
                "Proceeding with an empty selection."
            )
    logger.info(f"Selected {len(cases)} test case(s).")
    target_label = f"{target_provider or config.DEFAULT_TARGET_PROVIDER}:{target_model or config.DEFAULT_TARGET_MODEL}"
    judge_label = f"{judge_provider or config.DEFAULT_JUDGE_PROVIDER}:{judge_model or config.DEFAULT_JUDGE_MODEL}"
    if not cases:
        if not technique_filter:
            logger.warning("No test cases match the selection criteria.")
        return {"results": [], "summary": build_summary([], target_label, judge_label, 0.0)}
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    start = time.time()
    raw_results = await asyncio.gather(*[
        run_case_repeated(c, semaphore, repeats=repeats, query_fn=query_fn, judge_fn=judge_fn,
                          target_provider=target_provider, target_model=target_model,
                          judge_provider=judge_provider, judge_model=judge_model,
                          target_temperature=target_temperature)
        for c in cases], return_exceptions=True)
    total_time = time.time() - start
    # Convert any unexpected exception into an ERROR result so one bad case
    # can't sink the entire suite (and lose a whole paid run's results).
    results = []
    for c, r in zip(cases, raw_results):
        if isinstance(r, Exception):
            logger.error(f"{c.get('id')}: unexpected error: {r}")
            results.append(_internal_error_result(c, r))
        else:
            results.append(r)
    summary = build_summary(results, target_label, judge_label, total_time)
    logger.info(
        f"Suite completed. Pass rate: {summary['pass_rate']}% "
        f"({summary['passed']}/{summary['evaluated']} evaluated, {summary['errors']} error(s)) "
        f"| Time: {summary['total_time_seconds']}s"
    )
    return {"results": results, "summary": summary}
