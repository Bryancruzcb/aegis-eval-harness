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

from target import query_target
from evaluators import run_deterministic_eval, run_llm_judge_eval, JudgeParseError
from providers import ProviderError
import config

logger = logging.getLogger("AegisEval.Runner")


def _result(base: dict, *, response, score, reasoning, passed, status, eval_type, latency) -> dict:
    return {
        **base,
        "response": response,
        "score": score,
        "reasoning": reasoning,
        "passed": passed,
        "status": status,
        "eval_type": eval_type,
        "latency_seconds": latency,
    }


async def run_single_test(
    test_case: dict,
    semaphore: asyncio.Semaphore,
    *,
    query_fn=None,
    judge_fn=None,
    target_provider: str = None,
    target_model: str = None,
    judge_provider: str = None,
    judge_model: str = None,
) -> dict:
    """Run one test case through the target and the evaluators.

    ``query_fn`` and ``judge_fn`` are injectable for testing; they default to
    the real target and judge.
    """
    query_fn = query_fn or query_target
    judge_fn = judge_fn or run_llm_judge_eval

    async with semaphore:
        base = {
            "id": test_case["id"],
            "category": test_case["category"],
            "prompt": test_case["prompt"],
            "expected_criteria": test_case["expected_criteria"],
            "description": test_case.get("description", ""),
            "tags": test_case.get("tags", []),
        }
        tc_id = base["id"]
        logger.info(f"Starting {tc_id} ({base['category']})")

        start_time = time.time()

        # 1. Query the target model. Infra failure -> ERROR, not a test failure.
        try:
            target_response = await query_fn(
                base["prompt"], provider=target_provider, model=target_model
            )
        except ProviderError as e:
            latency = round(time.time() - start_time, 2)
            logger.warning(f"{tc_id}: target error: {e}")
            return _result(
                base, response=f"[target error] {e}", score=None, reasoning=str(e),
                passed=False, status="error", eval_type="target_error", latency=latency,
            )

        latency = round(time.time() - start_time, 2)

        # 2. Deterministic checks first (fast, cost-free). A failure here is a
        #    genuine, unambiguous test failure — skip the judge to save cost.
        det = run_deterministic_eval(target_response)
        if not det["passed"]:
            return _result(
                base, response=target_response, score=det["score"], reasoning=det["reasoning"],
                passed=False, status="fail", eval_type="deterministic", latency=latency,
            )

        # 3. LLM judge. Infra/parse failure -> ERROR, not a test failure.
        logger.info(f"Deterministic checks passed for {tc_id}. Invoking LLM Judge...")
        try:
            judged = await judge_fn(
                prompt=base["prompt"],
                response=target_response,
                expected_criteria=base["expected_criteria"],
                provider=judge_provider,
                model=judge_model,
            )
        except (ProviderError, JudgeParseError) as e:
            logger.warning(f"{tc_id}: judge error: {e}")
            return _result(
                base, response=target_response, score=None, reasoning=str(e),
                passed=False, status="error", eval_type="judge_error", latency=latency,
            )

        passed = bool(judged["passed"])
        logger.info(f"Completed {tc_id}. Score: {judged['score']} | Passed: {passed}")
        return _result(
            base, response=target_response, score=judged["score"], reasoning=judged["reasoning"],
            passed=passed, status=("pass" if passed else "fail"), eval_type="llm_judge", latency=latency,
        )


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
    }


async def run_suite(
    tag_filter: str = None,
    category_filter: str = None,
    target_provider: str = None,
    target_model: str = None,
    judge_provider: str = None,
    judge_model: str = None,
) -> dict:
    """Load, filter, and run all test cases; return results plus a summary."""
    try:
        with open(config.BASE_DIR / "test_cases.json", "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load test_cases.json: {e}")
        return {"error": str(e)}

    if category_filter:
        test_cases = [tc for tc in test_cases if tc.get("category") == category_filter]
    if tag_filter:
        test_cases = [tc for tc in test_cases if tag_filter in tc.get("tags", [])]
    logger.info(f"Selected {len(test_cases)} test case(s).")

    target_label = f"{target_provider or config.DEFAULT_TARGET_PROVIDER}:{target_model or config.DEFAULT_TARGET_MODEL}"
    judge_label = f"{judge_provider or config.DEFAULT_JUDGE_PROVIDER}:{judge_model or config.DEFAULT_JUDGE_MODEL}"

    if not test_cases:
        logger.warning("No test cases match the selection criteria.")
        return {"results": [], "summary": build_summary([], target_label, judge_label, 0.0)}

    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    tasks = [
        run_single_test(
            tc, semaphore,
            target_provider=target_provider, target_model=target_model,
            judge_provider=judge_provider, judge_model=judge_model,
        )
        for tc in test_cases
    ]

    start_suite_time = time.time()
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    total_suite_time = time.time() - start_suite_time

    # Convert any unexpected exception into an ERROR result so one bad case
    # can't sink the entire suite.
    results = []
    for tc, r in zip(test_cases, raw_results):
        if isinstance(r, Exception):
            logger.error(f"{tc.get('id')}: unexpected error: {r}")
            results.append(_result(
                {
                    "id": tc.get("id"), "category": tc.get("category"), "prompt": tc.get("prompt"),
                    "expected_criteria": tc.get("expected_criteria"),
                    "description": tc.get("description", ""), "tags": tc.get("tags", []),
                },
                response=f"[unexpected error] {r}", score=None, reasoning=str(r),
                passed=False, status="error", eval_type="internal_error", latency=0.0,
            ))
        else:
            results.append(r)

    summary = build_summary(results, target_label, judge_label, total_suite_time)
    logger.info(
        f"Suite completed. Pass rate: {summary['pass_rate']}% "
        f"({summary['passed']}/{summary['evaluated']} evaluated, {summary['errors']} error(s)) "
        f"| Time: {summary['total_time_seconds']}s"
    )

    return {"results": results, "summary": summary}
