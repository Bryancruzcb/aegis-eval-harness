"""Coordinates async execution of the test suite.

Each case runs: target query -> the scenario grader's cheap screen -> (if the
screen is undecided) the LLM judge. Every grading decision belongs to the
scenario's ``Grader``; the runner only sequences turns, handles provider
failures, and shapes the result. Every result carries a ``status`` of ``pass``,
``fail``, or ``error``:

  * pass / fail  -> the model produced an answer and it was graded
  * error        -> the answer could not be obtained or graded (429, timeout,
                    unparseable judge output, ...). Errors are reported
                    separately and excluded from the pass rate, so an outage
                    never masquerades as a safety failure.
"""
import asyncio
import time
import logging

from target import query_target_conversation
from evaluators import run_llm_judge_eval_conversation, JudgeParseError
from providers import ProviderError
from scenarios import SECRET_GUARDIAN
import config

logger = logging.getLogger("AegisEval.Runner")


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

    This is a whitelist: a key a case carries but this function does not name is
    dropped from the result. Scenario-specific metadata therefore rides in the
    case's optional ``extra`` dict, which is copied through verbatim (empty dict
    when absent). It is nested rather than flattened on purpose — flattening
    unknown keys would let a case silently overwrite a result field.
    """
    turns = test_case.get("turns") or [test_case.get("prompt", "")]
    return {
        "id": test_case.get("id"), "category": test_case.get("category"),
        "prompt": test_case.get("prompt") or "\n".join(turns),
        "expected_criteria": test_case.get("expected_criteria"),
        "description": test_case.get("description", ""), "tags": test_case.get("tags", []),
        "expect": test_case.get("expect", "refuse"),
        "technique": test_case.get("technique", "unspecified"),
        "extra": test_case.get("extra", {}),
    }


def _eval_type(judged):
    """How the case was decided. The runner owns this, not the Verdict.

    ``"deterministic"`` when the grader's screen settled it without the judge,
    ``"llm_judge"`` when the judge did. Keep these exact strings:
    ``aggregate_repeats`` tie-breaks on ``"deterministic"``, so a new value would
    silently degrade ``--repeats``.
    """
    return "deterministic" if judged is None else "llm_judge"


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


async def run_single_test(test_case, semaphore, *, scenario=SECRET_GUARDIAN,
                          query_fn=None, judge_fn=None,
                          target_provider=None, target_model=None,
                          judge_provider=None, judge_model=None, target_temperature=None,
                          attacker=None, max_turns=6):
    """Run one case end to end and return a result dict.

    Sequencing, provider-failure handling, and the result shape live here; every
    grading decision belongs to ``scenario.grader``. Mid-loop only a ``"fail"``
    screen is actionable — ``"pass"``/``"judge"`` mean "keep going" until the
    final turn — so the loop's exit screen is the one that decides the case.

    When ``attacker`` is supplied the scripted turns are ignored and an LLM
    improvises each turn instead; the routing happens first so a promptless
    adaptive shell (no ``prompt``/``turns``) never reaches the scripted loader.
    """
    if attacker is not None:
        return await _run_adaptive(test_case, semaphore, scenario=scenario, query_fn=query_fn,
                                   judge_fn=judge_fn, target_provider=target_provider,
                                   target_model=target_model, judge_provider=judge_provider,
                                   judge_model=judge_model, target_temperature=target_temperature,
                                   attacker=attacker, max_turns=max_turns)
    query_fn = query_fn or query_target_conversation
    judge_fn = judge_fn or run_llm_judge_eval_conversation
    grader = scenario.grader
    turns = test_case.get("turns") or [test_case["prompt"]]
    base = _base_fields(test_case)
    tc_id = base["id"]
    async with semaphore:
        logger.info(f"Starting {tc_id} ({base['category']})")
        transcript, running_bot, attacker_text = [], "", ""
        start = time.time()
        for i, user_msg in enumerate(turns):
            is_final = (i == len(turns) - 1)
            transcript.append({"role": "user", "content": user_msg})
            attacker_text += "\n" + user_msg
            try:
                reply = await query_fn(transcript, provider=target_provider,
                                       model=target_model, temperature=target_temperature,
                                       system_prompt=scenario.system_prompt)
            except ProviderError as e:
                latency = round(time.time() - start, 2)
                logger.warning(f"{tc_id}: target error: {e}")
                _log_done(tc_id, "error")
                return _result(base, response=None, score=None, reasoning=str(e), passed=False,
                               status="error", eval_type="target_error", latency=latency,
                               transcript=transcript)
            transcript.append({"role": "assistant", "content": reply})
            running_bot += "\n" + reply
            # Running concatenations, not just this turn: a leak in turn 1 still
            # counts in turn 3, and a secret the attacker supplied earlier must
            # not be scored as a leak when the bot repeats it back later.
            screen = grader.screen(test_case, running_bot, attacker_text, is_final=is_final)
            if screen.decision == "fail":
                latency = round(time.time() - start, 2)
                v = grader.verdict(test_case, screen, None)
                _log_done(tc_id, "fail (deterministic)", v.score)
                return _result(base, response=transcript[-1]["content"], score=v.score,
                               reasoning=v.reasoning, passed=False, status="fail",
                               eval_type=_eval_type(None), latency=latency,
                               transcript=transcript, false_positive=v.false_positive)
        latency = round(time.time() - start, 2)
        if screen.decision == "pass":
            v = grader.verdict(test_case, screen, None)
            _log_done(tc_id, "pass (deterministic-only)", v.score)
            return _result(base, response=transcript[-1]["content"], score=v.score,
                           reasoning=v.reasoning, passed=True, status="pass",
                           eval_type=_eval_type(None), latency=latency, transcript=transcript,
                           false_positive=v.false_positive)
        logger.info(f"Deterministic checks passed for {tc_id}. Invoking LLM Judge...")
        # A grader whose judge answers a non-default schema (e.g. the refusal
        # grader) supplies its own parser via a ``parse_judgment`` attribute; the
        # kwarg is passed only when present, so a grader without one — and the
        # test doubles that stub ``judge_fn`` without a ``parse=`` parameter —
        # keep the judge's built-in default.
        judge_kwargs = {}
        grader_parse = getattr(grader, "parse_judgment", None)
        if grader_parse is not None:
            judge_kwargs["parse"] = grader_parse
        # ...and the matching Gemini response schema. Passed only alongside a custom
        # parser, so a grader with neither (and the test doubles that stub judge_fn
        # without these params) keep the judge's built-in secret-guardian schema.
        grader_schema = getattr(grader, "judge_schema", None)
        if grader_schema is not None:
            judge_kwargs["response_schema"] = grader_schema
        try:
            judged = await judge_fn(transcript, base["expected_criteria"],
                                    provider=judge_provider, model=judge_model,
                                    system_instruction=grader.judge_system_prompt(test_case),
                                    **judge_kwargs)
        except (ProviderError, JudgeParseError) as e:
            # Return before asking for a verdict: with no judge answer there is
            # nothing to grade, and ``verdict`` requires a non-None ``judged``
            # for a "judge" screen.
            logger.warning(f"{tc_id}: judge error: {e}")
            _log_done(tc_id, "error")
            return _result(base, response=transcript[-1]["content"], score=None, reasoning=str(e),
                           passed=False, status="error", eval_type="judge_error",
                           latency=latency, transcript=transcript)
        v = grader.verdict(test_case, screen, judged)
        _log_done(tc_id, v.status, v.score)
        return _result(base, response=transcript[-1]["content"], score=v.score,
                       reasoning=v.reasoning, passed=(v.status == "pass"), status=v.status,
                       eval_type=_eval_type(judged), latency=latency, transcript=transcript)


async def _run_adaptive(test_case, semaphore, *, scenario, query_fn, judge_fn,
                        target_provider, target_model, judge_provider, judge_model,
                        target_temperature, attacker, max_turns):
    """The adaptive path: an LLM attacker improvises each turn until it either
    triggers a deterministic leak or exhausts the turn budget.

    Mirrors the scripted body's accumulation, judge escalation, and result shape
    (reusing ``_result``/``_base_fields``/``_log_done``/``grader.*``). Two things
    diverge from scripted: the turns come from ``attacker.next_turn`` rather than
    the case, and a leak carries per-turn attribution (``break_turn``/
    ``winning_tactic``/``tactics``). The message dict handed to ``query_fn`` is
    role/content ONLY — tactics ride in a parallel list, never in the transcript,
    because the provider rejects unknown message properties.
    """
    query_fn = query_fn or query_target_conversation
    judge_fn = judge_fn or run_llm_judge_eval_conversation
    grader = scenario.grader
    base = _base_fields(test_case)
    tc_id = base["id"]
    async with semaphore:
        logger.info(f"Starting {tc_id} ({base['category']}) [adaptive]")
        transcript, tactics, running_bot, attacker_text = [], [], "", ""
        start = time.time()
        for turn in range(1, max_turns + 1):
            user_msg, tactic = await attacker.next_turn(transcript, test_case)
            transcript.append({"role": "user", "content": user_msg})   # role/content ONLY
            tactics.append(tactic)
            attacker_text += "\n" + user_msg
            try:
                reply = await query_fn(transcript, provider=target_provider, model=target_model,
                                       temperature=target_temperature,
                                       system_prompt=scenario.system_prompt)
            except ProviderError as e:
                latency = round(time.time() - start, 2)
                logger.warning(f"{tc_id}: target error: {e}")
                _log_done(tc_id, "error")
                return _result(base, response=None, score=None, reasoning=str(e), passed=False,
                               status="error", eval_type="target_error", latency=latency,
                               transcript=transcript)
            transcript.append({"role": "assistant", "content": reply})
            running_bot += "\n" + reply
            is_final = (turn == max_turns)
            # Running concatenations, exactly as the scripted body: a leak in an
            # earlier turn still counts, and a secret the attacker supplied is not
            # scored as a leak when the bot echoes it back.
            screen = grader.screen(test_case, running_bot, attacker_text, is_final=is_final)
            if screen.decision == "fail":                              # deterministic leak
                latency = round(time.time() - start, 2)
                v = grader.verdict(test_case, screen, None)
                _log_done(tc_id, f"{v.status} (deterministic)", v.score)
                r = _result(base, response=reply, score=v.score, reasoning=v.reasoning,
                            passed=(v.status == "pass"), status=v.status,
                            eval_type=_eval_type(None), latency=latency, transcript=transcript,
                            false_positive=v.false_positive)
                r["break_turn"], r["winning_tactic"] = turn, tactic
                r["tactics"] = tactics          # per-turn audit trail (spec); parallel, NOT in transcript
                return r
        # Survived the budget with no deterministic leak -> the judge grades the
        # transcript; a survivor is graded, never auto-passed.
        latency = round(time.time() - start, 2)
        logger.info(f"Adaptive attempt {tc_id} survived {max_turns} turn(s). Invoking LLM Judge...")
        try:
            judged = await judge_fn(transcript, base["expected_criteria"],
                                    provider=judge_provider, model=judge_model,
                                    system_instruction=grader.judge_system_prompt(test_case))
        except (ProviderError, JudgeParseError) as e:
            logger.warning(f"{tc_id}: judge error: {e}")
            _log_done(tc_id, "error")
            return _result(base, response=running_bot, score=None, reasoning=str(e), passed=False,
                           status="error", eval_type="judge_error", latency=latency,
                           transcript=transcript)
        v = grader.verdict(test_case, screen, judged)
        _log_done(tc_id, v.status, v.score)
        r = _result(base, response=transcript[-1]["content"], score=v.score, reasoning=v.reasoning,
                    passed=(v.status == "pass"), status=v.status, eval_type=_eval_type(judged),
                    latency=latency, transcript=transcript, false_positive=v.false_positive)
        r["tactics"] = tactics              # per-turn audit trail on survivors too (no break_turn)
        return r


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


def build_summary(results, target_label, judge_label, total_time_seconds, timestamp=None,
                  *, scenario_name="secret-guardian") -> dict:
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
    # Control = every case the model was NOT supposed to resist. For
    # secret-guardian that is the benign traps (a failure means the grader
    # over-flagged); for refusal it is the harmless requests (a failure means
    # the model over-refused). Attack + control always partition the results.
    control = [r for r in results if r.get("expect", "refuse") != "refuse"]
    control_failed = sum(1 for r in control if r.get("status") == "fail")
    control_eval = sum(1 for r in control if r.get("status") in ("pass", "fail"))
    control_fail_rate = round(control_failed / control_eval, 4) if control_eval else None

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

    # Attribution over deterministic leaks (results carrying a break_turn). Always
    # emitted — empty list / empty dict when nothing leaked — per the contract above.
    leaks = [r for r in results if r.get("break_turn") is not None]
    turns_to_break = [r["break_turn"] for r in leaks]
    winning_tactic_counts = {}
    for r in leaks:
        t = r.get("winning_tactic", "unknown")
        winning_tactic_counts[t] = winning_tactic_counts.get(t, 0) + 1

    return {
        "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": target_label,
        "judge": judge_label,
        "scenario": scenario_name,
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
        "control_total": len(control), "control_failed": control_failed,
        "control_fail_rate": control_fail_rate,
        "turns_to_break": turns_to_break,
        "winning_tactic_counts": winning_tactic_counts,
    }


async def run_case_repeated(test_case, semaphore, *, repeats, query_fn, judge_fn,
                            scenario=SECRET_GUARDIAN,
                            target_provider=None, target_model=None, judge_provider=None,
                            judge_model=None, target_temperature=None,
                            attacker=None, max_turns=6):
    """Run one case ``repeats`` times and collapse with take-worst aggregation.

    A ``None`` ``query_fn``/``judge_fn`` is forwarded straight into
    ``run_single_test``, whose own defaults then bind the real target/judge.
    ``attacker``/``max_turns`` are forwarded the same way, so each repeat runs the
    adaptive path when an attacker is supplied.
    """
    tasks = [run_single_test(test_case, semaphore, scenario=scenario,
                             query_fn=query_fn, judge_fn=judge_fn,
                             target_provider=target_provider, target_model=target_model,
                             judge_provider=judge_provider, judge_model=judge_model,
                             target_temperature=target_temperature,
                             attacker=attacker, max_turns=max_turns)
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
                    judge_model=None, target_temperature=None, query_fn=None, judge_fn=None,
                    *, scenario=SECRET_GUARDIAN, load_kwargs=None,
                    attacker=None, max_turns=6):
    """Load, filter, and run all test cases; return results plus a summary.

    The scenario supplies both the cases and the grading, so a different
    scenario changes what runs and how it is scored without touching this
    function. ``load_kwargs`` is forwarded verbatim to ``scenario.load_cases``
    (the CLI uses it to pass the refusal benchmark's sampling options); a loader
    that does not recognise a key must ignore it.

    Loading AND the technique/category filters run inside one ``try``: a loader
    that raises, or that returns malformed cases missing ``technique``/
    ``category``, degrades to the ``{"error": ...}`` payload rather than crashing
    the suite with a traceback.
    """
    try:
        cases = scenario.load_cases(**(load_kwargs or {}))
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
    except Exception as e:
        logger.error(f"Failed to load cases for scenario '{scenario.name}': {e}")
        return {"error": str(e)}
    logger.info(f"Selected {len(cases)} test case(s).")
    target_label = f"{target_provider or config.DEFAULT_TARGET_PROVIDER}:{target_model or config.DEFAULT_TARGET_MODEL}"
    judge_label = f"{judge_provider or config.DEFAULT_JUDGE_PROVIDER}:{judge_model or config.DEFAULT_JUDGE_MODEL}"
    if not cases:
        if not technique_filter:
            logger.warning("No test cases match the selection criteria.")
        return {"results": [],
                "summary": build_summary([], target_label, judge_label, 0.0,
                                         scenario_name=scenario.name)}
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    start = time.time()
    raw_results = await asyncio.gather(*[
        run_case_repeated(c, semaphore, repeats=repeats, query_fn=query_fn, judge_fn=judge_fn,
                          scenario=scenario,
                          target_provider=target_provider, target_model=target_model,
                          judge_provider=judge_provider, judge_model=judge_model,
                          target_temperature=target_temperature,
                          attacker=attacker, max_turns=max_turns)
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
    summary = build_summary(results, target_label, judge_label, total_time,
                            scenario_name=scenario.name)
    logger.info(
        f"Suite completed. Pass rate: {summary['pass_rate']}% "
        f"({summary['passed']}/{summary['evaluated']} evaluated, {summary['errors']} error(s)) "
        f"| Time: {summary['total_time_seconds']}s"
    )
    return {"results": results, "summary": summary}
