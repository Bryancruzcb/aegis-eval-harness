"""AegisEval CLI entry point.

Runs the evaluation suite non-interactively (safe for CI), writes a JSON result
file and an HTML dashboard, and exits with a status code suitable for a quality
gate:

  0  success (attack pass rate meets --fail-under, if set)
  1  quality gate failed, or every case errored, or every attack case errored
     (run inconclusive)
  2  configuration error (missing API key, unreadable suite)
"""
import asyncio
import argparse
import json
import logging
import shutil
import sys

import config
from runner import run_suite
from scenarios import SCENARIOS
from reporter import print_terminal_summary, generate_html_report, redact

logger = logging.getLogger("AegisEval.CLI")

YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def configure_logging():
    """Configure logging once, here at the entry point. Keep third-party
    libraries quiet so the console shows AegisEval's own progress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "google_genai", "google", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="AegisEval: Autonomous AI Security & Quality Evaluation Suite"
    )
    parser.add_argument("--tag", default=None,
                        help="Filter test cases by tag (e.g. jailbreak, direct_leak, profanity)")
    parser.add_argument("--category", default=None, choices=["functional", "safety", "security"],
                        help="Filter test cases by category")
    parser.add_argument("--target-provider", default=None, choices=["gemini", "openai", "ollama"],
                        help="API provider for the Target model under test")
    parser.add_argument("--target-model", default=None,
                        help="Model name for the Target (e.g. gemini-2.5-flash)")
    parser.add_argument("--judge-provider", default=None, choices=["gemini", "openai", "ollama"],
                        help="API provider for the Judge evaluator model")
    parser.add_argument("--judge-model", default=None,
                        help="Model name for the Judge evaluator")
    parser.add_argument("--fail-under", type=float, default=None, metavar="PCT",
                        help="Exit non-zero if the attack pass rate (expect=refuse cases) is "
                             "below this percentage (e.g. 80)")
    parser.add_argument("--technique", default=None, help="Filter test cases by technique tag")
    parser.add_argument("--repeats", type=int, default=config.REPEATS_PER_CASE,
                        help="Run each case N times; take-worst aggregation")
    parser.add_argument("--target-temp", type=float, default=config.TARGET_TEMPERATURE,
                        help="Target sampling temperature (repeats vary only when > 0)")
    parser.add_argument("--scenario", default="secret-guardian",
                        choices=sorted(SCENARIOS), help="Which scenario to run")
    parser.add_argument("--full", action="store_true",
                        help="Run the whole benchmark instead of a sample (refusal only)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a minimal 2-per-category sample (refusal only)")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed for the stratified sample")
    parser.add_argument("--refresh-benchmark", action="store_true",
                        help="Ignore the cache and re-download the benchmark")
    parser.add_argument("--fail-over-refusal", type=float, default=None, metavar="PCT",
                        help="Exit non-zero if the control-case failure rate (over-refusal of "
                             "harmless requests) exceeds this percentage (e.g. 20)")
    parser.add_argument("--allow-self-grading", action="store_true",
                        help="Permit the same model to grade its own output (target == judge); "
                             "refused by default because a self-report is not an evaluation")
    parser.add_argument("--include-responses", action="store_true",
                        help="Write full model responses to the report/JSON instead of redacting "
                             "harmful-prompt responses")
    args = parser.parse_args(argv)
    # Also validates config.REPEATS_PER_CASE, which supplies the default: a
    # repeats of 0 would aggregate an empty run list.
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.full and args.smoke:
        parser.error("--full and --smoke are mutually exclusive")
    return args


def load_kwargs_from_args(args) -> dict:
    """Translate the sampling CLI flags into ``load_jbb_cases`` keyword args.

    The flags are user-facing (``--full``/``--smoke``/``--sample-seed``/
    ``--refresh-benchmark``); the loader takes ``mode``/``seed``/``refresh``.
    Neither ``--full`` nor ``--smoke`` means the default stratified sample.
    """
    mode = "full" if args.full else "smoke" if args.smoke else "sample"
    return {"mode": mode, "seed": args.sample_seed, "refresh": args.refresh_benchmark}


def missing_keys(target_provider: str, judge_provider: str) -> list:
    """Return the names of required-but-unset API keys for the chosen providers."""
    providers = {
        target_provider or config.DEFAULT_TARGET_PROVIDER,
        judge_provider or config.DEFAULT_JUDGE_PROVIDER,
    }
    missing = []
    if "gemini" in providers and not config.GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if "openai" in providers and not config.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    # "ollama" needs no key (local, OpenAI-compatible endpoint).
    return missing


def _gate_rate(summary: dict):
    """The rate the quality gate keys off: the attack pass rate when present,
    else the global pass rate. Single source of truth, so the number printed on
    failure can never contradict the exit code."""
    return summary.get("attack_pass_rate", summary["pass_rate"])


def _gate_eval(summary: dict):
    """The evaluated-case count matching ``_gate_rate``'s denominator."""
    return summary.get("attack_evaluated", summary["evaluated"])


def is_self_grading(target_label: str, judge_label: str) -> bool:
    """True when the same model grades its own output — a self-report, not an evaluation."""
    return target_label == judge_label


def decide_exit_code(summary: dict, fail_under, fail_over_refusal=None) -> int:
    """Map a run summary to a process exit code for CI.

    0 = ok, 1 = quality gate failed or every case errored (inconclusive).
    An empty selection (no cases matched) is not a gate failure.
    """
    if summary["total"] > 0 and summary["evaluated"] == 0:
        return 1  # cases existed but none could be evaluated
    if summary.get("attack_total", 0) > 0 and summary.get("attack_evaluated", 1) == 0:
        return 1  # attack suite fully errored (e.g. judge outage)
    gate_rate = _gate_rate(summary)
    gate_eval = _gate_eval(summary)
    if fail_under is not None and gate_eval and gate_rate is not None and gate_rate < fail_under:
        return 1  # quality gate failed
    # A model that refuses everything scores a perfect attack rate; the control
    # half is the only thing that catches it. control_fail_rate is a FRACTION
    # (0-1); fail_over_refusal is a PERCENT — the *100 bridges the units.
    cfr = summary.get("control_fail_rate")
    if (fail_over_refusal is not None and summary.get("control_total", 0) > 0
            and cfr is not None and cfr * 100.0 > fail_over_refusal):
        return 1  # over-refusal ceiling exceeded
    return 0


def redact_harmful_responses(payload: dict) -> dict:
    """Redact model responses for harmful-prompt (``expect == "refuse"``) cases,
    in place, in both the ``response`` field and the assistant turns of the
    transcript.

    The whole harmful split is blanket-redacted, not just the cases the model
    complied with: a redacted refusal is harmless (a truncated refusal), while an
    un-redacted compliance is the exact leak this guard exists to prevent. Non-
    harmful cases (the harmless controls) are left untouched.
    """
    for r in payload.get("results", []):
        if r.get("expect", "refuse") != "refuse":
            continue
        r["response"] = redact(r.get("response"))
        for msg in r.get("transcript") or []:
            if msg.get("role") in ("assistant", "bot"):
                msg["content"] = redact(msg.get("content"))
    return payload


def maybe_redact_responses(payload: dict, args) -> dict:
    """Apply harmful-response redaction before the report/JSON are written,
    unless this is not the refusal scenario or ``--include-responses`` opted out.

    Redaction is ON by default (the flag is off by default): the report is the
    artifact most likely to be shared or screenshotted, so a complied harmful
    response must not ship in it unless the operator explicitly asks for it.
    """
    if args.scenario == "refusal" and not args.include_responses:
        redact_harmful_responses(payload)
    return payload


def _archive(stable_path: str, timestamp: str, suffix: str):
    """Copy a stable output file to a timestamped name so history is kept."""
    stamp = timestamp.replace("-", "").replace(":", "").replace(" ", "_")
    ext = stable_path.rsplit(".", 1)[-1]
    dest = config.OUTPUT_DIR / f"{suffix}_{stamp}.{ext}"
    shutil.copyfile(stable_path, dest)
    return str(dest)


async def main_async() -> int:
    args = parse_args()

    missing = missing_keys(args.target_provider, args.judge_provider)
    if missing:
        print(f"{YELLOW}[Config] Missing required API key(s): {', '.join(missing)}{RESET}")
        print("Set them in a .env file in this directory (copy env.example to .env), e.g.:")
        for name in missing:
            print(f"  {name}=your_key_here")
        print("Or run against a local model with:  --target-provider ollama --judge-provider ollama")
        return 2

    # Labels are built exactly as the runner builds them, so the self-grading
    # check keys off the same provider:model strings the report will show.
    target_label = f"{args.target_provider or config.DEFAULT_TARGET_PROVIDER}:{args.target_model or config.DEFAULT_TARGET_MODEL}"
    judge_label = f"{args.judge_provider or config.DEFAULT_JUDGE_PROVIDER}:{args.judge_model or config.DEFAULT_JUDGE_MODEL}"
    if is_self_grading(target_label, judge_label) and not args.allow_self_grading:
        print(f"{YELLOW}[Config] Refusing to run: the target and the judge are the same model "
              f"({target_label}). A model grading its own output is a self-report, not an "
              f"independent evaluation.{RESET}")
        print("Use a different --judge-model/--judge-provider, or pass --allow-self-grading "
              "to acknowledge the limitation.")
        return 2  # distinct from the quality-gate's exit 1

    print("Initializing AegisEval Suite...")

    payload = await run_suite(
        tag_filter=args.tag,
        category_filter=args.category,
        technique_filter=args.technique,
        repeats=args.repeats,
        target_provider=args.target_provider,
        target_model=args.target_model,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        target_temperature=args.target_temp,
        scenario=SCENARIOS[args.scenario],
        load_kwargs=load_kwargs_from_args(args),
    )

    if "error" in payload:
        print(f"{RED}Run failed:{RESET} {payload['error']}")
        return 2

    summary = payload["summary"]

    # Redact harmful-prompt responses (refusal scenario, unless --include-responses)
    # BEFORE either artifact is written, so neither the JSON nor the HTML ships them.
    maybe_redact_responses(payload, args)

    # Persist raw results (stable name + timestamped archive).
    raw_path = config.OUTPUT_DIR / "run_results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print_terminal_summary(payload)

    html_path = generate_html_report(payload)

    if summary["total"] > 0:
        _archive(str(raw_path), summary["timestamp"], "run_results")
        _archive(html_path, summary["timestamp"], "report")

    print(f"✔ Raw results saved to: {raw_path}")
    print(f"✔ HTML dashboard generated at: {html_path}")

    if summary["errors"]:
        print(f"{YELLOW}⚠ {summary['errors']} case(s) errored (infrastructure issue, not a model failure).{RESET}")

    # Exit-code policy for CI.
    exit_code = decide_exit_code(summary, args.fail_under, args.fail_over_refusal)
    if exit_code != 0:
        cfr = summary.get("control_fail_rate")
        over_refusing = (
            args.fail_over_refusal is not None and summary.get("control_total", 0) > 0
            and cfr is not None and cfr * 100.0 > args.fail_over_refusal
        )
        under_performing = (
            args.fail_under is not None and _gate_eval(summary)
            and _gate_rate(summary) is not None and _gate_rate(summary) < args.fail_under
        )
        if summary["total"] > 0 and summary["evaluated"] == 0:
            print(f"{RED}Run inconclusive: every case errored.{RESET}")
        elif summary.get("attack_total", 0) > 0 and summary.get("attack_evaluated", 1) == 0:
            print(f"{RED}Run inconclusive: every attack case errored (judge/target outage).{RESET}")
        else:
            if under_performing:
                print(f"{RED}Quality gate failed: attack pass rate {_gate_rate(summary)}% "
                      f"< {args.fail_under}%.{RESET}")
            if over_refusing:
                print(f"{RED}Over-refusal gate failed: control failure rate "
                      f"{round(cfr * 100.0, 1)}% > {args.fail_over_refusal}% "
                      f"(the model refuses harmless requests).{RESET}")
    return exit_code


def _force_utf8_output():
    """Windows consoles default to cp1252, which can't encode the ✔/✘/⚠ glyphs
    used in the report. Switch stdout/stderr to UTF-8 so output never crashes."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def main():
    _force_utf8_output()
    configure_logging()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
