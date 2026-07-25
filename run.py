"""AegisEval CLI entry point.

Runs the evaluation suite non-interactively (safe for CI), writes a JSON result
file and an HTML dashboard, and exits with a status code suitable for a quality
gate:

  0  success (pass rate meets --fail-under, if set)
  1  quality gate failed, or every case errored (run inconclusive)
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
from reporter import print_terminal_summary, generate_html_report

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
                        help="Exit non-zero if the pass rate is below this percentage (e.g. 80)")
    parser.add_argument("--technique", default=None, help="Filter test cases by technique tag")
    parser.add_argument("--repeats", type=int, default=config.REPEATS_PER_CASE,
                        help="Run each case N times; take-worst aggregation")
    parser.add_argument("--target-temp", type=float, default=config.TARGET_TEMPERATURE,
                        help="Target sampling temperature (repeats vary only when > 0)")
    return parser.parse_args(argv)


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


def decide_exit_code(summary: dict, fail_under) -> int:
    """Map a run summary to a process exit code for CI.

    0 = ok, 1 = quality gate failed or every case errored (inconclusive).
    An empty selection (no cases matched) is not a gate failure.
    """
    if summary["total"] > 0 and summary["evaluated"] == 0:
        return 1  # cases existed but none could be evaluated
    if summary.get("attack_total", 0) > 0 and summary.get("attack_evaluated", 1) == 0:
        return 1  # attack suite fully errored (e.g. judge outage)
    gate_rate = summary.get("attack_pass_rate", summary["pass_rate"])
    gate_eval = summary.get("attack_evaluated", summary["evaluated"])
    if fail_under is not None and gate_eval and gate_rate is not None and gate_rate < fail_under:
        return 1  # quality gate failed
    return 0


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
    )

    if "error" in payload:
        print(f"{RED}Run failed:{RESET} {payload['error']}")
        return 2

    summary = payload["summary"]

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
    exit_code = decide_exit_code(summary, args.fail_under)
    if exit_code != 0:
        if summary["total"] > 0 and summary["evaluated"] == 0:
            print(f"{RED}Run inconclusive: every case errored.{RESET}")
        elif summary.get("attack_total", 0) > 0 and summary.get("attack_evaluated", 1) == 0:
            print(f"{RED}Run inconclusive: every attack case errored (judge/target outage).{RESET}")
        else:
            gate_rate = summary.get("attack_pass_rate", summary["pass_rate"])
            print(f"{RED}Quality gate failed: attack pass rate {gate_rate}% < {args.fail_under}%.{RESET}")
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
