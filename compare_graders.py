"""Compare frozen refusal graders on identical JBB rows without saving response text."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time
from urllib.request import urlopen

import calibrate
import config
from benchmarks import jbb
from evaluators import run_llm_judge_eval_conversation
from graders import Screen


def summarize(records, names):
    common = [r for r in records if all(
        "prediction" in r["variants"].get(name, {}) for name in names)]
    variants = {}
    for name in names:
        values = [r["variants"][name] for r in records if name in r["variants"]]
        pairs = [(r["variants"][name]["prediction"], r["actual"]) for r in common]
        matrix = calibrate.confusion(pairs)
        judge_times = [v["seconds"] for v in values if v["stage"] == "judge"]
        variants[name] = {
            "scored": sum("prediction" in v for v in values),
            "errors": sum("error" in v for v in values),
            "stage1_decisions": sum(v["stage"] == "screen" and "prediction" in v for v in values),
            "judge_calls": len(judge_times),
            "judge_mean_seconds": statistics.mean(judge_times) if judge_times else None,
            "common_confusion": matrix,
            "common_metrics": calibrate.agreement_stats(matrix) if pairs else None,
            "stage1_confusion": calibrate.confusion([
                (r["variants"][name]["prediction"], r["actual"]) for r in records
                if r["variants"].get(name, {}).get("stage") == "screen"
                and "prediction" in r["variants"][name]]),
        }
    return {"common_resolved": len(common), "variants": variants}


PROVENANCE_VERSION = "aegis_eval.1"

IDENTITY_MODULES = {
    "evaluators.py": "evaluators",
    "providers.py": "providers",
    "config.py": "config",
    "calibrate.py": "calibrate",
    "graders.py": "graders",
    "jailjudge_validation.py": "jailjudge_validation",
    "benchmarks/jbb.py": "benchmarks.jbb",
}


def module_digest(modname: str) -> str:
    spec = importlib.util.find_spec(modname)
    if spec is None or not spec.origin:
        raise ValueError(f"Identity module missing: {modname}")
    return hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()


def register_snapshot_imports():
    import evaluators as evaluators_mod
    import graders as graders_mod
    sys.modules["evaluators"] = evaluators_mod
    sys.modules["graders"] = graders_mod


def load_variant(name, path):
    register_snapshot_imports()
    spec = importlib.util.spec_from_file_location(f"comparison_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RefusalGrader()


async def evaluate(grader, goal, response, provider, model, timeout, *, judge_only=False):
    start = time.perf_counter()
    case = {"expect": "refuse"}
    stage = "screen"
    try:
        screen = (Screen(decision="judge", reason="screening ablation") if judge_only
                  else grader.screen(case, response, goal, is_final=True))
        stage = "screen" if screen.decision != "judge" else "judge"
        judged = None
        if stage == "judge":
            judged = await asyncio.wait_for(run_llm_judge_eval_conversation(
                [{"role": "user", "content": goal},
                 {"role": "assistant", "content": response}], "",
                provider=provider, model=model,
                system_instruction=grader.judge_system_prompt(case),
                parse=grader.parse_judgment, response_schema=grader.judge_schema), timeout)
        verdict = grader.verdict(case, screen, judged)
        return {"prediction": verdict.status == "fail", "stage": stage,
                "seconds": round(time.perf_counter() - start, 4)}
    except Exception as exc:
        return {"error": type(exc).__name__, "stage": stage,
                "seconds": round(time.perf_counter() - start, 4)}


async def evaluate_screening_modes(grader, goal, response, provider, model, timeout):
    screened = await evaluate(grader, goal, response, provider, model, timeout)
    judged = (dict(screened) if screened["stage"] == "judge" else
              await evaluate(grader, goal, response, provider, model, timeout, judge_only=True))
    return {"two_stage": screened, "judge_only": judged}


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def model_identity(provider, model):
    if provider != "ollama":
        return {"model": model}
    url = config.OLLAMA_BASE_URL.rstrip("/").removesuffix("/v1") + "/api/tags"
    with urlopen(url, timeout=10) as response:
        models = json.load(response)["models"]
    for entry in models:
        if entry["name"] == model:
            return {"model": model, "digest": entry["digest"], "details": entry.get("details", {})}
    raise ValueError(f"Ollama model is not installed: {model}")


def execution_identity():
    return {"provenance_version": PROVENANCE_VERSION,
            "harness_sha256": module_digest("compare_graders"),
            "files": {name: module_digest(mod) for name, mod in IDENTITY_MODULES.items()},
            "python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("openai", "google-genai", "pydantic", "tenacity")},
            "runtime": {"ollama_endpoint_sha256": hashlib.sha256(config.OLLAMA_BASE_URL.encode()).hexdigest(),
                        "max_retries": config.MAX_RETRIES, "retry_max_wait": config.RETRY_MAX_WAIT,
                        "judge_temperature": 0.0}}


def validate_resume(prior, expected, rows):
    for key, value in expected.items():
        if key not in {"started_utc", "status", "records"} and prior.get(key) != value:
            raise ValueError(f"Cannot resume: {key} changed")
    if len(prior["records"]) > len(rows):
        raise ValueError("Cannot resume: too many saved rows")
    for index, record in enumerate(prior["records"]):
        if record["row"] != index or record["row_sha256"] != row_hash(rows[index]):
            raise ValueError(f"Cannot resume: row {index} changed")
    return prior


async def compare(args):
    jailjudge_path = getattr(args, "jailjudge_id", None)
    if jailjudge_path:
        from urllib.parse import urlparse
        if args.provider != "ollama" or urlparse(config.OLLAMA_BASE_URL).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("JAILJUDGE evaluation requires local Ollama under its dataset license")
        if args.held_out or args.limit:
            raise ValueError("JAILJUDGE validation uses a fixed 200-row selection")
    variants, sources = {}, {}
    for item in args.variant:
        name, raw_path = item.split("=", 1)
        if name in variants:
            raise ValueError(f"Duplicate variant: {name}")
        path = Path(raw_path).resolve()
        variants[name] = load_variant(name, path)
        sources[name] = {"path": str(path),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "prompt_sha256": hashlib.sha256(variants[name].judge_system_prompt(
                             {"expect": "refuse"}).encode()).hexdigest()}
    if args.screening_ablation:
        grader = next(iter(variants.values()))
        source = next(iter(sources.values()))
        variants = {"two_stage": grader, "judge_only": grader}
        sources = {name: dict(source, screening_mode=name) for name in variants}
    rows = calibrate._fetch_rows(refresh=False)
    dataset = {"split": "held-out" if args.held_out else "dev",
               "seed": calibrate.SPLIT_SEED, "dataset_commit": jbb.JBB_COMMIT,
               "dataset_sha256": jbb.SHA256[jbb.JUDGE_URL]}
    if jailjudge_path:
        from jailjudge_validation import load
        chosen, dataset = load(jailjudge_path, rows)
    else:
        dev, held = calibrate._split(rows, seed=calibrate.SPLIT_SEED)
        chosen = held if args.held_out else dev
    if args.limit:
        chosen = chosen[:args.limit]
    names = list(variants)
    payload = {"started_utc": datetime.now(timezone.utc).isoformat(),
               **dataset,
               "provider": args.provider, "model": args.model, "temperature": 0,
               "model_identity": model_identity(args.provider, args.model),
               "execution_identity": execution_identity(),
               "screening_ablation": args.screening_ablation,
               "timeout_seconds": args.timeout,
               "comparison_semantics": ("Shared judge output; route costs are estimates"
                                         if args.screening_ablation else "Independent variant calls"),
               "selected_rows": len(chosen), "sources": sources,
               "status": "running", "records": []}
    if args.resume:
        payload = validate_resume(json.loads(args.output.read_text(encoding="utf-8")), payload, chosen)
        if payload["status"] == "complete":
            print(json.dumps(payload["summary"], indent=2))
            return 0 if all(v["errors"] == 0 for v in payload["summary"]["variants"].values()) else 1
    elif args.output.exists():
        raise ValueError("Output already exists; use --resume or a new output path")
    payload["status"] = "running"
    save(args.output, payload)
    consecutive_errors = 0
    start_index = len(payload["records"])
    for index in range(start_index, len(chosen)):
        row = chosen[index]
        record = {"row": index, "actual": calibrate._actual_jailbroken(row),
                  "row_sha256": row_hash(row),
                  "variants": {}}
        # Rotate order to avoid always charging one variant for warm-up or drift.
        order = names[index % len(names):] + names[:index % len(names)]
        if args.screening_ablation:
            record["variants"] = await evaluate_screening_modes(
                grader, row["goal"], row[calibrate.RESPONSE_FIELD],
                args.provider, args.model, args.timeout)
            record["unique_judge_evaluations"] = 1
        else:
            for name in order:
                record["variants"][name] = await evaluate(
                    variants[name], row["goal"], row[calibrate.RESPONSE_FIELD],
                    args.provider, args.model, args.timeout)
            record["unique_judge_evaluations"] = sum(v["stage"] == "judge" for v in record["variants"].values())
        payload["records"].append(record)
        all_failed = ("error" in record["variants"]["judge_only"] if args.screening_ablation else
                      all("error" in v for v in record["variants"].values()))
        consecutive_errors = consecutive_errors + 1 if all_failed else 0
        payload["summary"] = summarize(payload["records"], names)
        payload["unique_judge_evaluations"] = sum(r["unique_judge_evaluations"] for r in payload["records"])
        save(args.output, payload)
        print(f"{payload['split']} {index + 1}/{len(chosen)} common scored="
              f"{payload['summary']['common_resolved']}", flush=True)
        if consecutive_errors >= 3:
            payload["status"] = "stopped_after_three_failed_rows"
            break
    else:
        payload["status"] = "complete"
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))
    return 0 if payload["status"] == "complete" and all(
        v["errors"] == 0 for v in payload["summary"]["variants"].values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, metavar="NAME=PYTHON_FILE")
    parser.add_argument("--provider", choices=["ollama", "gemini", "openai"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--jailjudge-id", type=Path,
                        help="Pinned JAILJUDGE ID file, local Ollama only; fixed 200-row validation.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screening-ablation", action="store_true",
                        help="Compare screening against judge-only with shared judge outputs; one variant required.")
    parser.add_argument("--resume", action="store_true", help="Continue a matching checkpoint without repeating completed rows.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.screening_ablation and len(args.variant) != 1:
        parser.error("--screening-ablation requires exactly one --variant")
    return asyncio.run(compare(args))


if __name__ == "__main__":
    raise SystemExit(main())
