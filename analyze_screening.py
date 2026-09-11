"""Summarize a completed paired screening experiment without exporting response text."""
import argparse
import json
import random
from pathlib import Path

import calibrate
from compare_graders import summarize


def diagnostics(records):
    counts = dict.fromkeys([
        "screen_false_positives_fixed", "screen_true_positives_lost",
        "screen_false_negatives_fixed", "screen_true_negatives_lost",
        "shared_judge_false_positives", "shared_judge_false_negatives",
        "new_judge_errors", "shared_judge_errors", "screen_errors",
    ], 0)
    for record in records:
        actual = record["actual"]
        before = record["variants"]["two_stage"]
        after = record["variants"]["judge_only"]
        if "error" in before:
            counts["shared_judge_errors" if before["stage"] == "judge" else "screen_errors"] += 1
            continue
        if "error" in after:
            counts["new_judge_errors"] += 1
            continue
        b, a = before["prediction"], after["prediction"]
        if before["stage"] == "judge":
            if b and not actual:
                counts["shared_judge_false_positives"] += 1
            elif not b and actual:
                counts["shared_judge_false_negatives"] += 1
        elif b != a:
            key = ("screen_false_positives_fixed" if not actual else "screen_true_positives_lost") if b else (
                "screen_false_negatives_fixed" if actual else "screen_true_negatives_lost")
            counts[key] += 1
    return counts


def paired_intervals(records, *, draws=2000):
    names = ("two_stage", "judge_only")
    common = [r for r in records if all("prediction" in r["variants"][name] for name in names)]
    classes = [[r for r in common if r["actual"] is label] for label in (False, True)]
    if not all(classes):
        return {"unavailable": "Both label classes are required"}
    samples = {name: {metric: [] for metric in ("mcc", "fpr", "fnr")}
               for name in (*names, "judge_only_minus_two_stage")}
    rng = random.Random(20260907)
    for _ in range(draws):
        sample = [row for group in classes for row in rng.choices(group, k=len(group))]
        metrics = {name: calibrate.agreement_stats(calibrate.confusion([
            (r["variants"][name]["prediction"], r["actual"]) for r in sample])) for name in names}
        for metric in ("mcc", "fpr", "fnr"):
            for name in names:
                samples[name][metric].append(metrics[name][metric])
            samples["judge_only_minus_two_stage"][metric].append(
                metrics["judge_only"][metric] - metrics["two_stage"][metric])
    result = {"method": "Paired class-stratified percentile bootstrap, 95% intervals",
              "draws": draws, "seed": 20260907,
              "scope": "Conditional on this balanced cohort; one row per normalized exact request. Semantic request families may remain correlated."}
    for name, values in samples.items():
        result[name] = {metric: [round(sorted(v)[int(draws * .025)], 4),
                                 round(sorted(v)[min(draws - 1, int(draws * .975))], 4)]
                        for metric, v in values.items()}
    return result


def analyze(payload):
    if payload["status"] != "complete" or not payload.get("screening_ablation"):
        raise ValueError("A completed screening-ablation artifact is required")
    result = {key: value for key, value in payload.items() if key != "records"}
    result["summary"] = summarize(payload["records"], ["two_stage", "judge_only"])
    result["diagnostics"] = diagnostics(payload["records"])
    if payload.get("split") == "jailjudge-id-validation":
        result["uncertainty"] = paired_intervals(payload["records"])
    for name, values in result["summary"]["variants"].items():
        values["estimated_route_seconds"] = round(sum(
            r["variants"][name]["seconds"] for r in payload["records"]), 3)
    result["cost_note"] = "Route counts/timing are estimates from shared judge outputs. Unique evaluations exclude internal retry attempts."
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "diagnostics": result["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
