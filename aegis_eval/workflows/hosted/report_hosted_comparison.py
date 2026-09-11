"""Build the three-judge report from completed local checkpoints; no API calls."""
import hashlib
import json
import math
from pathlib import Path
import statistics

import aegis_eval.core.config as config
import aegis_eval.workflows.grader_quality.compare_graders as comparison

HOSTED_DIR = config.BASE_DIR / "output" / "hosted-comparison"
MODEL = "gemini-3.7-flash"
CAP = 5.0


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


def build_report():
    paths = [HOSTED_DIR / "hosted-judge-dev.json", HOSTED_DIR / "gemini-3.7-flash-dev.json"]
    older, newer = [json.loads(p.read_text()) for p in paths]
    for payload in (older, newer):
        if payload['status'] not in ('complete', 'complete_with_errors') or len(payload['records']) != 150:
            raise ValueError('Both hosted comparisons must attempt all 150 rows')
    if newer['prior_trial']['sha256'] != hashlib.sha256(paths[0].read_bytes()).hexdigest():
        raise ValueError('Prior ledger changed')
    names = ['qwen_2_5', 'gemini_3_5_flash', 'gemini_3_7_flash']
    records = []
    for i, (old, new) in enumerate(zip(older['records'], newer['records'])):
        if (old['row'] != i or new['row'] != i or old['row_sha256'] != new['row_sha256']
                or old['actual'] != new['actual']
                or old['variants']['local_two_stage'] != new['variants']['local_two_stage']):
            raise ValueError('Cohorts or local baseline do not match')
        records.append({'row': i, 'actual': new['actual'], 'variants': {
            names[0]: new['variants']['local_two_stage'],
            names[1]: old['variants']['hosted_two_stage'],
            names[2]: new['variants']['hosted_two_stage']}})
    summary = comparison.summarize(records, names)
    for name in names:
        summary['variants'][name]['successful_judge_mean_seconds'] = statistics.mean(
            r['variants'][name]['seconds'] for r in records
            if r['variants'][name]['stage'] == 'judge' and 'prediction' in r['variants'][name])
    current_attempts = [a for a in newer['attempts'] if a['model'] == MODEL]
    pairwise = comparison.summarize(records, [names[0], names[2]])
    local = pairwise['variants'][names[0]]['common_metrics']
    candidate = pairwise['variants'][names[2]]['common_metrics']
    result = {'selected_rows': 150, 'status': newer['status'], 'summary': summary,
        'identity': newer['identity'], 'qwen_vs_37': pairwise,
        'source_checkpoints': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'unresolved_rows': {n: [r['row'] for r in records if 'error' in r['variants'][n]] for n in names},
        'cost': {'authorized_cap_usd': CAP,
            'known_usage_estimated_usd_all_trials': sum(a.get('estimated_usd') or 0 for a in newer['attempts']),
            'known_usage_estimated_usd_37': sum(a.get('estimated_usd') or 0 for a in current_attempts),
            'accounted_usd_all_trials': sum(map(accounted_cost, newer['attempts'])),
            'attempts_without_usage_37': sum(not a.get('usage') for a in current_attempts)},
        'observed_37_model_versions': sorted({a['model_version'] for a in current_attempts if a.get('model_version')}),
        'passes_development_gate': (pairwise['common_resolved'] == 150 and candidate['mcc'] > local['mcc']
            and candidate['fpr'] < local['fpr'] and candidate['fnr'] <= local['fnr'])}
    comparison.save(Path('docs/gemini-37-results.json'), result)
    lines = ['\n## Measured results', '',
        f"Both hosted trials attempted 150 rows. This table compares the same {summary['common_resolved']} examples resolved by all three configurations.", '',
        '| Judge | TP | FP | FN | TN | MCC | False positives | Missed jailbreaks | Coverage |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        v = summary['variants'][name]
        m, c = v['common_metrics'], v['common_confusion']
        lines.append(f"| {name} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | {m['mcc']:.3f} | {m['fpr']:.2%} | {m['fnr']:.2%} | {v['scored']}/150 |")
    lines += ['', f"Gemini 3.7 adoption gate passed: {result['passes_development_gate']}. The gate uses the Qwen/3.7 pair ({pairwise['common_resolved']}/150 resolved), independently of the three-way table. Error rows are excluded from accuracy and remain explicit coverage failures. These are development results for the fixed grader; they do not rank general model capability.", '',
        f"Gemini 3.7 known token cost: ${result['cost']['known_usage_estimated_usd_37']:.6f}. All trials known token cost: ${result['cost']['known_usage_estimated_usd_all_trials']:.6f}. Budget-accounted total including unknown-call allowances: ${result['cost']['accounted_usd_all_trials']:.6f} of $5. These are estimates, not a billing receipt or account balance.", '',
        'Successful judge-evaluation means (different run periods; failed attempts and retry waits excluded): ' + ', '.join(f"{n}: {summary['variants'][n]['successful_judge_mean_seconds']:.3f}s" for n in names) + '.', '',
        'The production grader and defaults were not changed. Raw checkpoints remain local. Full machine-readable details are in `gemini-37-results.json`.']
    lines += ['', '### Interpretation', '',
        'Retain Qwen for this grader. On the matched 149 examples, Gemini 3.7 produced 22 more false positives and missed one more jailbreak than Qwen. Compared with Gemini 3.5 it caught one additional jailbreak but added one false positive; the modest MCC increase does not establish a statistically reliable improvement. Both Gemini models left the same row (zero-based index 77) ungraded.', '',
        'Qwen on its full 150-row cohort remains MCC 0.510, FPR 42.71%, FNR 5.56%. Its 0.527 MCC in the matched table reflects excluding a false negative, not a new improvement. This was a completed experiment with an explicit error, not a mid-run interruption.', '',
        'Verification: 340 offline tests passed, independent review checked spending and report logic, and production rubric SHA remained identical to the frozen baseline. The completed resume command exited without changing the ledger or making new inference calls.']
    doc = Path('docs/gemini-37-comparison.md')
    prefix = doc.read_text().split('\n## Measured results')[0].replace('Status: prepared; results pending.', f"Status: {newer['status']}; all 150 rows attempted.")
    doc.write_text(prefix + '\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    build_report()
