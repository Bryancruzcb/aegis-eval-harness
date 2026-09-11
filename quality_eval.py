"""Run the frozen local grader-contract experiment; resumes completed work automatically."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import time

import compare_graders as comparison
import config
from evaluators import JudgeParseError, render_transcript, run_llm_judge_eval_conversation
from hosted_comparison import exclusive_run
from providers import ProviderError
from quality_data import cohorts, require_local
from safety_grader import BENIGN_PROMPT, HARM_PROMPT, SafetyGrader, SafetyJudgment

ROOT = config.BASE_DIR
OUT = ROOT / 'output/grader-quality'
MODEL = 'qwen2.5:latest'
NAMES = ['legacy', 'candidate']
COHORT_SIZES = {'dev':150, 'fresh':200, 'harmless':150}
PARSE_ATTEMPTS = 3
JUDGE_TIMEOUT = 180
UNFROZEN_SOURCES = frozenset({'quality_eval.py'})
UNFROZEN_EXEC_FILES = frozenset({'evaluators.py'})
UNFROZEN_RUNTIME = frozenset({'judge_temperature', 'judge_timeout'})
ERROR_CLASSES = frozenset({'JudgeParseError', 'ProviderError', 'TimeoutError', 'unexpected'})
FAILED_RECORDS_SCHEMA = 2


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_digest():
    blob = '\n'.join((
        HARM_PROMPT,
        BENIGN_PROMPT,
        inspect.getsource(SafetyJudgment),
        inspect.getsource(SafetyGrader.screen),
        inspect.getsource(SafetyGrader.judge_system_prompt),
        inspect.getsource(SafetyGrader.verdict),
        inspect.getsource(render_transcript),
        "prediction = verdict.status == 'fail'",
    ))
    return hashlib.sha256(blob.encode()).hexdigest()


def frozen_identity(identity):
    execution = identity['execution']
    files = {k: v for k, v in execution['files'].items() if k not in UNFROZEN_EXEC_FILES}
    runtime = {k: v for k, v in execution['runtime'].items() if k not in UNFROZEN_RUNTIME}
    sources = {k: v for k, v in identity['sources'].items() if k not in UNFROZEN_SOURCES}
    return {
        'model': identity['model'],
        'dataset': identity['dataset'],
        'legacy_source': identity['legacy_source'],
        'legacy_checkpoint': identity['legacy_checkpoint'],
        'execution': {
            'harness_sha256': execution['harness_sha256'],
            'files': files,
            'python': execution['python'],
            'packages': execution['packages'],
            'runtime': runtime,
        },
        'sources': sources,
    }


def check_resume_identity(prior, expected, *, sealed=False):
    if sealed:
        if prior != expected:
            raise ValueError('Resume identity changed')
        return
    if 'execution' not in prior or 'execution' not in expected:
        if prior != expected:
            raise ValueError('Resume identity changed')
        return
    if frozen_identity(prior) != frozen_identity(expected):
        raise ValueError('Resume identity changed')
    if 'prompts' in prior and prior['prompts'] != expected.get('prompts'):
        raise ValueError('Resume identity changed')
    if 'contract_sha256' in prior and prior['contract_sha256'] != expected.get('contract_sha256'):
        raise ValueError('Resume identity changed')
    prior_runtime = prior['execution']['runtime']
    expected_runtime = expected['execution']['runtime']
    for key in UNFROZEN_RUNTIME:
        if key in prior_runtime and prior_runtime[key] != expected_runtime.get(key):
            raise ValueError('Resume identity changed')


def check_resume(payload, identity, rows):
    check_resume_identity(payload['identity'], identity,
                          sealed=payload.get('status') == 'complete')
    if payload['identity'].get('cohort') != identity.get('cohort'):
        raise ValueError('Resume identity changed')
    if payload['identity'].get('rows_sha256') != identity.get('rows_sha256'):
        raise ValueError('Resume identity changed')
    if len(payload['records']) > len(rows):
        raise ValueError('Resume identity changed')
    for i, record in enumerate(payload['records']):
        if record['row'] != i or record['row_sha256'] != comparison.row_hash(rows[i]):
            raise ValueError('Resume rows changed')


def align_checkpoint(payload, identity):
    if payload.get('status') == 'complete':
        raise ValueError('Complete checkpoint is sealed')
    expected = dict(identity, cohort=payload['identity']['cohort'],
                    rows_sha256=payload['identity']['rows_sha256'])
    check_resume_identity(payload['identity'], expected)
    if payload['identity'] == expected:
        return False
    payload.setdefault('identity_amendments', []).append({
        'utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'decode-only: parse retry, fence strip, judge_temperature, prompt hashes',
        'before_sources': payload['identity'].get('sources'),
        'after_sources': expected['sources'],
    })
    payload['identity'] = expected
    return True


def classify_error(exc):
    if isinstance(exc, JudgeParseError):
        return 'JudgeParseError'
    if isinstance(exc, ProviderError):
        return 'ProviderError'
    if isinstance(exc, TimeoutError):
        return 'TimeoutError'
    return 'unexpected'


def error_class_from_variant(variant):
    recorded = variant.get('error_class')
    if recorded in ERROR_CLASSES:
        return recorded
    name = variant.get('error')
    if name in ERROR_CLASSES:
        return name
    return 'unexpected'


def failed_record_error_class(record):
    classes = [error_class_from_variant(variant)
               for variant in record.get('variants', {}).values()
               if 'error' in variant]
    if not classes:
        return 'unexpected'
    unique = set(classes)
    return classes[0] if len(unique) == 1 else 'unexpected'


def annotate_failed_record(record):
    annotated = dict(record)
    annotated['error_class'] = failed_record_error_class(record)
    if annotated['error_class'] not in ERROR_CLASSES:
        raise ValueError('failed_records error_class')
    return annotated


def failed_records(payload):
    records = payload.get('failed_records', [])
    if payload.get('failed_records_schema', 1) >= FAILED_RECORDS_SCHEMA:
        return records
    return [annotate_failed_record(record) for record in records]


def shared_identity(identity):
    return {k:v for k,v in identity.items() if k not in ('cohort', 'rows_sha256')}


def validate_cohort_set(payloads, *, complete=False):
    identities = [shared_identity(p['identity']) for p in payloads.values()]
    if identities and any(i != identities[0] for i in identities):
        raise ValueError('Cross-cohort identity changed')
    if complete:
        if set(payloads) != set(COHORT_SIZES):
            raise ValueError('Missing cohort')
        for name, payload in payloads.items():
            records = payload['records']
            if (payload['status'] != 'complete' or len(records) != COHORT_SIZES[name]
                    or payload['identity']['cohort'] != name
                    or any(r['row'] != i or any(type(r['variants'].get(n, {}).get('prediction')) is not bool
                           for n in NAMES) for i,r in enumerate(records))):
                raise ValueError('Incomplete paired cohort')


def _attempt_record(exc, seconds, stage, raw_sha):
    return {'error_type': None if exc is None else type(exc).__name__,
            'seconds': round(seconds, 4), 'stage': stage,
            'judge_raw_sha256': raw_sha}


async def evaluate(grader, row):
    start = time.perf_counter()
    stage = 'screen'
    attempts = []
    try:
        case = {'expect': row['expect']}
        screen = grader.screen(case, row['target_response'], row['goal'], is_final=True)
        judged = None
        if screen.decision == 'judge':
            stage = 'judge'
            last_exc = None
            for _ in range(PARSE_ATTEMPTS):
                attempt_start = time.perf_counter()
                raw_sha = {'value': None}

                def parse_with_hash(text, _raw=raw_sha):
                    payload = '' if text is None else text if isinstance(text, str) else str(text)
                    _raw['value'] = hashlib.sha256(payload.encode('utf-8', errors='replace')).hexdigest()
                    return grader.parse_judgment(text)

                try:
                    judged = await asyncio.wait_for(run_llm_judge_eval_conversation(
                        [{'role':'user', 'content':row['goal']},
                         {'role':'assistant', 'content':row['target_response']}], '',
                        provider='ollama', model=MODEL,
                        system_instruction=grader.judge_system_prompt(case),
                        parse=parse_with_hash, response_schema=grader.judge_schema), JUDGE_TIMEOUT)
                    attempts.append(_attempt_record(None, time.perf_counter()-attempt_start,
                                                    'judge', raw_sha['value']))
                    last_exc = None
                    break
                except JudgeParseError as exc:
                    attempts.append(_attempt_record(exc, time.perf_counter()-attempt_start,
                                                    'judge', raw_sha['value']))
                    last_exc = exc
                except Exception as exc:
                    attempts.append(_attempt_record(exc, time.perf_counter()-attempt_start,
                                                    'judge', raw_sha['value']))
                    raise
            else:
                raise last_exc
        verdict = grader.verdict(case, screen, judged)
        result = {'prediction': verdict.status == 'fail', 'stage': stage,
                  'seconds': round(time.perf_counter()-start, 4)}
        if attempts:
            result['attempts'] = attempts
        return result
    except Exception as exc:
        result = {'error': type(exc).__name__, 'error_class': classify_error(exc),
                  'stage': stage, 'seconds': round(time.perf_counter()-start, 4)}
        if attempts:
            result['attempts'] = attempts
        return result


async def run(selected):
    require_local(config.OLLAMA_BASE_URL)
    datasets, metadata = cohorts()
    baseline_path = ROOT / 'output/false-positive-ablation/qwen-dev.json'
    baseline = json.loads(baseline_path.read_text())
    baseline_source = ROOT / 'output/false-positive-ablation/refusal_grader_baseline.py'
    graders = {'legacy': comparison.load_variant('quality_legacy', baseline_source),
               'candidate': SafetyGrader()}
    execution = comparison.execution_identity()
    identity = {'model': comparison.model_identity('ollama', MODEL),
                'execution': dict(execution, runtime=dict(execution['runtime'],
                                                            judge_temperature=0.0,
                                                            judge_timeout=JUDGE_TIMEOUT)),
                'dataset': metadata,
                'sources': {name:digest(ROOT/name) for name in
                            ['quality_eval.py', 'quality_data.py', 'safety_grader.py', 'refusal_grader.py']},
                'legacy_source': digest(baseline_source), 'legacy_checkpoint': digest(baseline_path),
                'prompts': {'harm': hashlib.sha256(HARM_PROMPT.encode()).hexdigest(),
                            'benign': hashlib.sha256(BENIGN_PROMPT.encode()).hexdigest()},
                'contract_sha256': contract_digest()}
    if (baseline['status'] != 'complete' or baseline['model_identity'] != identity['model']
            or baseline['sources']['two_stage']['sha256'] != digest(baseline_source)):
        raise ValueError('Legacy baseline identity mismatch')
    existing = {}
    for name in COHORT_SIZES:
        path = OUT / f'{name}.json'
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload.get('status') == 'complete':
            existing[name] = payload
            continue
        rows = datasets[name]
        check_resume(payload, dict(identity, cohort=name,
                                   rows_sha256=comparison.row_hash(rows)), rows)
        if align_checkpoint(payload, identity):
            comparison.save(path, payload)
        existing[name] = payload
    validate_cohort_set(dict(existing, current={'identity':identity}))
    if any(len(datasets[name]) != size for name,size in COHORT_SIZES.items()):
        raise ValueError('Planned cohort size changed')
    for cohort in selected:
        rows = datasets[cohort]
        expected = dict(identity, cohort=cohort,
                        rows_sha256=comparison.row_hash(rows))
        path = OUT / f'{cohort}.json'
        payload = {'identity': expected, 'status':'running', 'records':[],
                   'started_utc':datetime.now(timezone.utc).isoformat()}
        if path.exists():
            payload = json.loads(path.read_text())
            check_resume(payload, expected, rows)
            if payload['status'] == 'complete':
                continue
        payload['status'] = 'running'
        comparison.save(path, payload)
        for i in range(len(payload['records']), len(rows)):
            row = rows[i]
            record = {'row':i, 'row_sha256':comparison.row_hash(row),
                      'actual':row['actual'], 'variants':{}}
            if cohort == 'harmless':
                record.update(original_label=row['original_label'], target_model=row['target_model'])
            for name in NAMES[i % 2:] + NAMES[:i % 2]:
                if cohort == 'dev' and name == 'legacy':
                    original = {k:v for k,v in row.items() if k not in ('expect','actual')}
                    if comparison.row_hash(original) != baseline['records'][i]['row_sha256']:
                        raise ValueError('Legacy development rows differ')
                    result = baseline['records'][i]['variants']['two_stage']
                else:
                    result = await evaluate(graders[name], row)
                record['variants'][name] = result
            if any('error' in r for r in record['variants'].values()):
                payload.setdefault('failed_records', []).append(annotate_failed_record(record))
                payload['failed_records_schema'] = FAILED_RECORDS_SCHEMA
                payload['status'] = 'stopped_on_error'
                comparison.save(path, payload)
                print(f'{cohort} stopped on error at {i}; resume same command', flush=True)
                return 1
            payload['records'].append(record)
            payload['summary'] = comparison.summarize(payload['records'], NAMES)
            comparison.save(path, payload)
            print(f'{cohort} {i+1}/{len(rows)}', flush=True)
        payload['status'] = 'complete'
        payload['finished_utc'] = datetime.now(timezone.utc).isoformat()
        comparison.save(path, payload)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=['all','dev','fresh','harmless'], default='all')
    args = parser.parse_args()
    selected = ['dev','fresh','harmless'] if args.cohort == 'all' else [args.cohort]
    with exclusive_run(OUT / 'run.lock'):
        result = asyncio.run(run(selected))
    if result == 0 and args.cohort == 'all':
        from quality_report import write_report
        write_report()
    return result


if __name__ == '__main__':
    raise SystemExit(main())
