"""Pinned local cohorts for the grader-contract experiment."""
import csv
import hashlib
import io
import json
import random
from urllib.parse import urlparse

import calibrate
import config
import jailjudge_validation as jj

SEED = 20260908
LABELS = {'1_full_compliance': False, '2_full_refusal': True, '3_partial_refusal': True}


def require_local(url):
    if urlparse(url).hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('Restricted datasets require local Ollama')


def select_harmless(files, previous):
    old_prompts = {jj.normalize(r['goal']) for r in previous}
    old_responses = {jj.normalize(r['target_response']) for r in previous}
    by_prompt = {}
    for model, rows in sorted(files.items()):
        for row in rows:
            if row['type'].startswith('contrast_'):
                continue
            if row['final_label'] not in LABELS:
                raise ValueError('Unknown human label')
            prompt = jj.normalize(row['prompt'])
            if prompt in old_prompts or jj.normalize(row['completion']) in old_responses:
                continue
            if not prompt or not row['completion'].strip():
                raise ValueError('Empty XSTest text')
            by_prompt.setdefault(prompt, {})[model] = row
    prompts = sorted(by_prompt)
    random.Random(SEED).shuffle(prompts)
    models = sorted(files)
    selected = []
    for i, prompt in enumerate(prompts):
        choices = by_prompt[prompt]
        preferred = models[i % len(models)]
        model = preferred if preferred in choices else sorted(choices)[0]
        row = choices[model]
        selected.append({'goal': row['prompt'], 'target_response': row['completion'],
                         'actual': LABELS[row['final_label']], 'expect': 'comply',
                         'original_label': row['final_label'], 'target_model': model})
    return selected


def cohorts():
    require_local(config.OLLAMA_BASE_URL)
    previous = calibrate._fetch_rows(refresh=False)
    dev, _ = calibrate._split(previous, seed=calibrate.SPLIT_SEED)
    path = config.BASE_DIR / '.cache/jailjudge/JAILJUDGE_ID.json'
    old, _ = jj.load(path, previous)
    from compare_graders import row_hash
    prior_run = json.loads((config.BASE_DIR / 'output/false-positive-ablation/qwen-jailjudge.json').read_text())
    if len(old) != len(prior_run['records']) or any(
            row_hash(row) != record['row_sha256'] for row, record in zip(old, prior_run['records'])):
        raise ValueError('Prior validation cohort does not match the excluded rows')
    fresh, counts = jj.select_rows(json.loads(path.read_bytes()), previous + old)
    inventory = json.loads((config.BASE_DIR / 'docs/xstest-data-inventory.json').read_text())
    files = {}
    for name, metadata in inventory['files'].items():
        raw = (config.BASE_DIR / '.cache/xstest' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata['sha256']:
            raise ValueError('XSTest pinned hash changed')
        files[name] = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    harmless = select_harmless(files, previous)
    convert = lambda rows: [dict(r, actual=calibrate._actual_jailbroken(r), expect='refuse') for r in rows]
    return {'dev': convert(dev), 'fresh': convert(fresh), 'harmless': harmless}, {
        'seed': SEED, 'jailjudge_selection_seed': jj.SEED,
        'jailjudge_sha256': jj.SHA256, 'jailjudge_counts': counts,
        'excluded_prior_jailjudge_rows': len(old), 'excluded_jbb_rows': len(previous),
        'xstest_commit': inventory['commit'],
        'xstest_sha256': {n: m['sha256'] for n, m in inventory['files'].items()},
        'harmless_selection': 'All unseen safe prompts, one response per prompt; rotating target assignment after seeded shuffle, without using labels',
        'freshness_limit': 'Normalized exact overlap excluded; public contamination and semantic overlap not ruled out'}
