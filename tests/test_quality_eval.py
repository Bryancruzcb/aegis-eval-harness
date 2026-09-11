import pytest

from providers import ProviderError
from quality_eval import check_resume, check_resume_identity, classify_error


def test_different_candidate_across_cohorts_is_rejected():
    from quality_eval import validate_cohort_set
    with pytest.raises(ValueError, match='Cross-cohort'):
        validate_cohort_set({'dev': {'identity':{'model':'a','cohort':'dev'}},
                             'fresh': {'identity':{'model':'b','cohort':'fresh'}}})


def test_partial_cohorts_cannot_be_reported_as_complete():
    from quality_eval import validate_cohort_set
    with pytest.raises(ValueError, match='Missing cohort'):
        validate_cohort_set({'dev': {'identity':{'model':'a','cohort':'dev'}}}, complete=True)


def test_resume_rejects_changed_identity_and_rows():
    rows = [{'goal':'x'}]
    from compare_graders import row_hash
    prior = {'identity': {'model':'a'}, 'records':[{'row':0, 'row_sha256':row_hash(rows[0])}]}
    check_resume(prior, {'model':'a'}, rows)
    with pytest.raises(ValueError):
        check_resume(prior, {'model':'b'}, rows)
    with pytest.raises(ValueError):
        check_resume(prior, {'model':'a'}, [{'goal':'changed'}])


async def test_judge_error_does_not_become_a_prediction(monkeypatch):
    import quality_eval
    from safety_grader import SafetyGrader
    async def broken(*args, **kwargs):
        raise RuntimeError('unavailable')
    monkeypatch.setattr(quality_eval, 'run_llm_judge_eval_conversation', broken)
    result = await quality_eval.evaluate(SafetyGrader(), {'expect':'comply', 'goal':'x', 'target_response':'y'})
    assert result['error'] == 'RuntimeError'
    assert 'prediction' not in result


async def test_harmless_evaluation_uses_benign_prompt_and_refusal_contract(monkeypatch):
    import quality_eval
    from safety_grader import SafetyGrader, BENIGN_PROMPT
    async def judge(*args, **kwargs):
        assert kwargs['provider'] == 'ollama'
        assert kwargs['system_instruction'] == BENIGN_PROMPT
        return dict(harmful_assistance=False, refused=False, partial_refusal=True, reasoning='partial')
    monkeypatch.setattr(quality_eval, 'run_llm_judge_eval_conversation', judge)
    result = await quality_eval.evaluate(SafetyGrader(), {'expect':'comply', 'goal':'x', 'target_response':'y'})
    assert result['prediction'] is True


def _full_identity():
    return {
        'model': {'model': 'qwen', 'digest': 'abc'},
        'dataset': {'seed': 1},
        'legacy_source': 'a',
        'legacy_checkpoint': 'b',
        'execution': {
            'harness_sha256': 'h',
            'files': {'evaluators.py': 'e1', 'providers.py': 'p'},
            'python': '3.11',
            'packages': {'pydantic': '2'},
            'runtime': {'max_retries': 6, 'retry_max_wait': 60},
        },
        'sources': {
            'quality_eval.py': 'q1',
            'quality_data.py': 'd1',
            'safety_grader.py': 's1',
            'refusal_grader.py': 'r1',
        },
        'contract_sha256': 'contract',
        'cohort': 'fresh',
        'rows_sha256': 'rows',
    }


def test_quality_eval_writes_under_provenance_version():
    from compare_graders import PROVENANCE_VERSION
    from quality_eval import OUT
    assert OUT.name == PROVENANCE_VERSION
    assert OUT.parent.name == "grader-quality"


def test_decode_only_source_hash_may_change_on_resume():
    prior = _full_identity()
    expected = _full_identity()
    expected['sources']['quality_eval.py'] = 'q2'
    expected['execution']['files']['evaluators.py'] = 'e2'
    expected['execution']['runtime']['judge_temperature'] = 0.0
    expected['execution']['runtime']['judge_timeout'] = 180
    check_resume_identity(prior, expected)


def test_safety_grader_hash_change_fails_incomplete_resume():
    prior = _full_identity()
    expected = _full_identity()
    expected['sources']['safety_grader.py'] = 's2'
    with pytest.raises(ValueError, match='Resume identity changed'):
        check_resume_identity(prior, expected)


def test_complete_checkpoint_rejects_decode_file_hash_change():
    prior = _full_identity()
    expected = _full_identity()
    expected['sources']['quality_eval.py'] = 'q2'
    payload = {'identity': prior, 'status': 'complete', 'records': []}
    with pytest.raises(ValueError, match='Resume identity changed'):
        check_resume(payload, expected, [{'goal': 'x'}])


def test_align_does_not_rewrite_complete_checkpoint():
    from quality_eval import align_checkpoint
    prior = _full_identity()
    payload = {'status': 'complete', 'identity': prior}
    expected = _full_identity()
    expected['sources']['quality_eval.py'] = 'q2'
    with pytest.raises(ValueError, match='sealed'):
        align_checkpoint(payload, expected)
    assert payload['identity'] == prior
    assert 'identity_amendments' not in payload


def test_contract_digest_change_fails_resume():
    prior = _full_identity()
    expected = _full_identity()
    expected['contract_sha256'] = 'other'
    with pytest.raises(ValueError, match='Resume identity changed'):
        check_resume_identity(prior, expected)


def test_model_or_dataset_change_still_fails_resume():
    prior = _full_identity()
    expected = _full_identity()
    expected['model'] = {'model': 'qwen', 'digest': 'other'}
    with pytest.raises(ValueError, match='Resume identity changed'):
        check_resume_identity(prior, expected)
    expected = _full_identity()
    expected['sources']['quality_data.py'] = 'changed'
    with pytest.raises(ValueError, match='Resume identity changed'):
        check_resume_identity(prior, expected)


def test_prompt_hash_change_fails_resume_after_recorded():
    prior = _full_identity()
    prior['prompts'] = {'harm': 'h1', 'benign': 'b1'}
    expected = _full_identity()
    expected['prompts'] = {'harm': 'h2', 'benign': 'b1'}
    with pytest.raises(ValueError, match='Resume identity changed'):
        check_resume_identity(prior, expected)


async def test_parse_error_retries_then_succeeds(monkeypatch):
    import quality_eval
    from safety_grader import SafetyGrader
    calls = {'n': 0}
    valid = '{"reasoning":"ok","harmful_assistance":false,"refused":false,"partial_refusal":false}'

    async def flaky(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            return kwargs['parse']('not json')
        return kwargs['parse'](valid)

    monkeypatch.setattr(quality_eval, 'run_llm_judge_eval_conversation', flaky)
    result = await quality_eval.evaluate(
        SafetyGrader(), {'expect': 'comply', 'goal': 'x', 'target_response': 'y'})
    assert result['prediction'] is False
    assert 'error' not in result
    assert calls['n'] == 2
    assert result['attempts'][0]['error_type'] == 'JudgeParseError'
    assert result['attempts'][1]['error_type'] is None
    assert len(result['attempts'][0]['judge_raw_sha256']) == 64


async def test_three_parse_failures_become_an_error(monkeypatch):
    import quality_eval
    from safety_grader import SafetyGrader
    calls = {'n': 0}

    async def always_bad(*args, **kwargs):
        calls['n'] += 1
        return kwargs['parse']('not json')

    monkeypatch.setattr(quality_eval, 'run_llm_judge_eval_conversation', always_bad)
    result = await quality_eval.evaluate(
        SafetyGrader(), {'expect': 'comply', 'goal': 'x', 'target_response': 'y'})
    assert result['error'] == 'JudgeParseError'
    assert result['error_class'] == 'JudgeParseError'
    assert 'prediction' not in result
    assert calls['n'] == 3
    assert len(result['attempts']) == 3


async def test_provider_error_does_not_retry(monkeypatch):
    import quality_eval
    from safety_grader import SafetyGrader
    calls = {'n': 0}

    async def boom(*args, **kwargs):
        calls['n'] += 1
        raise ProviderError('down')

    monkeypatch.setattr(quality_eval, 'run_llm_judge_eval_conversation', boom)
    result = await quality_eval.evaluate(
        SafetyGrader(), {'expect': 'comply', 'goal': 'x', 'target_response': 'y'})
    assert result['error'] == 'ProviderError'
    assert result['error_class'] == 'ProviderError'
    assert calls['n'] == 1
    assert 'prediction' not in result


def test_error_taxonomy_covers_the_named_classes():
    from evaluators import JudgeParseError
    from quality_eval import ERROR_CLASSES
    assert classify_error(JudgeParseError('x')) == 'JudgeParseError'
    assert classify_error(ProviderError('x')) == 'ProviderError'
    assert classify_error(TimeoutError()) == 'TimeoutError'
    assert classify_error(RuntimeError('x')) == 'unexpected'
    assert ERROR_CLASSES == {'JudgeParseError', 'ProviderError', 'TimeoutError', 'unexpected'}


def test_failed_record_has_top_level_error_class():
    from quality_eval import annotate_failed_record
    record = {'row': 48, 'row_sha256': 'h', 'actual': True, 'variants': {
        'legacy': {'prediction': False, 'stage': 'judge', 'seconds': 1},
        'candidate': {'error': 'JudgeParseError', 'error_class': 'JudgeParseError',
                      'stage': 'judge', 'seconds': 2}}}
    annotated = annotate_failed_record(record)
    assert annotated['error_class'] == 'JudgeParseError'
    assert 'error_class' not in record


def test_schema1_failed_records_gain_error_class_without_rewriting_source():
    from quality_eval import failed_records
    payload = {'failed_records': [
        {'row': 48, 'variants': {
            'legacy': {'prediction': True, 'stage': 'judge', 'seconds': 1},
            'candidate': {'error': 'JudgeParseError', 'stage': 'judge', 'seconds': 2}}},
        {'row': 80, 'variants': {
            'legacy': {'prediction': False, 'stage': 'judge', 'seconds': 1},
            'candidate': {'error': 'TimeoutError', 'error_class': 'TimeoutError',
                          'stage': 'judge', 'seconds': 2}}},
    ]}
    view = failed_records(payload)
    assert [r['error_class'] for r in view] == ['JudgeParseError', 'TimeoutError']
    assert all('error_class' not in r for r in payload['failed_records'])
    assert payload.get('failed_records_schema') is None


def test_failed_records_schema2_keeps_recorded_class():
    from quality_eval import FAILED_RECORDS_SCHEMA, failed_records
    payload = {
        'failed_records_schema': FAILED_RECORDS_SCHEMA,
        'failed_records': [
            {'row': 1, 'error_class': 'ProviderError', 'variants': {
                'candidate': {'error': 'ProviderError', 'error_class': 'ProviderError'}}},
        ],
    }
    view = failed_records(payload)
    assert view[0]['error_class'] == 'ProviderError'
    assert view[0] is payload['failed_records'][0]
