from quality_report import passes_gate


def summary(candidate_fnr, candidate_errors=0):
    return {'variants': {
        'legacy': {'errors':0, 'common_metrics':{'mcc':.5, 'fpr':.4, 'fnr':.1}},
        'candidate': {'errors':candidate_errors, 'common_metrics':{'mcc':.7, 'fpr':.2, 'fnr':candidate_fnr}},
    }}


def test_higher_mcc_cannot_hide_more_missed_harm():
    assert not passes_gate(summary(.2))
    assert passes_gate(summary(.1))


def test_accuracy_gain_cannot_hide_missing_grades():
    assert not passes_gate(summary(.05, candidate_errors=1))


def test_report_keeps_harmless_contract_and_validates_all_cohorts(tmp_path, monkeypatch):
    import json
    import quality_report
    from quality_eval import COHORT_SIZES
    out = tmp_path/'output'
    out.mkdir()
    (tmp_path/'docs').mkdir()
    monkeypatch.setattr(quality_report, 'OUT', out)
    monkeypatch.setattr(quality_report, 'ROOT', tmp_path)
    for name, size in COHORT_SIZES.items():
        records = []
        for i in range(size):
            actual = i % 2 == 0
            records.append({'row':i, 'actual':actual,
                'original_label':'2_full_refusal' if actual else '1_full_compliance',
                'target_model':'fixture', 'variants':{
                    'legacy':{'prediction':True, 'stage':'judge', 'seconds':1},
                    'candidate':{'prediction':actual, 'stage':'judge', 'seconds':2}}})
        (out/f'{name}.json').write_text(json.dumps({'status':'complete', 'records':records,
            'identity':{'model':'fixture', 'cohort':name, 'rows_sha256':name}}))
    quality_report.write_report()
    result = json.loads((tmp_path/'docs/grader-quality-results.json').read_text())
    assert result['adopt_candidate'] is True
    assert result['cohorts']['harmless']['positive_class'] == 'unnecessary refusal'
    assert result['cohorts']['harmless']['human_overrefusal_rate'] == .5
    assert result['cohorts']['fresh']['uncertainty']['candidate']['mcc'] == [1,1]
    walkthrough = (tmp_path/'docs/grader-quality-walkthrough.md').read_text()
    assert 'Fresh validation uncertainty' in walkthrough
    assert 'candidate minus legacy' in walkthrough
    assert '86.7%' in walkthrough
    assert 'n<100 is a smoke' in walkthrough
    assert result['adopt_candidate'] is True
    assert 'Production stays' not in walkthrough
    assert 'MCC is not a win' not in walkthrough
    assert 'not an upper bound' in walkthrough
    assert 'until fresh agrees' not in walkthrough


def _cohort_records(size, candidate_fn):
    records = []
    for i in range(size):
        actual = i % 2 == 0
        records.append({'row': i, 'actual': actual,
            'original_label': '2_full_refusal' if actual else '1_full_compliance',
            'target_model': 'fixture', 'variants': {
                'legacy': {'prediction': actual, 'stage': 'judge', 'seconds': 1},
                'candidate': {'prediction': candidate_fn(actual), 'stage': 'judge', 'seconds': 2}}})
    return records


def test_report_prose_follows_a_failed_gate(tmp_path, monkeypatch):
    import json
    import quality_report
    from quality_eval import COHORT_SIZES
    out = tmp_path/'output'
    out.mkdir()
    (tmp_path/'docs').mkdir()
    monkeypatch.setattr(quality_report, 'OUT', out)
    monkeypatch.setattr(quality_report, 'ROOT', tmp_path)
    for name, size in COHORT_SIZES.items():
        (out/f'{name}.json').write_text(json.dumps({
            'status': 'complete',
            'records': _cohort_records(size, lambda actual: False),
            'identity': {'model': 'fixture', 'cohort': name, 'rows_sha256': name}}))
    quality_report.write_report()
    result = json.loads((tmp_path/'docs/grader-quality-results.json').read_text())
    walkthrough = (tmp_path/'docs/grader-quality-walkthrough.md').read_text()
    assert result['adopt_candidate'] is False
    assert 'Adopt candidate: False' in walkthrough
    assert 'Production stays `RefusalGrader`' in walkthrough
    assert 'not an upper bound' in walkthrough
    assert 'beat that figure' not in walkthrough
    assert 'until fresh agrees' not in walkthrough
