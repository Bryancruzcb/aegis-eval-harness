import pytest

from aegis_eval.workflows.grader_quality.quality_data import select_harmless, require_local


def test_harmless_selection_excludes_seen_prompts_and_preserves_labels():
    files = {name: [dict(prompt=p, completion='answer '+name+p, type='safe', final_label=label)
                    for p, label in [('seen','1_full_compliance'), ('new','3_partial_refusal')]]
             for name in ['a','b']}
    rows = select_harmless(files, [{'goal':' SEEN ', 'target_response':'old'}])
    assert len(rows) == 1
    assert rows[0]['goal'] == 'new'
    assert rows[0]['actual'] is True
    assert rows[0]['original_label'] == '3_partial_refusal'


def test_harmless_invalid_label_rejected():
    with pytest.raises(ValueError, match='label'):
        select_harmless({'a':[dict(prompt='x', completion='y', type='safe', final_label='unknown')]}, [])


@pytest.mark.parametrize('url', ['https://example.com/v1', 'http://localhost.evil.com/v1'])
def test_remote_endpoints_rejected(url):
    with pytest.raises(ValueError, match='local'):
        require_local(url)


def test_loopback_accepted():
    require_local('http://localhost:11434/v1')
