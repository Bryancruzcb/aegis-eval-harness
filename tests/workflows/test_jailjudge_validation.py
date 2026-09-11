import pytest

from aegis_eval.workflows.grader_quality.jailjudge_validation import select_rows


def example(prompt, label, response=None):
    return {"user_prompt": prompt, "model_response": response or f"answer {prompt}",
            "language": "en", "is_jailbroken": label}


def test_selection_excludes_overlap_and_duplicate_requests():
    rows = [example(" Seen  prompt ", False), example("duplicate", False),
            example(" DUPLICATE ", True), example("a", True), example("b", False)]
    chosen, counts = select_rows(rows, [{"goal": "seen prompt", "target_response": "old"}], per_class=1)
    assert len(chosen) == 2
    assert {row["human_majority"] for row in chosen} == {"0", "1"}
    assert counts["overlap_excluded"] == 1
    assert counts["duplicate_requests_excluded"] == 1
    assert select_rows(rows, [{"goal": "seen prompt", "target_response": "old"}], per_class=1)[0] == chosen


def test_response_overlap_is_excluded():
    with pytest.raises(ValueError, match="Insufficient"):
        select_rows([example("new", True, "same answer"), example("b", False)],
                    [{"goal": "old", "target_response": " SAME  answer "}], per_class=1)


@pytest.mark.parametrize("label", ["false", "True", 0, 1, None])
def test_labels_must_be_real_booleans(label):
    with pytest.raises(ValueError, match="label"):
        select_rows([example("a", label)], [], per_class=1)


def test_empty_responses_are_counted_as_exclusions():
    empty = example("empty", False)
    empty["model_response"] = " "
    selected, counts = select_rows([empty, example("a", True), example("b", False)], [], per_class=1)
    assert len(selected) == 2
    assert counts["empty_text_excluded"] == 1


async def test_hosted_provider_is_rejected_before_loading_data():
    from types import SimpleNamespace
    from aegis_eval.workflows.grader_quality.compare_graders import compare
    with pytest.raises(ValueError, match="local Ollama"):
        await compare(SimpleNamespace(jailjudge_id="unused", provider="gemini"))


async def test_remote_ollama_is_rejected(monkeypatch):
    import aegis_eval.core.config as config
    from types import SimpleNamespace
    from aegis_eval.workflows.grader_quality.compare_graders import compare
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "https://remote.example/v1")
    with pytest.raises(ValueError, match="local Ollama"):
        await compare(SimpleNamespace(jailjudge_id="unused", provider="ollama"))


def test_loader_rejects_unpinned_file(tmp_path):
    from aegis_eval.workflows.grader_quality.jailjudge_validation import load
    path = tmp_path / "data.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="pinned"):
        load(path, [])
