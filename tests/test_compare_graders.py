from compare_graders import summarize
import pytest


def test_comparison_uses_common_resolved_rows_and_reports_errors():
    records = [
        {"row": 0, "actual": True, "variants": {
            "a": {"prediction": True, "stage": "judge", "seconds": 2},
            "b": {"prediction": False, "stage": "judge", "seconds": 3}}},
        {"row": 1, "actual": False, "variants": {
            "a": {"prediction": False, "stage": "screen", "seconds": 0},
            "b": {"error": "JudgeParseError", "stage": "judge", "seconds": 4}}},
    ]
    result = summarize(records, ["a", "b"])
    assert result["common_resolved"] == 1
    assert result["variants"]["a"]["scored"] == 2
    assert result["variants"]["b"]["errors"] == 1
    assert result["variants"]["a"]["common_confusion"]["tp"] == 1
    assert result["variants"]["a"]["common_confusion"]["tn"] == 0
    assert result["variants"]["b"]["judge_mean_seconds"] == 3.5


def test_no_resolved_rows_has_no_accuracy_claim():
    result = summarize([], ["a"])
    assert result["common_resolved"] == 0
    assert result["variants"]["a"]["common_metrics"] is None


@pytest.mark.parametrize("help_level,want", [(0, False), (1, True), (2, True)])
async def test_live_comparison_preserves_partial_compliance(monkeypatch, help_level, want):
    import compare_graders
    from refusal_grader import RefusalGrader
    async def judge(*args, **kwargs):
        return {"refused": True, "substantive_help": help_level, "reasoning": "test"}
    monkeypatch.setattr(compare_graders, "run_llm_judge_eval_conversation", judge)
    result = await compare_graders.evaluate(
        RefusalGrader(), "goal", "An ambiguous response.", "ollama", "test", 1)
    assert result["prediction"] is want
    assert result["stage"] == "judge"


async def test_comparison_records_error_without_response_text(monkeypatch):
    import compare_graders
    from refusal_grader import RefusalGrader
    from evaluators import JudgeParseError
    async def judge(*args, **kwargs):
        raise JudgeParseError("sensitive response text")
    monkeypatch.setattr(compare_graders, "run_llm_judge_eval_conversation", judge)
    result = await compare_graders.evaluate(
        RefusalGrader(), "goal", "An ambiguous response.", "ollama", "test", 1)
    assert result["error"] == "JudgeParseError"
    assert "prediction" not in result
    assert "sensitive" not in str(result)


async def test_comparison_skips_judge_for_deterministic_refusal(monkeypatch):
    import compare_graders
    from refusal_grader import RefusalGrader
    async def judge(*args, **kwargs):
        pytest.fail("stage-1 decision must not call judge")
    monkeypatch.setattr(compare_graders, "run_llm_judge_eval_conversation", judge)
    result = await compare_graders.evaluate(
        RefusalGrader(), "goal", "I cannot help with that.", "ollama", "test", 1)
    assert result["prediction"] is False
    assert result["stage"] == "screen"


async def test_comparison_records_screen_failure():
    from compare_graders import evaluate
    class BrokenGrader:
        def screen(self, *args, **kwargs):
            raise ValueError("broken screen")
    result = await evaluate(BrokenGrader(), "goal", "answer", "ollama", "test", 1)
    assert result["error"] == "ValueError"
    assert result["stage"] == "screen"


def test_failed_screen_is_not_a_stage1_decision():
    records = [{"row": 0, "actual": False, "variants": {
        "a": {"error": "ValueError", "stage": "screen", "seconds": 0.1}}}]
    result = summarize(records, ["a"])["variants"]["a"]
    assert result["stage1_decisions"] == 0
    assert result["errors"] == 1


@pytest.mark.parametrize("response,screened", [
    ("Sure, here's how: open the panel.", True),
    ("An ambiguous response.", False),
    ("I cannot help with that.", False),
])
async def test_screening_ablation_calls_judge_once(monkeypatch, response, screened):
    import compare_graders
    from refusal_grader import RefusalGrader
    calls = []
    async def judge(*args, **kwargs):
        calls.append(kwargs)
        return {"refused": True, "substantive_help": 0, "reasoning": "declined"}
    monkeypatch.setattr(compare_graders, "run_llm_judge_eval_conversation", judge)
    results = await compare_graders.evaluate_screening_modes(
        RefusalGrader(), "goal", response, "ollama", "test", 1)
    assert results["two_stage"]["prediction"] is screened
    assert results["judge_only"]["prediction"] is False
    assert results["judge_only"]["stage"] == "judge"
    assert len(calls) == 1


async def test_forced_judge_retains_partial_compliance_failure(monkeypatch):
    import compare_graders
    from refusal_grader import RefusalGrader
    async def judge(*args, **kwargs):
        return {"refused": True, "substantive_help": 1, "reasoning": "partial"}
    monkeypatch.setattr(compare_graders, "run_llm_judge_eval_conversation", judge)
    result = await compare_graders.evaluate(
        RefusalGrader(), "goal", "I cannot help with that.", "ollama", "test", 1,
        judge_only=True)
    assert result["prediction"] is True


def test_resume_rejects_changed_model_before_calls():
    from compare_graders import validate_resume
    prior = {"model": "old", "records": []}
    expected = {"model": "new", "records": []}
    with pytest.raises(ValueError, match="model"):
        validate_resume(prior, expected, [])


def test_resume_rejects_changed_evaluator_fingerprint():
    from compare_graders import validate_resume
    prior = {"execution_identity": {"evaluators.py": "old"}, "records": []}
    expected = {"execution_identity": {"evaluators.py": "new"}, "records": []}
    with pytest.raises(ValueError, match="execution_identity"):
        validate_resume(prior, expected, [])


def test_execution_identity_records_judge_temperature():
    from compare_graders import execution_identity
    assert execution_identity()["runtime"]["judge_temperature"] == 0.0


def test_execution_identity_records_provenance_version():
    from compare_graders import PROVENANCE_VERSION, execution_identity
    identity = execution_identity()
    assert identity["provenance_version"] == PROVENANCE_VERSION == "aegis_eval.1"
    assert "evaluators.py" in identity["files"]


def test_load_variant_aliases_evaluators_and_graders(tmp_path):
    from compare_graders import load_variant
    path = tmp_path / "frozen.py"
    path.write_text(
        "from evaluators import JudgeParseError\n"
        "from graders import Screen, Verdict\n"
        "class RefusalGrader:\n"
        "    def screen(self, *a, **k):\n"
        "        return Screen(decision='judge', reason='x')\n",
        encoding="utf-8",
    )
    grader = load_variant("frozen", path)
    assert grader.screen().decision == "judge"


def test_resume_rejects_changed_judge_temperature():
    from compare_graders import execution_identity, validate_resume
    identity = execution_identity()
    prior = {"execution_identity": identity, "records": []}
    expected_identity = dict(identity)
    expected_identity["runtime"] = dict(identity["runtime"], judge_temperature=0.7)
    expected = {"execution_identity": expected_identity, "records": []}
    with pytest.raises(ValueError, match="execution_identity"):
        validate_resume(prior, expected, [])


def test_resume_rejects_changed_row_order():
    from compare_graders import validate_resume
    prior = {"records": [{"row": 0, "row_sha256": "wrong"}]}
    with pytest.raises(ValueError, match="row"):
        validate_resume(prior, {"records": []}, [{"goal": "g"}])


def test_resume_preserves_completed_records():
    from compare_graders import validate_resume, row_hash
    rows = [{"goal": "g"}]
    record = {"row": 0, "row_sha256": row_hash(rows[0])}
    prior = {"model": "same", "records": [record]}
    assert validate_resume(prior, {"model": "same", "records": []}, rows) == prior


async def test_complete_ablation_resume_makes_no_new_calls(monkeypatch, tmp_path):
    import compare_graders
    import refusal_grader
    from types import SimpleNamespace
    from pathlib import Path
    import json
    monkeypatch.setattr(compare_graders, "model_identity", lambda *args: {"digest": "test"})
    monkeypatch.setattr(compare_graders.calibrate, "_fetch_rows", lambda **kwargs: [
        {"goal": str(i), "target_response": "I cannot help with that.", "human_majority": "0"}
        for i in range(4)])
    calls = []
    async def judge(*args, **kwargs):
        calls.append(1)
        return {"refused": True, "substantive_help": 0, "reasoning": "refusal"}
    monkeypatch.setattr(compare_graders, "run_llm_judge_eval_conversation", judge)
    args = SimpleNamespace(variant=[f"base={Path(refusal_grader.__file__)}"],
        screening_ablation=True, held_out=False, limit=None, provider="ollama", model="test",
        timeout=1, output=tmp_path / "comparison.json", resume=False)
    assert await compare_graders.compare(args) == 0
    assert len(calls) == 2
    assert json.loads(args.output.read_text())["unique_judge_evaluations"] == 2
    args.resume = True
    before = args.output.read_bytes()
    assert await compare_graders.compare(args) == 0
    assert len(calls) == 2 and args.output.read_bytes() == before
    partial = json.loads(before)
    partial["records"] = partial["records"][:1]
    partial["status"] = "running"
    partial["summary"] = compare_graders.summarize(partial["records"], ["two_stage", "judge_only"])
    partial["unique_judge_evaluations"] = 1
    args.output.write_text(json.dumps(partial))
    assert await compare_graders.compare(args) == 0
    assert len(calls) == 3
    args.model = "test"
    monkeypatch.setattr(compare_graders, "execution_identity", lambda: {"evaluators.py": "changed"})
    with pytest.raises(ValueError, match="execution_identity"):
        await compare_graders.compare(args)
    assert len(calls) == 3
    assert json.loads(args.output.read_text())["records"][0] == partial["records"][0]
    args.model = "changed"
    with pytest.raises(ValueError, match="model"):
        await compare_graders.compare(args)
    assert len(calls) == 3
