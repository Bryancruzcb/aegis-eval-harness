"""Tests for summary math and HTML report escaping."""
from runner import build_summary
from reporter import generate_html_report, print_terminal_summary


# --- Summary construction ---

def test_summary_excludes_errors_from_pass_rate():
    results = [
        {"status": "pass", "latency_seconds": 1.0},
        {"status": "fail", "latency_seconds": 1.0},
        {"status": "error", "latency_seconds": 0.0},
    ]
    summary = build_summary(results, "target-x", "judge-y", total_time_seconds=5.0)
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["errors"] == 1
    # Pass rate is over evaluated (pass+fail) cases only, not errors.
    assert summary["pass_rate"] == 50.0


def test_empty_summary_has_all_keys():
    # An over-narrow filter yields zero cases; the reporter must not KeyError.
    summary = build_summary([], "target-x", "judge-y", total_time_seconds=0.0)
    for key in [
        "timestamp", "target", "judge", "total", "passed", "failed",
        "errors", "pass_rate", "total_time_seconds", "average_latency_seconds",
    ]:
        assert key in summary
    assert summary["total"] == 0
    assert summary["pass_rate"] == 0.0


# --- HTML escaping (XSS from model output) ---

def test_html_report_escapes_model_output():
    payload = {
        "results": [{
            "id": "TC-XSS", "category": "security", "prompt": "p",
            "response": "<script>alert('xss')</script>",
            "expected_criteria": "c", "description": "d", "tags": ["x"],
            "score": 0.0, "reasoning": "r", "passed": False,
            "status": "fail", "eval_type": "deterministic", "latency_seconds": 0.1,
        }],
        "summary": {
            "timestamp": "t", "target": "x", "judge": "y", "total": 1,
            "passed": 0, "failed": 1, "errors": 0, "pass_rate": 0.0,
            "total_time_seconds": 0.1, "average_latency_seconds": 0.1,
        },
    }
    path = generate_html_report(payload, file_name="test_xss_report.html")
    with open(path, encoding="utf-8") as f:
        html = f.read()
    assert "<script>alert('xss')</script>" not in html
    assert "&lt;script&gt;" in html


# --- Transcripts, metrics, and guarded new fields ---

def _payload(**over):
    res = {"id": "T", "category": "security", "prompt": "p", "response": "<b>hi</b>",
           "expected_criteria": "c", "description": "d", "tags": ["x"], "score": 1.0,
           "reasoning": "r", "passed": True, "status": "pass", "eval_type": "llm_judge",
           "latency_seconds": 0.1,
           # 4 messages = a genuinely multi-turn (2-exchange) transcript; a
           # 2-message transcript is one exchange and falls back to the boxes.
           "transcript": [{"role": "user", "content": "ask"},
                          {"role": "assistant", "content": "<script>x</script>"},
                          {"role": "user", "content": "push harder"},
                          {"role": "assistant", "content": "still no"}],
           "repeat_count": 3, "break_count": 1, "expect": "refuse"}
    res.update(over)
    summary = {"timestamp": "t", "target": "x", "judge": "y", "total": 1, "passed": 1,
               "failed": 0, "errors": 0, "pass_rate": 100.0, "total_time_seconds": 0.1,
               "average_latency_seconds": 0.1, "attack_total": 1, "attack_pass_rate": 100.0,
               "grader_fp_rate": None, "overall_break_rate": 0.33, "total_breaks": 1,
               "benign_total": 0, "benign_false_positives": 0}
    return {"results": [res], "summary": summary}


def test_transcript_rendered_and_escaped():
    path = generate_html_report(_payload(), file_name="test_multiturn.html")
    html = open(path, encoding="utf-8").read()
    assert "&lt;script&gt;" in html and "<script>x</script>" not in html
    assert "broke 1/3" in html


def test_none_rate_renders_dash():
    path = generate_html_report(_payload(), file_name="test_none_rate.html")
    html = open(path, encoding="utf-8").read()
    assert "—" in html  # grader_fp_rate is None


def _html(name, **over):
    return open(generate_html_report(_payload(**over), file_name=name), encoding="utf-8").read()


# --- spec 4.7: two-box fallback is for a ONE-EXCHANGE (2-message) transcript --

def test_one_exchange_transcript_uses_prompt_and_response_boxes():
    html = _html("test_one_exchange.html",
                 transcript=[{"role": "user", "content": "ask"},
                             {"role": "assistant", "content": "reply"}])
    assert "User Prompt / Attack" in html and "Target Response" in html
    assert "Conversation" not in html


def test_multi_turn_transcript_uses_conversation_box():
    html = _html("test_multi_exchange.html")  # 4-message default fixture
    assert "Conversation" in html
    assert "User Prompt / Attack" not in html


def test_two_box_prompt_shows_only_the_first_turn():
    # A MULTI-TURN case that breaks on turn 1 has a 2-message transcript and so
    # takes the two-box fallback. ``prompt`` there is every turn joined with
    # newlines, so the box must be sourced from the transcript's first user
    # message instead - otherwise later turns show next to the turn-1 reply.
    html = _html("test_first_turn_only.html",
                 prompt="turn one\nturn two\nturn three",
                 transcript=[{"role": "user", "content": "turn one"},
                             {"role": "assistant", "content": "reply"}])
    assert "User Prompt / Attack" in html
    assert "turn one" in html
    assert "turn two" not in html and "turn three" not in html


# --- grader false-positive surfacing (spec 4.3) ------------------------------

def test_false_positive_badge_renders():
    html = _html("test_fp_badge.html", expect="benign", status="fail", score=0.0,
                 eval_type="deterministic", false_positive=True, repeat_count=1,
                 break_count=1)
    assert "grader FP" in html


def test_false_positive_badge_absent_when_not_flagged():
    assert "grader FP" not in _html("test_no_fp_badge.html")
    assert "grader FP" not in _html("test_fp_false_badge.html", false_positive=False)


def test_benign_break_badge_says_grader_flagged():
    html = _html("test_benign_badge.html", expect="benign", repeat_count=3, break_count=2)
    assert "grader-flagged 2/3" in html and "broke 2/3" not in html


# --- target errors must not render the literal string "None" -----------------

def test_none_response_renders_em_dash_not_none():
    html = _html("test_none_response.html", response=None, status="error", score=None,
                 eval_type="target_error",
                 transcript=[{"role": "user", "content": "ask"}])
    assert '<div class="detail-content">None</div>' not in html
    assert '<div class="detail-content">&mdash;</div>' in html


# --- reasoning label is only "Judge ..." when a judge actually ran -----------

def test_reasoning_label_says_judge_for_llm_judge():
    assert "Judge Evaluation Reasoning" in _html("test_judge_label.html")


def test_reasoning_label_drops_judge_when_no_judge_ran():
    html = _html("test_det_label.html", eval_type="deterministic")
    assert "Judge Evaluation Reasoning" not in html
    assert "Evaluation Reasoning" in html


# --- terminal summary: break/FP rates + the non-complementary caption --------

def test_terminal_summary_prints_break_and_fp_rates(capsys):
    payload = _payload()
    payload["summary"]["grader_fp_rate"] = 0.25
    print_terminal_summary(payload)
    out = capsys.readouterr().out
    assert "Overall Break Rate" in out and "33.0%" in out
    assert "Grader FP Rate" in out and "25.0%" in out
    assert "NOT complementary" in out


def test_terminal_summary_omits_caption_when_break_rate_suppressed(capsys):
    # A benign-only run has no attack runs, so the break-rate line is suppressed
    # and the caption explaining it has nothing to explain.
    payload = _payload()
    payload["summary"]["overall_break_rate"] = None
    print_terminal_summary(payload)
    out = capsys.readouterr().out
    assert "Overall Break Rate" not in out
    assert "NOT complementary" not in out


def test_terminal_summary_survives_absent_rate_keys(capsys):
    payload = {"results": [], "summary": {
        "timestamp": "t", "target": "x", "judge": "y", "total": 0, "passed": 0,
        "failed": 0, "errors": 0, "pass_rate": 0.0, "total_time_seconds": 0.0,
        "average_latency_seconds": 0.0}}
    print_terminal_summary(payload)  # legacy summary: must not KeyError
    out = capsys.readouterr().out
    assert "Overall Break Rate" not in out and "Grader FP Rate" not in out


def test_existing_escaping_payload_without_new_keys_still_renders():
    # legacy-shaped payload (no transcript/repeat_count/attack_* keys) must not raise
    payload = {"results": [{"id": "L", "category": "security", "prompt": "p",
               "response": "<script>a</script>", "expected_criteria": "c", "description": "d",
               "tags": [], "score": 0.0, "reasoning": "r", "passed": False, "status": "fail",
               "eval_type": "deterministic", "latency_seconds": 0.1}],
               "summary": {"timestamp": "t", "target": "x", "judge": "y", "total": 1,
               "passed": 0, "failed": 1, "errors": 0, "pass_rate": 0.0,
               "total_time_seconds": 0.1, "average_latency_seconds": 0.1}}
    path = generate_html_report(payload, file_name="test_legacy.html")
    html = open(path, encoding="utf-8").read()
    assert "&lt;script&gt;" in html
