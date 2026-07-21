"""Tests for summary math and HTML report escaping."""
from runner import build_summary
from reporter import generate_html_report


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
