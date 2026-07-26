"""Renders results as a color-coded terminal summary and a standalone HTML report.

The HTML template is rendered with autoescaping ON: the target model's output is
untrusted (a jailbroken response could contain markup), so every interpolated
value is HTML-escaped to keep the report from executing model-supplied scripts.
"""
import logging

from jinja2 import Template

import config

logger = logging.getLogger("AegisEval.Reporter")

# Modern, premium dark-mode HTML dashboard template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AegisEval Report - {{ summary.timestamp }}</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #141b2d;
            --border-color: #232d45;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.15);
            --fail: #ef4444;
            --fail-glow: rgba(239, 68, 68, 0.15);
            --error: #f59e0b;
            --error-glow: rgba(245, 158, 11, 0.15);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.2);
            --font-main: 'Plus Jakarta Sans', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: var(--font-main);
            line-height: 1.5;
            padding: 2rem;
        }

        .container { max-width: 1200px; margin: 0 auto; }

        header {
            margin-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
        }

        .title-area h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }

        .title-area p { color: var(--text-secondary); font-size: 0.95rem; }

        .meta-tag {
            background-color: var(--border-color);
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-family: var(--font-mono);
            display: inline-block;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .metric-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .metric-card:hover { transform: translateY(-2px); border-color: var(--accent); }

        .metric-card .label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .metric-card .value { font-size: 2.2rem; font-weight: 700; font-family: var(--font-mono); }
        .metric-card.success-card .value { color: var(--success); }
        .metric-card.fail-card .value { color: var(--fail); }
        .metric-card.error-card .value { color: var(--error); }
        .metric-card.accent-card .value { color: var(--accent); }

        .metric-card::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 4px;
        }

        .metric-card.success-card::after { background-color: var(--success); }
        .metric-card.fail-card::after { background-color: var(--fail); }
        .metric-card.error-card::after { background-color: var(--error); }
        .metric-card.accent-card::after { background-color: var(--accent); }

        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }

        .tabs {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            background-color: var(--card-bg);
            padding: 0.3rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-family: var(--font-main);
            font-weight: 600;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }

        .tab-btn:hover { color: var(--text-primary); }
        .tab-btn.active { background-color: var(--accent); color: white; }

        .test-list { display: flex; flex-direction: column; gap: 1rem; }

        .test-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            overflow: hidden;
            transition: border-color 0.2s ease;
        }

        .test-card.passed-card { border-left: 5px solid var(--success); }
        .test-card.failed-card { border-left: 5px solid var(--fail); }
        .test-card.error-card { border-left: 5px solid var(--error); }

        .test-header {
            padding: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
        }

        .test-header:hover { background-color: rgba(255, 255, 255, 0.02); }
        .test-info { display: flex; align-items: center; gap: 1rem; }
        .test-id { font-family: var(--font-mono); font-weight: 700; color: var(--text-secondary); }

        .badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            text-transform: uppercase;
        }

        .badge-functional { background-color: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
        .badge-safety { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-security { background-color: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }

        .test-description { font-size: 0.9rem; color: var(--text-secondary); }
        .test-status-area { display: flex; align-items: center; gap: 1.5rem; }
        .test-score { font-family: var(--font-mono); font-weight: 600; font-size: 1rem; }
        .score-pass { color: var(--success); }
        .score-fail { color: var(--fail); }
        .score-error { color: var(--error); }

        .status-pill {
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        .status-pill.pass { background-color: var(--success-glow); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }
        .status-pill.fail { background-color: var(--fail-glow); color: var(--fail); border: 1px solid rgba(239, 68, 68, 0.3); }
        .status-pill.error { background-color: var(--error-glow); color: var(--error); border: 1px solid rgba(245, 158, 11, 0.3); }

        .test-details {
            display: none;
            padding: 1.5rem;
            border-top: 1px solid var(--border-color);
            background-color: rgba(0, 0, 0, 0.15);
        }

        .test-details.show { display: block; }

        .detail-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }

        @media (max-width: 768px) { .detail-grid { grid-template-columns: 1fr; } }

        .detail-box {
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        }

        .detail-title {
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .detail-content {
            font-family: var(--font-mono);
            font-size: 0.85rem;
            white-space: pre-wrap;
            word-break: break-word;
            background-color: rgba(0,0,0,0.2);
            padding: 0.75rem;
            border-radius: 6px;
            max-height: 250px;
            overflow-y: auto;
        }

        .reasoning-box { grid-column: span 2; }
        @media (max-width: 768px) { .reasoning-box { grid-column: span 1; } }

        .reasoning-box .detail-content {
            font-family: var(--font-main);
            font-size: 0.9rem;
            background-color: var(--accent-glow);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #e0e7ff;
        }

        .test-meta {
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-family: var(--font-mono);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title-area">
                <h1>AegisEval</h1>
                <p>AI Safety, Jailbreak, and Quality Evaluation Suite</p>
            </div>
            <div>
                <span class="meta-tag">Timestamp: {{ summary.timestamp }}</span>
            </div>
        </header>

        <div style="margin-bottom: 1.5rem; display: flex; gap: 1rem; flex-wrap: wrap;">
            <span class="meta-tag" style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3);">Target Model: {{ summary.target }}</span>
            <span class="meta-tag" style="background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.3);">Judge Model: {{ summary.judge }}</span>
            <span class="meta-tag">Duration: {{ summary.total_time_seconds }}s</span>
            <span class="meta-tag">Avg Latency: {{ summary.average_latency_seconds }}s</span>
        </div>

        <section class="metrics-grid">
            <div class="metric-card accent-card">
                <div class="label">Attack Pass Rate</div>
                <div class="value">{% if summary.attack_pass_rate is defined and summary.attack_pass_rate is not none %}{{ summary.attack_pass_rate }}%{% else %}—{% endif %}</div>
            </div>
            <div class="metric-card error-card">
                <div class="label">Grader FP Rate</div>
                <div class="value">{% if summary.grader_fp_rate is defined and summary.grader_fp_rate is not none %}{{ (summary.grader_fp_rate * 100) | round(1) }}%{% else %}—{% endif %}</div>
            </div>
            <div class="metric-card accent-card">
                <div class="label">Overall Break Rate</div>
                <div class="value">{% if summary.overall_break_rate is defined and summary.overall_break_rate is not none %}{{ (summary.overall_break_rate * 100) | round(1) }}%{% else %}—{% endif %}</div>
            </div>
            <div class="metric-card accent-card">
                <div class="label">Pass Rate</div>
                <div class="value">{{ summary.pass_rate }}%</div>
            </div>
            <div class="metric-card accent-card">
                <div class="label">Total Cases</div>
                <div class="value">{{ summary.total }}</div>
            </div>
            <div class="metric-card success-card">
                <div class="label">Passed</div>
                <div class="value">{{ summary.passed }}</div>
            </div>
            <div class="metric-card fail-card">
                <div class="label">Failed</div>
                <div class="value">{{ summary.failed }}</div>
            </div>
            <div class="metric-card error-card">
                <div class="label">Errors</div>
                <div class="value">{{ summary.errors }}</div>
            </div>
        </section>

        <section class="controls">
            <div class="tabs">
                <button class="tab-btn active" onclick="filterResults('all', this)">All ({{ summary.total }})</button>
                <button class="tab-btn" onclick="filterResults('pass', this)">Passed ({{ summary.passed }})</button>
                <button class="tab-btn" onclick="filterResults('fail', this)">Failed ({{ summary.failed }})</button>
                <button class="tab-btn" onclick="filterResults('error', this)">Errors ({{ summary.errors }})</button>
                <button class="tab-btn" onclick="filterResults('functional', this)">Functional</button>
                <button class="tab-btn" onclick="filterResults('safety', this)">Safety</button>
                <button class="tab-btn" onclick="filterResults('security', this)">Security</button>
            </div>
        </section>

        <section class="test-list">
            {% for tc in results %}
            <div class="test-card {% if tc.status == 'pass' %}passed-card{% elif tc.status == 'fail' %}failed-card{% else %}error-card{% endif %}"
                 data-status="{{ tc.status }}"
                 data-category="{{ tc.category }}">

                <div class="test-header" onclick="toggleDetails('details-{{ tc.id }}')">
                    <div class="test-info">
                        <span class="test-id">{{ tc.id }}</span>
                        <span class="badge badge-{{ tc.category }}">{{ tc.category }}</span>
                        <span class="test-description">{{ tc.description }}</span>
                    </div>

                    <div class="test-status-area">
                        {% if tc.repeat_count is defined and tc.repeat_count > 1 %}
                        <span class="meta-tag">{% if tc.expect == 'benign' %}grader-flagged{% elif tc.expect == 'comply' %}over-refused{% else %}broke{% endif %} {{ tc.break_count }}/{{ tc.repeat_count }}</span>
                        {% endif %}
                        {% if tc.false_positive is defined and tc.false_positive %}<span class="meta-tag">grader FP</span>{% endif %}
                        <span class="test-score {% if tc.status == 'pass' %}score-pass{% elif tc.status == 'fail' %}score-fail{% else %}score-error{% endif %}">
                            Score: {% if tc.score is not none %}{{ tc.score }}{% else %}&mdash;{% endif %}
                        </span>
                        {% if tc.status == 'pass' %}
                        <span class="status-pill pass">&#10004; Pass</span>
                        {% elif tc.status == 'fail' %}
                        <span class="status-pill fail">&#10008; Fail</span>
                        {% else %}
                        <span class="status-pill error">&#9888; Error</span>
                        {% endif %}
                    </div>
                </div>

                <div id="details-{{ tc.id }}" class="test-details">
                    <div class="detail-grid">
                        {% if tc.transcript is defined and tc.transcript | length > 2 %}
                        <div class="detail-box reasoning-box">
                            <div class="detail-title">Conversation</div>
                            <div class="detail-content">{% for m in tc.transcript %}{{ '[USER] ' if m.role == 'user' else '[BOT] ' }}{{ m.content }}
{% endfor %}</div>
                        </div>
                        {% else %}
                        <div class="detail-box">
                            <div class="detail-title">User Prompt / Attack</div>
                            <div class="detail-content">{% if tc.transcript is defined and tc.transcript %}{{ tc.transcript[0].content }}{% else %}{{ tc.prompt }}{% endif %}</div>
                        </div>
                        <div class="detail-box">
                            <div class="detail-title">Target Response</div>
                            <div class="detail-content">{% if tc.response is defined and tc.response is not none %}{{ tc.response }}{% else %}&mdash;{% endif %}</div>
                        </div>
                        {% endif %}
                        <div class="detail-box reasoning-box">
                            <div class="detail-title">{% if tc.eval_type == 'llm_judge' %}Judge Evaluation Reasoning{% else %}Evaluation Reasoning{% endif %}</div>
                            <div class="detail-content">{{ tc.reasoning }}</div>
                        </div>
                    </div>
                    <div class="test-meta">
                        <span>Expected Criteria: {{ tc.expected_criteria }}</span>
                        <span>|</span>
                        <span>Tags: {{ tc.tags | join(', ') }}</span>
                        <span>|</span>
                        <span>Latency: {{ tc.latency_seconds }}s</span>
                        <span>|</span>
                        <span>Evaluator: {{ tc.eval_type }}</span>
                    </div>
                </div>
            </div>
            {% endfor %}
        </section>
    </div>

    <script>
        function toggleDetails(id) {
            document.getElementById(id).classList.toggle('show');
        }

        function filterResults(type, btn) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const statuses = ['pass', 'fail', 'error'];
            const categories = ['functional', 'safety', 'security'];

            document.querySelectorAll('.test-card').forEach(card => {
                const status = card.getAttribute('data-status');
                const category = card.getAttribute('data-category');
                let show;
                if (type === 'all') {
                    show = true;
                } else if (statuses.includes(type)) {
                    show = status === type;
                } else if (categories.includes(type)) {
                    show = category === type;
                } else {
                    show = false;
                }
                card.style.display = show ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>
"""


def generate_html_report(results_payload: dict, file_name: str = "report.html") -> str:
    """Render results into a premium dark-themed HTML report (autoescaped)."""
    template = Template(HTML_TEMPLATE, autoescape=True)
    html_content = template.render(
        results=results_payload["results"],
        summary=results_payload["summary"],
    )

    file_path = config.OUTPUT_DIR / file_name
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML report written to {file_path}")
    return str(file_path)


def print_terminal_summary(results_payload: dict):
    """Print a clean summary table to the console."""
    summary = results_payload["summary"]
    results = results_payload["results"]

    print("\n" + "=" * 80)
    print(" AegisEval - RUN SUMMARY")
    print("=" * 80)
    print(f"Timestamp:       {summary['timestamp']}")
    print(f"Target Model:    {summary['target']}")
    print(f"Judge Model:     {summary['judge']}")
    print(f"Total Cases:     {summary['total']}")
    print(f"Passed:          {summary['passed']}")
    print(f"Failed:          {summary['failed']}")
    print(f"Errors:          {summary['errors']}")
    print(f"Pass Rate:       {summary['pass_rate']}%  (of {summary.get('evaluated', 0)} evaluated; errors excluded)")
    if summary.get("attack_pass_rate") is not None:
        print(f"Attack Pass Rate: {summary['attack_pass_rate']}%")
    if summary.get("total_breaks"):
        print(f"Breaks:          {summary['total_breaks']}")
    break_rate_shown = summary.get("overall_break_rate") is not None
    if break_rate_shown:
        print(f"Overall Break Rate: {round(summary['overall_break_rate'] * 100, 1)}%"
              f"  (of all evaluated attack runs)")
    if summary.get("benign_false_positives"):
        print(f"Benign FP:       {summary['benign_false_positives']}")
    if summary.get("grader_fp_rate") is not None:
        print(f"Grader FP Rate:  {round(summary['grader_fp_rate'] * 100, 1)}%"
              f"  (of evaluated benign cases)")
    print(f"Average Latency: {summary['average_latency_seconds']}s")
    print(f"Total Time:      {summary['total_time_seconds']}s")
    if break_rate_shown:  # nothing to disambiguate when the rate wasn't printed
        print("Note: pass_rate is per-CASE take-worst over repeats; overall_break_rate is "
              "per-RUN. They are NOT complementary.")
    print("=" * 80)

    print(f"{'ID':<8} | {'Category':<10} | {'Status':<7} | {'Score':<5} | {'Description'}")
    print("-" * 80)

    colors = {"pass": "\033[92m", "fail": "\033[91m", "error": "\033[93m"}
    labels = {"pass": "PASS", "fail": "FAIL", "error": "ERROR"}
    for r in results:
        status = r.get("status", "error")
        label = f"{colors.get(status, '')}{labels.get(status, status.upper()):<7}\033[0m"
        score = r.get("score")
        score_str = f"{score:<5}" if score is not None else "  -  "
        desc = r.get("description", "")
        if len(desc) > 40:
            desc = desc[:37] + "..."
        print(f"{r['id']:<8} | {r['category']:<10} | {label} | {score_str} | {desc}")
    print("=" * 80 + "\n")
