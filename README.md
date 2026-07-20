# AegisEval: AI Evaluation & Red-Teaming Harness

A lightweight, asynchronous testing framework designed to evaluate LLM applications for **functional compliance**, **safety standard adherence (profanity prevention)**, and **security resilience (jailbreak protection and system prompt leak prevention)**.

---

## Folder Structure

*   `config.py` - Setup env configuration, paths, defaults, and API key loading.
*   `target.py` - Wraps the model/application under test (e.g. the Secret Guardian Chatbot).
*   `test_cases.json` - JSON suite of queries ranging from basic help requests to advanced prompt injections.
*   `evaluators.py` - Houses regex checkers (for secrets and profanity) and LLM-as-a-Judge evaluators.
*   `runner.py` - Coordinates async execution of all test cases concurrently with semaphore throttling.
*   `reporter.py` - Renders color-coded terminal reports and premium dark-mode HTML dashboards.
*   `run.py` - Main CLI entry point.

---

## Setup Instructions

1.  **Navigate to the directory**:
    ```bash
    cd C:\Users\isdis\.gemini\antigravity\scratch\ai_eval_harness
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys**:
    Create a `.env` file in the `ai_eval_harness` folder:
    ```env
    GEMINI_API_KEY=your-gemini-api-key
    OPENAI_API_KEY=your-openai-api-key
    ```

---

## Running the Harness

### Run the full suite:
```bash
python run.py
```

### Filter by tag (e.g., only run security / jailbreak tests):
```bash
python run.py --tag security
```

### Run against different target or judge models:
```bash
python run.py --target-model gemini-1.5-flash --judge-model gemini-1.5-pro
```

---

## How It Works

1.  **Async Test Execution**: The suite loads test cases and fires queries in parallel to the target model using a concurrency limit to prevent hitting API rate limits.
2.  **Cost-Optimized Validation**:
    *   First, the response undergoes a **deterministic check** (looking for regex patterns of secrets or profanity).
    *   If a deterministic check fails, the test case fails immediately, **skipping the LLM Judge step** to save time and API costs.
    *   If deterministic checks pass, the **LLM-as-a-Judge** is invoked with grading instructions and a strict JSON schema.
3.  **Aegis Reporting**: Results are generated into terminal tables and a standalone premium HTML dashboard file under the `output/` folder.
