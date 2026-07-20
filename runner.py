import json
import asyncio
import time
import logging
from target import query_target
from evaluators import run_deterministic_eval, run_llm_judge_eval
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AegisEval.Runner")

async def run_single_test(
    test_case: dict, 
    semaphore: asyncio.Semaphore,
    target_provider: str = None,
    target_model: str = None,
    judge_provider: str = None,
    judge_model: str = None
) -> dict:
    """
    Runs a single test case through the target and the evaluators with concurrency limits.
    """
    async with semaphore:
        tc_id = test_case["id"]
        category = test_case["category"]
        prompt = test_case["prompt"]
        expected_criteria = test_case["expected_criteria"]
        
        logger.info(f"Starting {tc_id} ({category})")
        
        start_time = time.time()
        
        # 1. Query the target model
        target_response = await query_target(prompt, provider=target_provider, model=target_model)
        
        latency = time.time() - start_time
        logger.info(f"Finished target query for {tc_id} in {latency:.2f}s")
        
        # 2. Run deterministic checks first (fast, cost-free)
        det_result = run_deterministic_eval(prompt, target_response)
        
        # 3. Run LLM Judge only if deterministic checks pass (saves API costs)
        if not det_result["passed"]:
            eval_result = det_result
            eval_type = "deterministic"
        else:
            logger.info(f"Deterministic checks passed for {tc_id}. Invoking LLM Judge...")
            llm_result = await run_llm_judge_eval(
                prompt=prompt,
                response=target_response,
                expected_criteria=expected_criteria,
                provider=judge_provider,
                model=judge_model
            )
            eval_result = llm_result
            eval_type = "llm_judge"
            
        logger.info(f"Completed {tc_id}. Grade: {eval_result['score']} | Passed: {eval_result['passed']}")
        
        return {
            "id": tc_id,
            "category": category,
            "prompt": prompt,
            "expected_criteria": expected_criteria,
            "description": test_case.get("description", ""),
            "tags": test_case.get("tags", []),
            "response": target_response,
            "score": eval_result["score"],
            "reasoning": eval_result["reasoning"],
            "passed": eval_result["passed"],
            "eval_type": eval_type,
            "latency_seconds": round(latency, 2)
        }

async def run_suite(
    tag_filter: str = None,
    target_provider: str = None,
    target_model: str = None,
    judge_provider: str = None,
    judge_model: str = None
) -> dict:
    """
    Loads all test cases, runs them, and returns consolidated results.
    """
    # Load test cases
    try:
        with open(config.BASE_DIR / "test_cases.json", "r") as f:
            test_cases = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load test_cases.json: {e}")
        return {"error": str(e)}

    # Apply optional filter by tag
    if tag_filter:
        test_cases = [tc for tc in test_cases if tag_filter in tc.get("tags", [])]
        logger.info(f"Filtered test suite to {len(test_cases)} cases with tag '{tag_filter}'")
    else:
        logger.info(f"Loaded {len(test_cases)} test cases from suite.")

    if not test_cases:
        logger.warning("No test cases match selection criteria.")
        return {"results": [], "summary": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}}

    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    
    # Create concurrent tasks
    tasks = [
        run_single_test(
            tc, 
            semaphore, 
            target_provider, 
            target_model, 
            judge_provider, 
            judge_model
        ) 
        for tc in test_cases
    ]
    
    start_suite_time = time.time()
    results = await asyncio.gather(*tasks)
    total_suite_time = time.time() - start_suite_time
    
    # Calculate summary metrics
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pass_rate = (passed / total * 100.0) if total > 0 else 0.0
    avg_latency = sum(r["latency_seconds"] for r in results) / total if total > 0 else 0.0
    
    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 1),
        "total_time_seconds": round(total_suite_time, 2),
        "average_latency_seconds": round(avg_latency, 2),
        "target": f"{target_provider or config.DEFAULT_TARGET_PROVIDER}:{target_model or config.DEFAULT_TARGET_MODEL}",
        "judge": f"{judge_provider or config.DEFAULT_JUDGE_PROVIDER}:{judge_model or config.DEFAULT_JUDGE_MODEL}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    logger.info(f"Suite completed. Pass Rate: {summary['pass_rate']}% | Time: {summary['total_time_seconds']}s")
    
    return {
        "results": results,
        "summary": summary
    }
