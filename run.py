import asyncio
import argparse
import json
import os
import sys
import logging
from pathlib import Path
import config
from runner import run_suite
from reporter import print_terminal_summary, generate_html_report

# Ensure logger output is clean for CLI execution
logging.basicConfig(
    level=logging.WARNING, # Keep it clean unless errors happen
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AegisEval.CLI")

def parse_args():
    parser = argparse.ArgumentParser(description="AegisEval: Autonomous AI Security & Quality Evaluation Suite")
    
    parser.add_argument(
        "--tag", 
        type=str, 
        default=None, 
        help="Filter test cases by tag (e.g. jailbreak, profanity, support)"
    )
    
    parser.add_argument(
        "--target-provider",
        type=str,
        default=None,
        choices=["gemini", "openai", "ollama"],
        help="API provider for the Target model under test"
    )
    
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Model name for the Target (e.g. gemini-1.5-flash)"
    )
    
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        choices=["gemini", "openai", "ollama"],
        help="API provider for the Judge evaluator model"
    )
    
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for the Judge evaluator (e.g. gemini-1.5-flash, gemini-1.5-pro)"
    )
    
    return parser.parse_args()

async def main_async():
    args = parse_args()
    
    # Check for keys in environment
    if not config.GEMINI_API_KEY and not config.OPENAI_API_KEY:
        print("\n\033[93m[Warning] Neither GEMINI_API_KEY nor OPENAI_API_KEY are set in environment variables.\033[0m")
        print("Please create a `.env` file in this directory or export them before running:")
        print("  GEMINI_API_KEY=your_gemini_api_key")
        print("  OPENAI_API_KEY=your_openai_api_key\n")
        
        # If we have no API keys, let's suggest creating a mock environment file or aborting
        create_mock = input("Would you like me to create a template '.env' file? (y/n): ").strip().lower()
        if create_mock == 'y':
            env_path = config.BASE_DIR / ".env"
            with open(env_path, "w") as f:
                f.write(
                    "# AegisEval Environment Settings\n"
                    "GEMINI_API_KEY=\n"
                    "OPENAI_API_KEY=\n"
                    "TARGET_PROVIDER=gemini\n"
                    "TARGET_MODEL=gemini-1.5-flash\n"
                    "JUDGE_PROVIDER=gemini\n"
                    "JUDGE_MODEL=gemini-1.5-flash\n"
                )
            print(f"Created template at {env_path}. Please open it and paste your API key(s).\n")
            sys.exit(0)
        else:
            print("Aborting evaluation run.")
            sys.exit(1)

    print("Initializing AegisEval Suite...")
    
    # Run evaluation suite
    results_payload = await run_suite(
        tag_filter=args.tag,
        target_provider=args.target_provider,
        target_model=args.target_model,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model
    )
    
    if "error" in results_payload:
        print(f"\033[91mRun failed with error:\033[0m {results_payload['error']}")
        sys.exit(1)
        
    # Write raw JSON output
    raw_output_path = config.OUTPUT_DIR / "run_results.json"
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
        
    # Print console report
    print_terminal_summary(results_payload)
    
    # Generate HTML report
    html_path = generate_html_report(results_payload)
    
    print(f"✔ Raw results saved to: {raw_output_path}")
    print(f"✔ Premium HTML dashboard generated at: {html_path}\n")

def main():
    # Set event loop policy for Windows if needed (avoids crash on loop termination)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
