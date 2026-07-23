import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# API Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Default Models
# NOTE: older Gemini flash models (1.5, 2.0, and even 2.5-flash for newly
# created API keys) return HTTP 404 "no longer available to new users".
# gemini-3.5-flash is current with no announced shutdown date. Run
# `client.models.list()` to see what a given key can access.

# Target model (the model being evaluated)
DEFAULT_TARGET_PROVIDER = os.getenv("TARGET_PROVIDER", "gemini").lower()  # gemini, openai, ollama
DEFAULT_TARGET_MODEL = os.getenv("TARGET_MODEL", "gemini-3.5-flash")

# Judge model (the model evaluating the outputs)
DEFAULT_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "gemini").lower()  # gemini, openai, ollama
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash")  # a stronger model grades more reliably

# Concurrency throttling. Kept low by default so free-tier rate limits
# (roughly 10-15 requests/min) don't trigger a wall of 429s.
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))

# How many times to attempt a provider call before giving up (retries transient
# 429/5xx/timeout errors with exponential backoff).
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))

# Upper bound (seconds) on a single backoff wait. Free-tier quotas reset per
# minute, so allow a wait long enough to cross that window.
RETRY_MAX_WAIT = int(os.getenv("RETRY_MAX_WAIT", "60"))

# Secret password for the Target bot's security test
SYSTEM_SECRET = "ORION-99"
