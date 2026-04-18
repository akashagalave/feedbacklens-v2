import random
import requests
from locust import HttpUser, task, between, events

# ══════════════════════════════════════════════════════
# SLA THRESHOLDS — production standards
# ══════════════════════════════════════════════════════
LATENCY_SLA_SECONDS = 30.0
ERROR_RATE_THRESHOLD = 0.10


# ══════════════════════════════════════════════════════
# RESPONSE VALIDATOR — single helper, no duplicate code
# ══════════════════════════════════════════════════════
def validate_analyze_response(response) -> tuple[bool, str]:
    """
    Returns (success, reason)
    Classifies failures by type — not just generic error
    """
    # SLA check — latency
    elapsed = response.elapsed.total_seconds()
    if elapsed > LATENCY_SLA_SECONDS:
        return False, f"Latency SLA breach: {elapsed:.1f}s > {LATENCY_SLA_SECONDS}s"

    # HTTP check
    if response.status_code != 200:
        return False, f"HTTP error: {response.status_code}"

    # Parse check
    try:
        data = response.json()
    except Exception:
        return False, "JSON parse failure"

    # Content check
    top_issues = data.get("top_issues", [])
    if not top_issues:
        return False, "Empty top_issues"
    if top_issues == ["No data found for this company"]:
        return False, "No data found in Qdrant"
    if top_issues == ["Error parsing insights"]:
        return False, "LLM parsing error"
    if not data.get("recommendations"):
        return False, "Empty recommendations"

    return True, "ok"


class FeedbackLensUser(HttpUser):
    wait_time = between(2, 6)

    # ══════════════════════════════════════════════════
    # REAL QUERIES — varied, noisy, ambiguous
    # Matches eval dataset for consistency
    # ══════════════════════════════════════════════════
    swiggy_queries = [
        {"query": "What are swiggy delivery issues?",        "company": "swiggy"},
        {"query": "Swiggy late delivery complaints",          "company": "swiggy"},
        {"query": "Why are swiggy customers unhappy?",        "company": "swiggy"},
        {"query": "Swiggy refund problems",                   "company": "swiggy"},
        {"query": "Swiggy delivery charges too high",         "company": "swiggy"},
        {"query": "Swiggy executive behavior issues",         "company": "swiggy"},
        {"query": "What are swiggy customer support problems?","company": "swiggy"},
        {"query": "swiggi delivry prblms",                    "company": "swiggy"},
        {"query": "food arrived cold and late",               "company": "swiggy"},
        {"query": "swiggy bad",                               "company": "swiggy"},
    ]

    uber_queries = [
        {"query": "What are uber pricing problems?",    "company": "uber"},
        {"query": "Uber driver cancellation issues",    "company": "uber"},
        {"query": "Uber surge pricing complaints",      "company": "uber"},
        {"query": "Uber safety concerns",               "company": "uber"},
        {"query": "Uber long wait time problems",       "company": "uber"},
        {"query": "driver canceled my ride again",      "company": "uber"},
        {"query": "uber pricng issus very bad",         "company": "uber"},
        {"query": "uber problems",                      "company": "uber"},
    ]

    zomato_queries = [
        {"query": "What issues do zomato customers face?", "company": "zomato"},
        {"query": "Zomato delivery delay complaints",       "company": "zomato"},
        {"query": "Zomato wrong order problems",            "company": "zomato"},
        {"query": "Zomato refund issues",                   "company": "zomato"},
        {"query": "zomto ordr wrng item recieved",          "company": "zomato"},
        {"query": "zomato feedback",                        "company": "zomato"},
    ]

    # ══════════════════════════════════════════════════
    # TASKS — weighted by real traffic distribution
    # Swiggy 3x, Uber 2x, Zomato 1x, Health 1x
    # ══════════════════════════════════════════════════
    @task(3)
    def analyze_swiggy(self):
        self._run_analyze(
            payload=random.choice(self.swiggy_queries),
            name="Swiggy Analysis"
        )

    @task(2)
    def analyze_uber(self):
        self._run_analyze(
            payload=random.choice(self.uber_queries),
            name="Uber Analysis"
        )

    @task(1)
    def analyze_zomato(self):
        self._run_analyze(
            payload=random.choice(self.zomato_queries),
            name="Zomato Analysis"
        )

    @task(1)
    def health_check(self):
        with self.client.get(
            "/health",
            name="Health Check",
            catch_response=True,
            timeout=10
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    # ══════════════════════════════════════════════════
    # HELPER — single validate function
    # Classifies: timeout, http error, parse, content
    # ══════════════════════════════════════════════════
    def _run_analyze(self, payload: dict, name: str):
        with self.client.post(
            "/analyze",
            name=name,
            json=payload,
            catch_response=True,
            timeout=60
        ) as response:
            try:
                success, reason = validate_analyze_response(response)
                if success:
                    response.success()
                else:
                    response.failure(reason)
            except requests.exceptions.Timeout:
                response.failure("Timeout — no response within 60s")
            except requests.exceptions.ConnectionError:
                response.failure("Connection error — service unreachable")
            except Exception as e:
                response.failure(f"Unexpected error: {type(e).__name__}: {e}")