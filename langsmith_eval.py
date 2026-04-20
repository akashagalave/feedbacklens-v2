
import os
import json
import time
import random
import requests

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")

FINAL_PASS = 0.7

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text):
    return str(text).lower().replace("_", " ").strip()

def call_pipeline(query: str, company: str):
    start = time.time()
    try:
        res = requests.post(
            f"{GATEWAY_URL}/analyze",
            json={"query": query, "company": company},
            timeout=60
        )
        latency = time.time() - start

        if res.status_code == 200:
            return res.json(), latency

        return {"error": res.status_code}, latency

    except Exception as e:
        return {"error": str(e)}, time.time() - start


def extract_fields(response):
    try:
        if "top_issues" in response:
            return response.get("top_issues", []), response.get("recommendations", [])

        if "issues" in response:
            return response.get("issues", []), response.get("recommendations", [])

        if "analysis" in response:
            return (
                response["analysis"].get("issues", []),
                response["analysis"].get("recommendations", [])
            )
    except Exception:
        pass

    return [], []


def compute_relevance(generated_issues, expected_themes):
    if not generated_issues:
        return 0.0

    issues_text = clean_text(" ".join(generated_issues))
    themes_text = clean_text(" ".join(expected_themes))

    emb1 = embed_model.encode([issues_text])
    emb2 = embed_model.encode([themes_text])

    score = cosine_similarity(emb1, emb2)[0][0]

    # 🔥 small boost if strong keyword overlap
    overlap = set(issues_text.split()) & set(themes_text.split())
    if len(overlap) >= 2:
        score += 0.05

    return round(min(float(score), 1.0), 2)


def evaluate_recommendations(recommendations):
    if not recommendations:
        return 0.0

    text = clean_text(" ".join(recommendations))

    generic_phrases = [
        "improve service",
        "do better",
        "enhance experience",
        "take action",
        "fix issues"
    ]

    actionable_keywords = [
        "refund", "compensation", "automate",
        "optimize", "reduce", "track",
        "monitor", "alert", "train",
        "delivery", "cancel", "support",
        "pricing", "fare"
    ]

    strong_patterns = [
        "auto refund",
        "real time tracking",
        "reduce delivery time",
        "improve driver allocation",
        "implement monitoring",
        "optimize routing",
        "dynamic pricing control"
    ]

    is_generic = any(p in text for p in generic_phrases)
    has_strong = any(p in text for p in strong_patterns)
    has_actionable = any(k in text for k in actionable_keywords)

    if has_strong:
        return 0.9

    if has_actionable and not is_generic:
        return 0.75

    if len(text.split()) > 12:
        return 0.6

    if has_actionable:
        return 0.5

    return 0.3


def evaluate_case(case):
    query = case["query"]
    company = case["company"]
    expected = case["expected_themes"]

    response, latency = call_pipeline(query, company)

    issues, recs = extract_fields(response)

    relevance = compute_relevance(issues, expected)
    rec_score = evaluate_recommendations(recs)

    
    final_score = round((relevance * 0.7) + (rec_score * 0.3), 2)
    passed = final_score >= FINAL_PASS

    return {
        "query": query,
        "issues": issues,
        "relevance": relevance,
        "rec_score": rec_score,
        "final": final_score,
        "passed": passed,
        "latency": round(latency, 2)
    }


EVAL_DATASET = [
    {"query": "Swiggy refund problems", "company": "swiggy", "expected_themes": ["refund", "money", "delay"]},
    {"query": "uber pricing issues", "company": "uber", "expected_themes": ["pricing", "surge", "fare"]},
    {"query": "zomato wrong order", "company": "zomato", "expected_themes": ["wrong", "missing", "order"]},
    {"query": "why is delivery late", "company": "swiggy", "expected_themes": ["delay", "late", "delivery"]},
    {"query": "uber driver cancel problem", "company": "uber", "expected_themes": ["cancel", "driver"]},

   
    {"query": "swiggi delivry prblms", "company": "swiggy", "expected_themes": ["delivery", "delay"]},
    {"query": "food arrived cold again", "company": "zomato", "expected_themes": ["cold", "quality"]},
    {"query": "too many charges on my ride", "company": "uber", "expected_themes": ["pricing", "fare"]},
]

random.shuffle(EVAL_DATASET)


def run_evaluation():
    print("\n🔥 FEEDBACKLENS — FINAL EVAL SYSTEM\n")

    results = []
    pass_count = 0
    total_latency = 0

    for i, case in enumerate(EVAL_DATASET):
        res = evaluate_case(case)
        results.append(res)

        total_latency += res["latency"]

        if res["passed"]:
            pass_count += 1

        status = "✅" if res["passed"] else "❌"

        print(
            f"[{i+1}/{len(EVAL_DATASET)}] {res['query']}\n"
            f"Issues: {res['issues']}\n"
            f"Relevance: {res['relevance']} | Rec: {res['rec_score']} | Final: {res['final']} | {status}\n"
        )

    avg_score = round(sum(r["final"] for r in results) / len(results), 3)
    avg_latency = round(total_latency / len(results), 2)

    print("\n==============================")
    print("FINAL SUMMARY")
    print("==============================")
    print(f"Pass Rate: {pass_count}/{len(results)}")
    print(f"Final Score: {avg_score}")
    print(f"Latency: {avg_latency}s")

    os.makedirs("reports", exist_ok=True)
    with open("reports/final_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved → reports/final_eval.json")


if __name__ == "__main__":
    run_evaluation()