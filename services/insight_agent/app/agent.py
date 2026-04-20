
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from sentence_transformers import SentenceTransformer
from .config import settings
from .prompts import INSIGHT_PROMPT
from .hybrid_search import hybrid_search
from .cache import make_cache_key, get_cached, set_cache
import json
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("insight-agent")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=300,
    timeout=30,
    api_key=settings.openai_api_key
)

_filter_model = None

def get_filter_model():
    global _filter_model
    if _filter_model is None:
        _filter_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _filter_model



def normalize_query(query: str) -> str:
    original_query = query
    query = query.lower()

    # 🔤 typo correction
    corrections = {
        "delivry": "delivery", "delievry": "delivery", "dlivery": "delivery",
        "prblms": "problems", "problms": "problems",
        "wrng": "wrong", "wrang": "wrong",
        "issus": "issues", "isues": "issues", "isssues": "issues",
        "pricng": "pricing", "priing": "pricing",
        "drver": "driver", "drivr": "driver",
        "custmer": "customer", "costomer": "customer",
        "behaviur": "behavior", "behavoir": "behavior",
        "recieved": "received", "recived": "received",
        "nit": "not", "wont": "not",
        "vry": "very", "veyr": "very",
        "zomto": "zomato", "swiggi": "swiggy",
    }

    words = query.split()
    corrected = [corrections.get(w, w) for w in words]
    normalized = " ".join(corrected)

    #Semantic synonym mapping
    synonyms = {
        "charges": "pricing",
        "expensive": "pricing",
        "costly": "pricing",
        "price": "pricing",

        "late": "delay",
        "slow": "delay",
        "waiting": "delay",

        "cancel": "cancellation",
        "cancelled": "cancellation",

        "cold": "food_quality",
        "bad food": "food_quality",

        "charges": "pricing",
        "bill": "pricing",
        "fare": "pricing",

        "wrong": "incorrect",
        "missing": "missing"
    }

    for word, mapped in synonyms.items():
        if word in normalized:
            normalized += f" ({mapped})"

    # logging
    if normalized != original_query.lower():
        logger.info(f"Query normalized: '{original_query}' → '{normalized}'")

    return normalized


COMPANY_DEFAULT_ISSUES = {
    "swiggy":  ["delivery delays", "high delivery charges", "poor customer support"],
    "zomato":  ["delivery delays", "wrong orders", "refund issues"],
    "uber":    ["surge pricing", "driver cancellations", "long wait times"],
    "unknown": ["service issues", "delivery problems", "support unresponsive"]
}

def standardize_issues(issues: list) -> list:
    mapping = {
        "pricing issue": ["charge", "charges", "price", "pricing", "fare", "cost", "expensive", "surge", "bill", "amount"],
        "delivery delay": ["delay", "late", "slow", "delivery"],
        "cancellation issue": ["cancel", "cancelled", "cancellation"],
        "food quality issue": ["cold", "stale", "quality", "taste"],
        "order accuracy issue": ["wrong", "missing", "incorrect", "order"],
        "refund issue": ["refund", "money", "payment"]
    }

    standardized = []

    for issue in issues:
        issue_lower = issue.lower()
        matched_label = None

        for label, keywords in mapping.items():
            if any(k in issue_lower for k in keywords):
                matched_label = label
                break

        if matched_label:
            standardized.append(matched_label)

    
    return list(dict.fromkeys(standardized))[:3]
def is_vague_query(query: str) -> bool:
    words = query.strip().split()
    vague_terms = {"bad", "problems", "issues", "feedback", "complaints", "good", "wrong", "worst", "best"}
    return len(words) <= 2 or all(w.lower() in vague_terms for w in words)


def filter_chunks_by_query(query: str, chunks: list, threshold: float = 0.25) -> list:
    if not chunks:
        return chunks
    model = get_filter_model()
    query_emb = model.encode(query)
    scored = []
    for chunk in chunks:
        chunk_emb = model.encode(chunk.review)
        sim = float(np.dot(query_emb, chunk_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb) + 1e-8
        ))
        scored.append((chunk, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    filtered = [c for c, s in scored if s >= threshold]
    logger.info(f"Filtered {len(chunks)} → {len(filtered)} chunks (threshold={threshold})")
    return filtered[:8] if filtered else [c for c, _ in scored[:5]]


@traceable(name="insight-agent")
async def generate_insights(
    query: str,
    company: str,
    focus: str = None,
    top_k: int = 15
) -> dict:

    
    normalized_query = normalize_query(query)

    cache_key = make_cache_key(normalized_query, company, focus)
    cached = await get_cached(cache_key)
    if cached:
        return cached

    
    if is_vague_query(normalized_query):
        logger.info(f"Vague query detected: '{normalized_query}' — using company defaults")
        company_key = company.lower() if company else "unknown"
        default_issues = COMPANY_DEFAULT_ISSUES.get(company_key, COMPANY_DEFAULT_ISSUES["unknown"])
        return {
            "top_issues": default_issues,
            "patterns": ["common recurring complaints"],
            "sample_reviews": [],
            "confidence_score": 0.5
        }

    
    chunks = await hybrid_search(normalized_query, company, focus, top_k)

    if not chunks:
        logger.warning(f"No chunks found for company={company}")
        return {
            "top_issues": ["No data found for this company"],
            "patterns": [],
            "sample_reviews": [],
            "confidence_score": 0.0
        }

   
    filtered_chunks = filter_chunks_by_query(normalized_query, chunks, threshold=0.25)
    reviews_text = "\n".join([
    f"- {c.review}" for c in filtered_chunks[:8]
    ])
    sample_reviews = [c.review for c in filtered_chunks[:3]]
    logger.info(f"Using {len(filtered_chunks)} filtered chunks for {company}")

    
    user_prompt = f"""Query: "{normalized_query}"
Company: {company}
Focus: {focus if focus else 'general'}

IMPORTANT: Return ONLY issues directly related to "{normalized_query}".
Ignore issues unrelated to the query.

Reviews:
{reviews_text}"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=INSIGHT_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)

  
        raw_issues = result.get("top_issues", [])

        standardized = standardize_issues(raw_issues)

 
        if not standardized:
            logger.warning("Standardization failed → applying fallback mapping")

            fallback_map = {
            "delivery": "delivery delay",
            "delay": "delivery delay",
            "late": "delivery delay",

            "price": "pricing issue",
            "charge": "pricing issue",
            "cost": "pricing issue",
            "fare": "pricing issue",
            "bill": "pricing issue",

            "refund": "refund issue",
            "money": "refund issue",

            "cancel": "cancellation issue",

            "cold": "food quality issue",
            "food": "food quality issue",

            "wrong": "order accuracy issue",
            "missing": "order accuracy issue",
            "order": "order accuracy issue",
            "mix": "order accuracy issue"
        }

            forced = []
            for issue in raw_issues:
                issue_lower = issue.lower()
                for k, v in fallback_map.items():
                    if k in issue_lower:
                        forced.append(v)
                        break

            standardized = list(dict.fromkeys(forced))

        top_issues = []

        top_issues.extend(standardized)

        for issue in raw_issues:
            issue_lower = issue.lower()

            if "cold" in issue_lower:
                normalized_issue = "cold food"
            elif "charge" in issue_lower or "cost" in issue_lower or "price" in issue_lower:
                normalized_issue = "charges"
            elif "late" in issue_lower or "delay" in issue_lower:
                normalized_issue = "late delivery"
            elif "cancel" in issue_lower:
                normalized_issue = "ride cancelled"
            elif "wrong" in issue_lower or "missing" in issue_lower:
                normalized_issue = "wrong order"
            else:
                normalized_issue = issue_lower

            if 1 <= len(normalized_issue.split()) <= 5:
                top_issues.append(normalized_issue)

        semantic_boost = {
            "pricing issue": ["charges", "high cost", "expensive"],
            "delivery delay": ["late delivery", "slow delivery"],
            "food quality issue": ["cold food", "bad food"],
            "refund issue": ["refund delay", "money not returned"],
            "cancellation issue": ["ride cancelled", "driver cancelled"],
            "order accuracy issue": ["wrong order", "missing items"]
            }

        boosted = []
        for issue in top_issues:
            clean_issue = issue.strip().lower()
            boosted.append(clean_issue)

            for key in semantic_boost:
                if key in clean_issue:
                    boosted.extend(semantic_boost[key])
        top_issues = boosted
        top_issues = list(dict.fromkeys(top_issues))[:4]

        if not top_issues:
            top_issues = ["general issue"]

        patterns = result.get("patterns", [])[:2]
        confidence = result.get("confidence_score", 0.0)

        final = {
            "top_issues": top_issues,
            "patterns": patterns,
            "sample_reviews": sample_reviews,
            "confidence_score": confidence
        }

        await set_cache(cache_key, final)
        logger.info(f"Final standardized insights: {top_issues}")

        return final

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return {
        "top_issues": ["error parsing insights"],
        "patterns": [],
        "sample_reviews": sample_reviews,
        "confidence_score": 0.0
    }

    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise