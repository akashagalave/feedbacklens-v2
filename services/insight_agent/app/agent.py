from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from pydantic import BaseModel
from .config import settings
from .prompts import INSIGHT_PROMPT
from .hybrid_search import hybrid_search
from .cache import make_cache_key, get_cached, set_cache
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("insight-agent")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=500,
    timeout=30,
    api_key=settings.openai_api_key
)


class InsightOutput(BaseModel):
    top_issues: list[str]
    patterns: list[str]
    confidence_score: float


@traceable(name="insight-agent")
async def generate_insights(
    query: str,
    company: str,
    focus: str = None,
    top_k: int = 10
) -> dict:

    cache_key = make_cache_key(query, company, focus)
    cached = await get_cached(cache_key)
    if cached:
        return cached

    chunks = await hybrid_search(query, company, focus, top_k)

    if not chunks:
        logger.warning(f"No chunks found for company={company}")
        return {
            "top_issues": ["No data found for this company"],
            "patterns": [],
            "sample_reviews": [],
            "confidence_score": 0.0
        }

    reviews_text = "\n".join([
        f"- [{c.issue}] {c.review}" for c in chunks
    ])

    sample_reviews = [c.review for c in chunks[:3]]
    logger.info(f"Retrieved {len(chunks)} chunks for {company}")

    try:
        llm_structured = llm.with_structured_output(InsightOutput)

        result = await llm_structured.ainvoke([
            SystemMessage(content=INSIGHT_PROMPT),
            HumanMessage(content=f"Company: {company}\n\nReviews:\n{reviews_text}")
        ])

        final = {
            "top_issues": result.top_issues,
            "patterns": result.patterns,
            "sample_reviews": sample_reviews,
            "confidence_score": result.confidence_score
        }

        await set_cache(cache_key, final)

        logger.info(f"Insights generated for {company}")
        return final

    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise