import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from pydantic import BaseModel
from .config import settings
from .prompts import UNDERSTANDING_PROMPT
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("understanding-agent")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=100,
    timeout=30,
    api_key=settings.openai_api_key
)

KNOWN_COMPANIES = ["swiggy", "zomato", "uber"]


class UnderstandingOutput(BaseModel):
    company: str
    intent: str
    focus: str | None = None


def normalize_company(company: str, query: str):
    if not company:
        return None

    company = company.lower().strip()

    if company in KNOWN_COMPANIES:
        return company

    for c in KNOWN_COMPANIES:
        if c in company:
            return c

    query = query.lower()
    for c in KNOWN_COMPANIES:
        if c in query:
            return c

    return None


@traceable(name="understanding-agent")
async def understand_query(query: str, company: str = None) -> dict:
    logger.info(f"Understanding query: {query}")

    user_message = f"Query: {query}"
    if company:
        user_message += f"\nHint - company might be: {company}"

    try:
        llm_structured = llm.with_structured_output(UnderstandingOutput)

        result = await llm_structured.ainvoke([
            SystemMessage(content=UNDERSTANDING_PROMPT),
            HumanMessage(content=user_message)
        ])

        logger.info(f"Raw LLM output: {result}")

        extracted_company = normalize_company(result.company, query)

        logger.info(f"Final normalized company: {extracted_company}")

        final_response = {
            "company": extracted_company or "unknown",
            "intent": result.intent or "analyze",
            "focus": result.focus
        }

        logger.info(f"Final structured output: {final_response}")
        return final_response

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return {
            "company": normalize_company(company, query) or "unknown",
            "intent": "analyze",
            "focus": None
        }