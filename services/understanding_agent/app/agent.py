import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
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
        response = await llm.ainvoke([
            SystemMessage(content=UNDERSTANDING_PROMPT),
            HumanMessage(content=user_message)
        ])

        content = response.content.strip()
        logger.info(f"RAW LLM CONTENT: {content}")

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        extracted_company = normalize_company(result.get("company", ""), query)

        final_response = {
            "company": extracted_company or "unknown",
            "intent": result.get("intent", "analyze"),
            "focus": result.get("focus")
        }

        logger.info(f"Final structured output: {final_response}")
        return final_response

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return {
            "company": normalize_company(company, query) or "unknown",
            "intent": "analyze",
            "focus": None
        }
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return {
            "company": normalize_company(company, query) or "unknown",
            "intent": "analyze",
            "focus": None
        }