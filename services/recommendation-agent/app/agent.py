import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from .config import settings
from .prompts import RECOMMENDATION_PROMPT
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("recommendation-agent")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=500,
    timeout=30,
    api_key=settings.openai_api_key
)


@traceable(name="recommendation-agent")
async def generate_recommendations(
    company: str,
    top_issues: list[str],
    patterns: list[str]
) -> dict:

    context = f"""
Company: {company}

Top Issues:
{chr(10).join(f"- {issue}" for issue in top_issues)}

Patterns:
{chr(10).join(f"- {pattern}" for pattern in patterns)}
"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=RECOMMENDATION_PROMPT),
            HumanMessage(content=context)
        ])

        content = response.content.strip()
        logger.info(f"RAW LLM CONTENT: {content}")

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        logger.info(f"Recommendations generated for {company}")
        return {"recommendations": result.get("recommendations", [])}

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return {"recommendations": ["Unable to generate recommendations"]}
    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise