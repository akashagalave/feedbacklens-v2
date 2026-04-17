import httpx
from langgraph.graph import StateGraph, END
from langsmith import traceable
from .state import AgentState
from .config import settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("orchestrator")


def is_retryable(exception):
    if isinstance(exception, httpx.TimeoutException):
        return True
    if isinstance(exception, httpx.ConnectError):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in (429, 503)
    return False


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True
)
async def _call_understanding_agent(query: str, company: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{settings.understanding_agent_url}/understand",
            json={"query": query, "company": company}
        )
        response.raise_for_status()
        return response.json()


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
async def _call_insight_agent(query: str, company: str, focus: str, top_k: int) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.insight_agent_url}/insights",
            json={
                "query":   query,
                "company": company,
                "focus":   focus,
                "top_k":   top_k
            }
        )
        response.raise_for_status()
        return response.json()


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True
)
async def _call_recommendation_agent(company: str, top_issues: list, patterns: list) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{settings.recommendation_agent_url}/recommend",
            json={
                "company":    company,
                "top_issues": top_issues,
                "patterns":   patterns
            }
        )
        response.raise_for_status()
        return response.json()


@traceable(name="understanding-node")
async def understanding_node(state: AgentState) -> AgentState:
    try:
        data = await _call_understanding_agent(
            query=state["query"],
            company=state.get("company")
        )
        state["company"] = data.get("company", state.get("company", "unknown"))
        state["intent"]  = data.get("intent", "analyze")
        state["focus"]   = data.get("focus")
        logger.info(f"Understanding: company={state['company']} intent={state['intent']}")
        return state
    except Exception as e:
        logger.error(f"Understanding agent error after retries: {e}")
        state["error"] = str(e)
        return state


@traceable(name="insight-node")
async def insight_node(state: AgentState) -> AgentState:
    try:
        data = await _call_insight_agent(
            query=state["query"],
            company=state["company"],
            focus=state.get("focus"),
            top_k=int(state.get("top_k") or 10)
        )
        state["top_issues"]       = data.get("top_issues", [])
        state["patterns"]         = data.get("patterns", [])
        state["sample_reviews"]   = data.get("sample_reviews", [])
        state["confidence_score"] = data.get("confidence_score")
        logger.info(f"Insights: {len(state['top_issues'])} issues found")
        return state
    except Exception as e:
        logger.error(f"Insight agent error after retries: {e}")
        state["error"] = str(e)
        return state


@traceable(name="recommendation-node")
async def recommendation_node(state: AgentState) -> AgentState:
    try:
        data = await _call_recommendation_agent(
            company=state["company"],
            top_issues=state.get("top_issues", []),
            patterns=state.get("patterns", [])
        )
        state["recommendations"] = data.get("recommendations", [])
        logger.info(f"Recommendations: {len(state['recommendations'])} generated")
        return state
    except Exception as e:
        logger.error(f"Recommendation agent error after retries: {e}")
        state["error"] = str(e)
        return state


def should_continue(state: AgentState) -> str:
    if state.get("error"):
        logger.warning(f"Ending workflow due to error: {state['error']}")
        return "end"
    if not state.get("top_issues"):
        logger.warning("No issues found — ending early")
        return "end"
    if state.get("top_issues") == ["No data found for this company"]:
        logger.warning("No data found — skipping recommendations")
        return "end"
    return "recommend"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("understanding",  understanding_node)
    graph.add_node("insight",        insight_node)
    graph.add_node("recommendation", recommendation_node)
    graph.set_entry_point("understanding")
    graph.add_edge("understanding", "insight")
    graph.add_conditional_edges(
        "insight",
        should_continue,
        {
            "recommend": "recommendation",
            "end":       END
        }
    )
    graph.add_edge("recommendation", END)
    return graph.compile()


workflow = build_graph()