from typing import TypedDict

class AgentState(TypedDict):
    query: str
    company: str | None
    intent: str | None
    focus: str | None
    top_k: int | None
    top_issues: list[str] | None
    patterns: list[str] | None
    sample_reviews: list[str] | None
    confidence_score: float | None
    recommendations: list[str] | None
    error: str | None
    run_id: str | None