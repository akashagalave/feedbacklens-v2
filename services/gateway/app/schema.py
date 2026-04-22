from pydantic import BaseModel, Field, field_validator
from typing import Optional


class QueryRequest(BaseModel):
    query: str
    company: Optional[str] = None
    top_k: Optional[int] = 10


class BatchRequest(BaseModel):
    company: str
    reviews: list[str] = Field(min_length=1, max_length=20)  # max 20 reviews per batch

    @field_validator("reviews")
    @classmethod
    def validate_review_content(cls, reviews: list[str]) -> list[str]:
        for i, review in enumerate(reviews):
            if len(review.strip()) < 5:
                raise ValueError(f"Review[{i}] too short (min 5 chars)")
            if len(review) > 2000:
                raise ValueError(f"Review[{i}] too long (max 2000 chars)")
        return reviews