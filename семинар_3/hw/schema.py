# hw/schema.py
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════
# Round 1 — Information Extraction (IE)
# ══════════════════════════════════════════════════════════

class Issue(BaseModel):
    """Specific problem / complaint"""
    aspect: Literal["performance", "design", "support", "price", "ads", "reliability"] = Field(
        description="App aspect"
    )
    text: str = Field(description="Problem description (1 sentence)")
    quote: str = Field(description="VERBATIM quote from review")


class Like(BaseModel):
    """What the user liked"""
    aspect: Literal["performance", "design", "support", "price", "ads", "reliability"] = Field(
        description="App aspect"
    )
    text: str = Field(description="What exactly they liked")
    quote: str = Field(description="VERBATIM quote from review")


class Review(BaseModel):
    """Single app review"""
    app_name: str = Field(description="App name (Spotify)")
    username: str = Field(default="Anonymous", description="Username")
    rating: int = Field(description="Rating 1-5", ge=1, le=5)
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment"
    )
    likes: list[Like] = Field(default_factory=list, description="What users liked")
    issues: list[Issue] = Field(default_factory=list, description="What users disliked")
    would_recommend: bool = Field(default=True, description="Would recommend")
    
    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError('Rating must be between 1 and 5')
        return v


# ══════════════════════════════════════════════════════════
# Round 2 — Aspect-Based Sentiment Analysis
# ══════════════════════════════════════════════════════════

class AspectSentiment(BaseModel):
    """Single aspect sentiment"""
    aspect: Literal["performance", "design", "support", "price", "ads", "reliability"]
    sentiment: Literal["positive", "neutral", "negative"]
    quote: str


class AppAspects(BaseModel):
    """Aspect analysis for one review"""
    app_name: str
    aspects: list[AspectSentiment]


# ══════════════════════════════════════════════════════════
# Round 3 — Map-Reduce Summary
# ══════════════════════════════════════════════════════════

class ReviewSummary(BaseModel):
    """Summary of one review"""
    key_points: list[str]
    app_name: str
    rating: int
    sentiment: Literal["positive", "neutral", "negative"]


class AggregateReport(BaseModel):
    """Final summary across all reviews"""
    headline: str = Field(description="Main takeaway (1 sentence)")
    key_findings: list[str] = Field(description="Key findings (5-7)")
    common_praise: list[str] = Field(description="What users like (3-5)")
    common_criticism: list[str] = Field(description="What users dislike (3-5)")
    recommendations: list[str] = Field(description="Recommendations (3-5)")


# ══════════════════════════════════════════════════════════
# Round 5 — LLM-as-Judge
# ══════════════════════════════════════════════════════════

class ActionVerdict(BaseModel):
    """Verdict for one recommendation"""
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str]
    comment: str


class JudgeReport(BaseModel):
    """Judge's report"""
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str


# ══════════════════════════════════════════════════════════
# Round 2.5 — Autodiscovery (bonus)
# ══════════════════════════════════════════════════════════

class DiscoveredAspect(BaseModel):
    name: str = Field(description="Topic name (2-4 words)")
    description: str = Field(description="Brief topic description")


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(description="List of topics (5-8 items)")