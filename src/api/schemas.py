from pydantic import BaseModel, Field
from typing import List, Optional

class AnalysisRequest(BaseModel):
    sentence: str = Field(..., example="We may change, suspend, or terminate your access...")
    top_k: int = Field(default=3, ge=1, le=10)

class MatchResult(BaseModel):
    rank: int
    similarity_score: float
    company: str
    text: str
    historical_ground_truth: str

class AnalysisResponse(BaseModel):
    sentence: str
    is_unfair: bool
    status_label: str
    top_matches: List[MatchResult]