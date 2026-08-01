from pydantic import BaseModel
from typing import Dict, Any

class OptimizationMetrics(BaseModel):
    original_tokens: int
    optimized_tokens: int
    token_reduction_rate: float
    original_sentence_count: int
    optimized_sentence_count: int
    duplicate_sentences_removed: int
    stage_tokens: Dict[str, int]

class OptimizationResult(BaseModel):
    id: str
    filename: str
    optimized_text: str
    assembled_prompt: str
    metrics: OptimizationMetrics
    strategy: Dict[str, Any]
