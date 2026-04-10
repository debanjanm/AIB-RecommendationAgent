from pydantic import BaseModel
from typing import Optional
from models.product import Product


class SearchRequest(BaseModel):
    query: str
    image_base64: Optional[str] = None
    top_k: int = 10
    text_weight: float = 0.6
    image_weight: float = 0.4


class QueryExpansion(BaseModel):
    expanded_query: str
    extracted_intent: dict
    search_filters: dict


class RankedProduct(BaseModel):
    product: Product
    similarity_score: float   # from ChromaDB cosine distance
    rerank_score: float       # from GPT-4o reranking
    match_reason: str         # GPT-4o explanation


class SearchResponse(BaseModel):
    original_query: str
    expanded_query: str
    extracted_intent: dict
    results: list[RankedProduct]
    total_candidates: int
    search_time_ms: float
