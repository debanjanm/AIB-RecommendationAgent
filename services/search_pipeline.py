"""
Search pipeline orchestrator.
Ties together LLM query expansion, CLIP embedding, ChromaDB retrieval, and LLM reranking.
"""
import os
import time
from PIL import Image

from models.product import Product
from models.search import SearchRequest, SearchResponse, RankedProduct, QueryExpansion
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.llm_service import LLMService


class SearchPipeline:
    def __init__(
        self,
        embedder: EmbeddingService,
        store: VectorStore,
        llm: LLMService,
        top_k_candidates: int = 20,
        top_k_results: int = 10,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k_candidates = top_k_candidates
        self.top_k_results = top_k_results

    def run(
        self,
        query: str,
        image: Image.Image | None = None,
        top_k: int | None = None,
        text_weight: float = 0.6,
        image_weight: float = 0.4,
    ) -> SearchResponse:
        start = time.perf_counter()
        top_k = top_k or self.top_k_results

        # Step 1: LLM query expansion
        expansion: QueryExpansion = self.llm.expand_query(
            query=query, image_provided=(image is not None)
        )

        # Step 2: Embed the expanded query (+ optional image)
        query_embedding = self.embedder.embed_query(
            text=expansion.expanded_query,
            image=image,
            text_weight=text_weight,
            image_weight=image_weight,
        )

        # Step 3: ChromaDB retrieval
        category_filter = expansion.search_filters.get("category")
        # Only apply category filter if it's a known category (not "unknown")
        if category_filter and category_filter.lower() == "unknown":
            category_filter = None

        n_candidates = min(self.top_k_candidates, self.store.count())
        if n_candidates == 0:
            return SearchResponse(
                original_query=query,
                expanded_query=expansion.expanded_query,
                extracted_intent=expansion.extracted_intent,
                results=[],
                total_candidates=0,
                search_time_ms=0.0,
            )

        candidates = self.store.query(
            embedding=query_embedding,
            n_results=n_candidates,
            category_filter=category_filter,
        )

        # Step 4: LLM reranking
        reranked = self.llm.rerank(
            original_query=query,
            intent=expansion.extracted_intent,
            candidates=candidates,
        )

        # Step 5: Merge scores and build final results
        # Build a lookup from candidate id → cosine similarity
        cosine_scores = {c["id"]: c["similarity_score"] for c in candidates}
        meta_lookup = {c["id"]: c["metadata"] for c in candidates}

        results: list[RankedProduct] = []
        for ranked_item in reranked[:top_k]:
            pid = ranked_item["id"]
            meta = meta_lookup.get(pid)
            if meta is None:
                continue
            product = Product.from_chroma_metadata(meta)
            results.append(
                RankedProduct(
                    product=product,
                    similarity_score=cosine_scores.get(pid, 0.0),
                    rerank_score=float(ranked_item.get("score", 0.0)),
                    match_reason=ranked_item.get("match_reason", ""),
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return SearchResponse(
            original_query=query,
            expanded_query=expansion.expanded_query,
            extracted_intent=expansion.extracted_intent,
            results=results,
            total_candidates=len(candidates),
            search_time_ms=round(elapsed_ms, 1),
        )

    def find_similar(self, product_id: str, top_k: int | None = None) -> SearchResponse:
        """Find products similar to a known product using its stored embedding."""
        top_k = top_k or self.top_k_results
        product = self.store.get_product(product_id)
        if product is None:
            return SearchResponse(
                original_query=f"Similar to product {product_id}",
                expanded_query="",
                extracted_intent={},
                results=[],
                total_candidates=0,
                search_time_ms=0.0,
            )

        embedding = self.store.get_embedding(product_id)
        candidates = self.store.query(
            embedding=embedding,
            n_results=min(self.top_k_candidates + 1, self.store.count()),
        )
        # Exclude the query product itself
        candidates = [c for c in candidates if c["id"] != product_id]

        query_text = f"Similar to: {product.name} by {product.brand}"
        reranked = self.llm.rerank(
            original_query=query_text,
            intent={"category": product.category, "name": product.name},
            candidates=candidates,
        )

        cosine_scores = {c["id"]: c["similarity_score"] for c in candidates}
        meta_lookup = {c["id"]: c["metadata"] for c in candidates}

        results = []
        for ranked_item in reranked[:top_k]:
            pid = ranked_item["id"]
            meta = meta_lookup.get(pid)
            if meta is None:
                continue
            p = Product.from_chroma_metadata(meta)
            results.append(
                RankedProduct(
                    product=p,
                    similarity_score=cosine_scores.get(pid, 0.0),
                    rerank_score=float(ranked_item.get("score", 0.0)),
                    match_reason=ranked_item.get("match_reason", ""),
                )
            )

        return SearchResponse(
            original_query=query_text,
            expanded_query=query_text,
            extracted_intent={"source_product": product_id},
            results=results,
            total_candidates=len(candidates),
            search_time_ms=0.0,
        )
