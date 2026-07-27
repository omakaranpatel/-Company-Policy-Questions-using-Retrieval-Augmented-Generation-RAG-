"""Hybrid vector + BM25 retrieval with score fusion."""

from typing import Any

from src.config import (
    HYBRID_BM25_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    TOP_K_BM25,
    TOP_K_FINAL,
    TOP_K_VECTOR,
)
from src.models import DocumentChunk, RetrievedChunk
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import VectorStore


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        vector_hits = self.vector_store.search(query, top_k=TOP_K_VECTOR, metadata_filter=metadata_filter)
        bm25_hits = self.bm25_index.search(query, top_k=TOP_K_BM25)

        if metadata_filter:
            bm25_hits = [
                hit
                for hit in bm25_hits
                if all(hit.chunk.metadata.get(k) == v for k, v in metadata_filter.items())
            ]

        fused = self._reciprocal_rank_fusion(vector_hits, bm25_hits)
        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_hits: list[RetrievedChunk],
        bm25_hits: list[RetrievedChunk],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, DocumentChunk] = {}

        for rank, hit in enumerate(vector_hits):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + HYBRID_VECTOR_WEIGHT / (k + rank + 1)
            chunks[cid] = hit.chunk

        for rank, hit in enumerate(bm25_hits):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + HYBRID_BM25_WEIGHT / (k + rank + 1)
            chunks[cid] = hit.chunk

        ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        return [
            RetrievedChunk(chunk=chunks[cid], score=scores[cid], retrieval_method="hybrid")
            for cid in ranked_ids
        ]
