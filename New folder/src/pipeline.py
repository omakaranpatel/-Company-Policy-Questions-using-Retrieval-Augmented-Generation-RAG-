"""End-to-end RAG pipeline."""

import json
from pathlib import Path
from typing import Any

from src.config import INDEX_DIR, TOP_K_FINAL
from src.generation.answer_generator import AnswerGenerator
from src.ingestion.chunker import split_long_chunks
from src.ingestion.loader import load_documents
from src.models import AnswerResult
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_expansion import expand_query
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore

FEEDBACK_PATH = INDEX_DIR / "feedback.jsonl"


class PolicyRAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        self.retriever = HybridRetriever(self.vector_store, self.bm25_index)
        self.reranker = Reranker()
        self.generator = AnswerGenerator()

    def index_documents(
        self,
        documents_dir: Path,
        use_ocr: bool = False,
        incremental: bool = False,
    ) -> int:
        new_chunks = split_long_chunks(load_documents(documents_dir, use_ocr=use_ocr))
        if not new_chunks:
            return 0

        if incremental and INDEX_DIR.exists():
            existing = self._load_existing_chunk_ids()
            new_chunks = [c for c in new_chunks if c.chunk_id not in existing]

        if not new_chunks:
            return 0

        self.vector_store.add_chunks(new_chunks)

        if self.bm25_index.load():
            all_chunks = self.bm25_index._chunks + new_chunks
        else:
            all_chunks = new_chunks
        self.bm25_index.build(all_chunks)
        self.bm25_index.save()
        self.vector_store.save_manifest(all_chunks)
        return len(new_chunks)

    def is_indexed(self) -> bool:
        return self.vector_store.count() > 0 and self.bm25_index.load()

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        policy_filter: str | None = None,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
    ) -> AnswerResult:
        if not self.is_indexed():
            raise RuntimeError("Index not built. Run scripts/index_documents.py first.")

        search_query = expand_query(question) if use_query_expansion else question
        if history:
            last_user = next((t["content"] for t in reversed(history) if t["role"] == "user"), "")
            if last_user and last_user not in search_query:
                search_query = f"{last_user} {search_query}"

        metadata_filter = {"policy_type": policy_filter} if policy_filter else None
        hits = self.retriever.retrieve(search_query, top_k=TOP_K_FINAL * 2, metadata_filter=metadata_filter)

        if use_reranking and hits:
            hits = self.reranker.rerank(question, hits, top_k=TOP_K_FINAL)
        else:
            hits = hits[:TOP_K_FINAL]

        return self.generator.generate(question, hits, history)

    def stream_ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        policy_filter: str | None = None,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
    ):
        if not self.is_indexed():
            raise RuntimeError("Index not built. Run scripts/index_documents.py first.")

        search_query = expand_query(question) if use_query_expansion else question
        if history:
            last_user = next((t["content"] for t in reversed(history) if t["role"] == "user"), "")
            if last_user and last_user not in search_query:
                search_query = f"{last_user} {search_query}"

        metadata_filter = {"policy_type": policy_filter} if policy_filter else None
        hits = self.retriever.retrieve(search_query, top_k=TOP_K_FINAL * 2, metadata_filter=metadata_filter)

        if use_reranking and hits:
            hits = self.reranker.rerank(question, hits, top_k=TOP_K_FINAL)
        else:
            hits = hits[:TOP_K_FINAL]

        stream = self.generator.stream(question, hits, history)
        return stream, hits

    def record_feedback(
        self,
        question: str,
        answer: str,
        rating: int,
        comment: str = "",
    ) -> None:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "question": question,
            "answer": answer[:500],
            "rating": rating,
            "comment": comment,
        }
        with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_existing_chunk_ids(self) -> set[str]:
        manifest_path = INDEX_DIR / "manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if "chunk_ids" in data:
                    return set(data["chunk_ids"])
            except Exception:
                pass
        if self.bm25_index.load():
            return {c.chunk_id for c in self.bm25_index._chunks}
        return set()
