"""ChromaDB vector store with metadata filtering."""

import json
from pathlib import Path
from typing import Any

import src.utils  # noqa: F401

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, EMBEDDING_MODEL, MANIFEST_PATH
from src.models import DocumentChunk, RetrievedChunk


class VectorStore:
    def __init__(self, persist_dir: Path | None = None):
        self.persist_dir = persist_dir or CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="policy_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder: SentenceTransformer | None = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            import os
            os.environ["TQDM_DISABLE"] = "1"
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._sanitize_metadata(c.metadata) for c in chunks]
        embeddings = self.embed(documents)
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(
        self,
        query: str,
        top_k: int = 12,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        where = self._build_where(metadata_filter) if metadata_filter else None
        query_embedding = self.embed([query])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved

        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = 1.0 - distance
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=results["documents"][0][i],
                source_file=meta.get("source_file", ""),
                document_title=meta.get("document_title", ""),
                document_id=meta.get("document_id", ""),
                section=meta.get("section", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata=meta,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=score, retrieval_method="vector"))
        return retrieved

    def count(self) -> int:
        return self._collection.count()

    def save_manifest(self, chunks: list[DocumentChunk]) -> None:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "chunk_count": len(chunks),
            "chunk_ids": [c.chunk_id for c in chunks],
            "sources": sorted({c.source_file for c in chunks}),
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean

    @staticmethod
    def _build_where(metadata_filter: dict[str, Any]) -> dict[str, Any]:
        if len(metadata_filter) == 1:
            key, value = next(iter(metadata_filter.items()))
            return {key: value}
        return {"$and": [{k: v} for k, v in metadata_filter.items()]}
