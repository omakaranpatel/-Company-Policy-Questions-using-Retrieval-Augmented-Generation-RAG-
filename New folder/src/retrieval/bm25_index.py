"""BM25 keyword index for hybrid search."""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import BM25_PATH
from src.models import DocumentChunk, RetrievedChunk


class BM25Index:
    def __init__(self):
        self._chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        tokenized = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 12) -> list[RetrievedChunk]:
        if not self._bm25 or not self._chunks:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[RetrievedChunk] = []
        max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
        for idx, score in ranked:
            if score <= 0:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=float(score / max_score),
                    retrieval_method="bm25",
                )
            )
        return results

    def save(self, path: Path | None = None) -> None:
        path = path or BM25_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, f)

    def load(self, path: Path | None = None) -> bool:
        path = path or BM25_PATH
        if not path.exists():
            return False
        with path.open("rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._bm25 = data["bm25"]
        return True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"\w+", text.lower())
