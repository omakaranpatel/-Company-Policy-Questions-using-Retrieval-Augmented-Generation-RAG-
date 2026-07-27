import src.utils  # noqa: F401
from src.models import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str | None = None):
        from src.config import RERANKER_MODEL

        self.model_name = model_name or RERANKER_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import os
            os.environ["TQDM_DISABLE"] = "1"
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = self.model.predict(pairs)
        scored = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
        return [
            RetrievedChunk(chunk=hit.chunk, score=float(score), retrieval_method="reranked")
            for hit, score in scored[:top_k]
        ]
