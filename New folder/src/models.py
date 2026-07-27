"""Shared data models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_file: str
    document_title: str
    document_id: str
    section: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    retrieval_method: str = "hybrid"


@dataclass
class Citation:
    source_file: str
    document_title: str
    document_id: str
    section: str
    excerpt: str

    def format(self) -> str:
        return f"[{self.document_title} — {self.section}] ({self.document_id})"


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    insufficient_info: bool = False
