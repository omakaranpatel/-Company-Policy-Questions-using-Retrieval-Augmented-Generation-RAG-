"""Split long sections into overlapping chunks."""

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.models import DocumentChunk


def split_long_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    result: list[DocumentChunk] = []
    for chunk in chunks:
        text = chunk.text
        if len(text) <= CHUNK_SIZE:
            result.append(chunk)
            continue

        start = 0
        sub_idx = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            sub_text = text[start:end]
            sub_id = f"{chunk.chunk_id}_part{sub_idx}"
            result.append(
                DocumentChunk(
                    chunk_id=sub_id,
                    text=sub_text,
                    source_file=chunk.source_file,
                    document_title=chunk.document_title,
                    document_id=chunk.document_id,
                    section=chunk.section,
                    chunk_index=chunk.chunk_index * 100 + sub_idx,
                    metadata=dict(chunk.metadata),
                )
            )
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP
            sub_idx += 1
    return result
