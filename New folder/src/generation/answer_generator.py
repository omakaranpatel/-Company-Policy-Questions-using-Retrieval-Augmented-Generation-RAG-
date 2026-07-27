"""Generate grounded answers with citations using OpenAI or fallback mode."""

import os
import re
from collections.abc import Iterator

from src.config import MAX_CONTEXT_CHUNKS, OPENAI_MODEL, TEMPERATURE
from src.models import AnswerResult, Citation, RetrievedChunk

SYSTEM_PROMPT = """You are an internal HR/Finance policy assistant. Answer employee questions about reimbursement and company policies.

Rules:
1. Use ONLY the provided context. Do not use outside knowledge.
2. If the context does not contain enough information, say: "I could not find sufficient information in the provided policy documents to answer this question."
3. Be concise, accurate, and professional.
4. When citing, reference the document title and section from the context headers.
5. For follow-up questions, use the conversation history for context but still ground facts in the policy context.
6. Never invent dollar amounts, deadlines, or approval rules not present in the context."""

USER_TEMPLATE = """Conversation history:
{history}

Policy context:
{context}

Question: {question}

Provide a clear answer. End with a "Sources:" section listing each document and section used."""


class AnswerGenerator:
    def __init__(self, use_openai: bool | None = None):
        self._use_openai_override = use_openai

    @property
    def use_openai(self) -> bool:
        if self._use_openai_override is not None:
            return self._use_openai_override
        return bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))

    def _get_client_and_model(self):
        from openai import OpenAI

        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        key = groq_key or openai_key

        if key and (key.startswith("gsk_") or groq_key):
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
            return client, model

        model = os.getenv("OPENAI_MODEL", OPENAI_MODEL)
        client = OpenAI(api_key=openai_key) if openai_key else OpenAI()
        return client, model

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> AnswerResult:
        context_chunks = chunks[:MAX_CONTEXT_CHUNKS]
        context = self._format_context(context_chunks)
        history_text = self._format_history(history or [])

        if self.use_openai:
            answer = self._generate_openai(question, context, history_text)
        else:
            answer = self._generate_fallback(question, context_chunks)

        citations = self._build_citations(context_chunks)
        insufficient = self._is_insufficient(answer, context_chunks)

        return AnswerResult(
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
            insufficient_info=insufficient,
        )

    def stream(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        if not self.use_openai:
            result = self.generate(question, chunks, history)
            yield result.answer
            return

        client, model = self._get_client_and_model()
        context = self._format_context(chunks[:MAX_CONTEXT_CHUNKS])
        history_text = self._format_history(history or [])
        prompt = USER_TEMPLATE.format(history=history_text, context=context, question=question)

        stream = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _generate_openai(self, question: str, context: str, history_text: str) -> str:
        client, model = self._get_client_and_model()
        prompt = USER_TEMPLATE.format(history=history_text, context=context, question=question)
        response = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def _generate_fallback(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks or chunks[0].score < 0.15:
            return (
                "I could not find sufficient information in the provided policy documents "
                "to answer this question."
            )

        relevant = [c for c in chunks if c.score >= 0.15][:3]
        if not relevant:
            return (
                "I could not find sufficient information in the provided policy documents "
                "to answer this question."
            )

        lines = [
            "Based on the policy documents, here is what I found:",
            "",
        ]
        for hit in relevant:
            excerpt = self._first_relevant_sentences(hit.chunk.text, question)
            lines.append(f"**{hit.chunk.document_title} — {hit.chunk.section}**")
            lines.append(excerpt)
            lines.append("")

        lines.append("Sources:")
        seen = set()
        for hit in relevant:
            key = (hit.chunk.document_title, hit.chunk.section)
            if key not in seen:
                lines.append(f"- {hit.chunk.document_title} — {hit.chunk.section} ({hit.chunk.document_id})")
                seen.add(key)

        lines.append("")
        lines.append(
            "_Note: Running in fallback mode without an LLM. Set GROQ_API_KEY or OPENAI_API_KEY for richer answers._"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, hit in enumerate(chunks, start=1):
            c = hit.chunk
            parts.append(
                f"[Context {i}] Document: {c.document_title} | Section: {c.section} | ID: {c.document_id}\n{c.text}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _format_history(history: list[dict[str, str]]) -> str:
        if not history:
            return "(none)"
        lines = []
        for turn in history[-6:]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        for hit in chunks:
            c = hit.chunk
            key = (c.document_title, c.section)
            if key in seen:
                continue
            seen.add(key)
            excerpt = c.text[:240].replace("\n", " ").strip()
            citations.append(
                Citation(
                    source_file=c.source_file,
                    document_title=c.document_title,
                    document_id=c.document_id,
                    section=c.section,
                    excerpt=excerpt + ("..." if len(c.text) > 240 else ""),
                )
            )
        return citations

    @staticmethod
    def _is_insufficient(answer: str, chunks: list[RetrievedChunk]) -> bool:
        if "could not find sufficient information" in answer.lower():
            return True
        if not chunks:
            return True
        return chunks[0].score < 0.12

    @staticmethod
    def _first_relevant_sentences(text: str, question: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        q_terms = {t.lower() for t in re.findall(r"\w+", question) if len(t) > 3}
        scored = []
        for s in sentences:
            words = set(re.findall(r"\w+", s.lower()))
            overlap = len(words & q_terms)
            if overlap:
                scored.append((overlap, s))
        if scored:
            scored.sort(reverse=True)
            return " ".join(s for _, s in scored[:2])
        return sentences[0][:300] if sentences else text[:300]
