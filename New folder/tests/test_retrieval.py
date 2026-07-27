"""Unit tests for retrieval modules (BM25, query expansion, hybrid fusion)."""

import unittest
from src.models import DocumentChunk, RetrievedChunk
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_expansion import expand_query


class TestRetrieval(unittest.TestCase):
    def test_query_expansion(self):
        query = "What is the per diem for travel?"
        expanded = expand_query(query)
        self.assertIn("daily allowance", expanded)
        self.assertIn("trip", expanded)

    def test_bm25_index(self):
        chunks = [
            DocumentChunk(
                chunk_id="c1",
                text="Employees are entitled to meal per diem during business trips.",
                source_file="travel.md",
                document_title="Travel Policy",
                document_id="POL-TRV",
                section="Per Diem",
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="c2",
                text="Expense reports must be submitted within 30 calendar days.",
                source_file="expense.md",
                document_title="Expense Policy",
                document_id="POL-EXP",
                section="Submissions",
                chunk_index=1,
            ),
            DocumentChunk(
                chunk_id="c3",
                text="General company information and handbook guidelines for employees.",
                source_file="handbook.md",
                document_title="Handbook",
                document_id="POL-HNB",
                section="General",
                chunk_index=2,
            ),
            DocumentChunk(
                chunk_id="c4",
                text="Finance approval is required for all capital expenditures exceeding limit.",
                source_file="finance.md",
                document_title="Finance Policy",
                document_id="POL-FIN",
                section="Approvals",
                chunk_index=3,
            ),
        ]
        bm25 = BM25Index()
        bm25.build(chunks)
        results = bm25.search("expense reports", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk.chunk_id, "c2")

    def test_reciprocal_rank_fusion(self):
        c1 = DocumentChunk("c1", "text1", "f1.md", "Title1", "ID1", "Sec1", 0)
        c2 = DocumentChunk("c2", "text2", "f2.md", "Title2", "ID2", "Sec2", 1)

        vec_hits = [RetrievedChunk(c1, score=0.9), RetrievedChunk(c2, score=0.7)]
        bm25_hits = [RetrievedChunk(c2, score=1.0), RetrievedChunk(c1, score=0.5)]

        retriever = HybridRetriever(vector_store=None, bm25_index=None)
        fused = retriever._reciprocal_rank_fusion(vec_hits, bm25_hits)
        self.assertEqual(len(fused), 2)


if __name__ == "__main__":
    unittest.main()
