"""Unit tests for document ingestion, section parsing, and chunking."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.ingestion.chunker import split_long_chunks
from src.ingestion.loader import _extract_document_id, _extract_title, _policy_type, load_markdown
from src.models import DocumentChunk


class TestIngestion(unittest.TestCase):
    def test_extract_document_id(self):
        text = "# Title\n\n**Document ID:** POL-2024-EXPENSE\n\nBody content."
        doc_id = _extract_document_id(text)
        self.assertEqual(doc_id, "POL-2024-EXPENSE")

    def test_extract_title(self):
        text = "# Expense Reimbursement Policy\n\n## 1. Overview\nText"
        title = _extract_title(text, fallback="Fallback Title")
        self.assertEqual(title, "Expense Reimbursement Policy")

    def test_policy_type(self):
        self.assertEqual(_policy_type("expense_policy.md"), "expense")
        self.assertEqual(_policy_type("travel_policy.md"), "travel")
        self.assertEqual(_policy_type("finance_policy.md"), "finance")
        self.assertEqual(_policy_type("employee_handbook.md"), "handbook")
        self.assertEqual(_policy_type("custom_doc.md"), "general")

    def test_load_markdown(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_policy.md"
            path.write_text(
                "# Test Policy\n"
                "**Document ID:** DOC-001\n\n"
                "## Section One\n"
                "This is the body of section one.\n\n"
                "## Section Two\n"
                "This is the body of section two.\n",
                encoding="utf-8",
            )
            chunks = load_markdown(path)
            self.assertEqual(len(chunks), 3)
            self.assertEqual(chunks[0].section, "Introduction")
            self.assertEqual(chunks[1].section, "Section One")
            self.assertEqual(chunks[2].section, "Section Two")

    def test_split_long_chunks(self):
        long_text = "A" * 1000
        chunk = DocumentChunk(
            chunk_id="test_0",
            text=long_text,
            source_file="test.md",
            document_title="Test",
            document_id="DOC-1",
            section="General",
            chunk_index=0,
        )
        split = split_long_chunks([chunk])
        self.assertGreater(len(split), 1)
        self.assertLessEqual(len(split[0].text), 512)


if __name__ == "__main__":
    unittest.main()
