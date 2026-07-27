"""Integration tests for PolicyRAGPipeline."""

import unittest
from pathlib import Path
from src.config import DOCUMENTS_DIR
from src.pipeline import PolicyRAGPipeline


class TestPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = PolicyRAGPipeline()
        if not cls.pipeline.is_indexed() and DOCUMENTS_DIR.exists():
            cls.pipeline.index_documents(DOCUMENTS_DIR)

    def test_pipeline_is_indexed(self):
        self.assertTrue(self.pipeline.is_indexed())

    def test_ask_valid_policy_question(self):
        if not self.pipeline.is_indexed():
            self.skipTest("Index not present")
        res = self.pipeline.ask("What is the hotel cap for domestic travel?")
        self.assertIsNotNone(res.answer)
        self.assertGreater(len(res.citations), 0)

    def test_ask_out_of_domain_question(self):
        if not self.pipeline.is_indexed():
            self.skipTest("Index not present")
        res = self.pipeline.ask("What is the company policy regarding interstellar space travel?")
        self.assertTrue(res.insufficient_info or "could not find sufficient information" in res.answer.lower())

    def test_incremental_indexing(self):
        existing_ids = self.pipeline._load_existing_chunk_ids()
        self.assertGreater(len(existing_ids), 0)


if __name__ == "__main__":
    unittest.main()
