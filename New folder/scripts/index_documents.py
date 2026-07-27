"""Index policy documents into the vector store and BM25 index."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DOCUMENTS_DIR
from src.pipeline import PolicyRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Index policy documents for RAG retrieval.")
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=DOCUMENTS_DIR,
        help="Directory containing policy documents",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only index new or changed documents",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR for scanned PDF pages",
    )
    args = parser.parse_args()

    pipeline = PolicyRAGPipeline()
    count = pipeline.index_documents(
        documents_dir=args.documents_dir,
        use_ocr=args.ocr,
        incremental=args.incremental,
    )
    print(f"Indexed {count} chunk(s) from {args.documents_dir}")


if __name__ == "__main__":
    main()
