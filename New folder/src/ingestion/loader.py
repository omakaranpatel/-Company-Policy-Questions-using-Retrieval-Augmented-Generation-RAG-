"""Load policy documents from disk (markdown, text, PDF with optional OCR)."""

import re
from pathlib import Path

from src.models import DocumentChunk


def _extract_document_id(text: str) -> str:
    match = re.search(r"\*\*Document ID:\*\*\s*(.+)", text)
    return match.group(1).strip() if match else "UNKNOWN"


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown by ## headers into (section_title, section_body) pairs."""
    parts = re.split(r"(?=^##\s+)", text, flags=re.MULTILINE)
    sections: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        if lines[0].startswith("## "):
            title = lines[0][3:].strip()
            body = "\n".join(lines[1:]).strip()
        elif lines[0].startswith("# "):
            title = "Introduction"
            body = part
        else:
            title = "General"
            body = part
        if body:
            sections.append((title, body))
    return sections if sections else [("General", text)]


def load_markdown(path: Path) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8")
    doc_id = _extract_document_id(text)
    title = _extract_title(text, path.stem.replace("_", " ").title())
    sections = _split_sections(text)
    chunks: list[DocumentChunk] = []
    idx = 0
    for section_title, section_body in sections:
        header = f"# {title}\n## {section_title}\n\n"
        chunk = DocumentChunk(
            chunk_id=f"{path.stem}_{idx}",
            text=header + section_body,
            source_file=path.name,
            document_title=title,
            document_id=doc_id,
            section=section_title,
            chunk_index=idx,
            metadata={
                "source_file": path.name,
                "document_title": title,
                "document_id": doc_id,
                "section": section_title,
                "policy_type": _policy_type(path.name),
            },
        )
        chunks.append(chunk)
        idx += 1
    return chunks


def _policy_type(filename: str) -> str:
    name = filename.lower()
    if "expense" in name:
        return "expense"
    if "travel" in name:
        return "travel"
    if "finance" in name:
        return "finance"
    if "handbook" in name:
        return "handbook"
    return "general"


def load_pdf(path: Path, use_ocr: bool = False) -> list[DocumentChunk]:
    """Load PDF via pypdf; optional OCR for scanned pages."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf to load PDF documents.") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if use_ocr and len(text.strip()) < 30:
            text = _ocr_page(path, page.page_number)
        pages.append(text)

    full_text = "\n\n".join(pages)
    if not full_text.strip():
        return []

    doc_id = _extract_document_id(full_text) if "**Document ID:**" in full_text else path.stem.upper()
    title = _extract_title(full_text, path.stem.replace("_", " ").title())
    chunk = DocumentChunk(
        chunk_id=f"{path.stem}_0",
        text=full_text,
        source_file=path.name,
        document_title=title,
        document_id=doc_id,
        section="Full Document",
        chunk_index=0,
        metadata={
            "source_file": path.name,
            "document_title": title,
            "document_id": doc_id,
            "section": "Full Document",
            "policy_type": _policy_type(path.name),
        },
    )
    return [chunk]


def _ocr_page(pdf_path: Path, page_number: int) -> str:
    """OCR fallback for scanned PDF pages (optional dependency)."""
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path), first_page=page_number + 1, last_page=page_number + 1)
        if images:
            return pytesseract.image_to_string(images[0])
    except Exception:
        pass
    return ""


def load_documents(documents_dir: Path, use_ocr: bool = False) -> list[DocumentChunk]:
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    all_chunks: list[DocumentChunk] = []
    for path in sorted(documents_dir.iterdir()):
        if path.suffix.lower() in {".md", ".txt"}:
            all_chunks.extend(load_markdown(path))
        elif path.suffix.lower() == ".pdf":
            all_chunks.extend(load_pdf(path, use_ocr=use_ocr))
    return all_chunks
