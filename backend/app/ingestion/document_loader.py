import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    content: str
    metadata: Dict[str, Any]
    source: str
    page_num: Optional[int] = None
    doc_type: str = "unknown"


class DocumentLoader:
    """Multi-strategy document loader using Docling and PyMuPDF."""

    def load(self, file_path: str) -> List[RawDocument]:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext in (".docx", ".doc"):
            return self._load_docx(path)
        elif ext == ".txt":
            return self._load_txt(path)
        elif ext == ".md":
            return self._load_txt(path)
        else:
            return self._load_txt(path)

    def _load_pdf(self, path: Path) -> List[RawDocument]:
        docs = []
        # Try Docling first (best quality)
        try:
            docs = self._load_pdf_docling(path)
            if docs:
                return docs
        except Exception as e:
            logger.warning(f"Docling failed for {path.name}, falling back to PyMuPDF: {e}")

        # Fallback to PyMuPDF
        try:
            docs = self._load_pdf_pymupdf(path)
            if docs:
                return docs
        except Exception as e:
            logger.error(f"PyMuPDF also failed for {path.name}: {e}")

        return docs

    def _load_pdf_docling(self, path: Path) -> List[RawDocument]:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat

        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document

        docs = []
        # Extract by page with rich metadata
        for page_num, page in enumerate(result.pages, 1):
            # Get page text via markdown export filtered by page
            page_content = ""
            for element, _level in doc.iterate_items():
                if hasattr(element, "prov") and element.prov:
                    for prov in element.prov:
                        if hasattr(prov, "page_no") and prov.page_no == page_num:
                            if hasattr(element, "text"):
                                page_content += element.text + "\n"

            if not page_content.strip():
                continue

            metadata = {
                "source": path.name,
                "file_path": str(path),
                "page": page_num,
                "total_pages": len(result.pages),
                "loader": "docling",
                "file_type": "pdf",
            }
            docs.append(RawDocument(
                content=page_content.strip(),
                metadata=metadata,
                source=path.name,
                page_num=page_num,
                doc_type="pdf",
            ))

        # Fallback: full document if page extraction failed
        if not docs:
            full_text = doc.export_to_markdown()
            if full_text.strip():
                docs.append(RawDocument(
                    content=full_text,
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                        "loader": "docling",
                        "file_type": "pdf",
                    },
                    source=path.name,
                    doc_type="pdf",
                ))
        return docs

    def _load_pdf_pymupdf(self, path: Path) -> List[RawDocument]:
        import fitz  # PyMuPDF

        pdf = fitz.open(str(path))
        docs = []
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")
            if not text.strip():
                continue

            # Extract tables if present
            tables = page.find_tables()
            table_text = ""
            if tables.tables:
                for table in tables.tables:
                    table_text += "\n[TABLE]\n"
                    for row in table.extract():
                        row_text = " | ".join([str(c) if c else "" for c in row])
                        table_text += row_text + "\n"
                    table_text += "[/TABLE]\n"

            combined = text + table_text

            metadata = {
                "source": path.name,
                "file_path": str(path),
                "page": page_num + 1,
                "total_pages": len(pdf),
                "width": page.rect.width,
                "height": page.rect.height,
                "loader": "pymupdf",
                "file_type": "pdf",
            }
            docs.append(RawDocument(
                content=combined.strip(),
                metadata=metadata,
                source=path.name,
                page_num=page_num + 1,
                doc_type="pdf",
            ))
        pdf.close()
        return docs

    def _load_docx(self, path: Path) -> List[RawDocument]:
        try:
            import docx
            document = docx.Document(str(path))
            sections = []
            current_section = []
            current_heading = "Document"

            for para in document.paragraphs:
                if para.style.name.startswith("Heading"):
                    if current_section:
                        sections.append((current_heading, "\n".join(current_section)))
                        current_section = []
                    current_heading = para.text or current_heading
                else:
                    if para.text.strip():
                        current_section.append(para.text)

            if current_section:
                sections.append((current_heading, "\n".join(current_section)))

            docs = []
            for i, (heading, content) in enumerate(sections):
                if not content.strip():
                    continue
                docs.append(RawDocument(
                    content=f"{heading}\n\n{content}",
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                        "section": heading,
                        "section_index": i,
                        "loader": "python-docx",
                        "file_type": "docx",
                    },
                    source=path.name,
                    doc_type="docx",
                ))
            return docs
        except Exception as e:
            logger.error(f"DOCX loading failed: {e}")
            return []

    def _load_txt(self, path: Path) -> List[RawDocument]:
        try:
            with open(str(path), "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return [RawDocument(
                content=content,
                metadata={
                    "source": path.name,
                    "file_path": str(path),
                    "loader": "text",
                    "file_type": path.suffix.lstrip("."),
                },
                source=path.name,
                doc_type="text",
            )]
        except Exception as e:
            logger.error(f"Text loading failed: {e}")
            return []
