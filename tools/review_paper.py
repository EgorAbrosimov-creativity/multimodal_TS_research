import asyncio
import argparse
from datetime import date
from pathlib import Path

import anthropic
import docx
import pypdf

ROOT = Path(__file__).parent.parent
API_KEY_PATH = ROOT / ".claude" / "api_key.txt"
SPRINGER_PDF_PATH = ROOT / "docs" / "Instructions+for+proceedings+authors+(pdf).pdf"
DEFAULT_PAPER = ROOT / "paper_draft.docx"
DEFAULT_OUT = ROOT / "docs" / "review"
MODEL = "claude-opus-4-7"
TODAY = str(date.today())


def load_api_key() -> str:
    return API_KEY_PATH.read_text().strip()


def extract_docx_text(path: Path) -> str:
    doc = docx.Document(str(path))
    lines = []
    for para in doc.paragraphs:
        if not para.text.strip():
            continue
        if para.style.name.startswith("Heading"):
            raw_level = para.style.name.split()[-1]
            level = int(raw_level) if raw_level.isdigit() else 1
            lines.append(f"\n{'#' * level} {para.text}\n")
        else:
            lines.append(para.text)
    return "\n".join(lines)


def extract_pdf_text(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
