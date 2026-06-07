import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import docx as docx_lib
import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_load_api_key_reads_and_strips(tmp_path):
    key_file = tmp_path / "api_key.txt"
    key_file.write_text("  sk-ant-test123\n  ")

    from tools.review_paper import load_api_key
    with patch("tools.review_paper.API_KEY_PATH", key_file):
        key = load_api_key()

    assert key == "sk-ant-test123"


def test_extract_docx_text_includes_headings_and_body(tmp_path):
    doc = docx_lib.Document()
    doc.add_heading("Introduction", level=1)
    doc.add_paragraph("This is the intro text.")
    doc.add_heading("Methods", level=2)
    doc.add_paragraph("We used PatchTST.")
    doc_path = tmp_path / "test.docx"
    doc.save(str(doc_path))

    from tools.review_paper import extract_docx_text
    text = extract_docx_text(doc_path)

    assert "Introduction" in text
    assert "This is the intro text." in text
    assert "Methods" in text
    assert "We used PatchTST." in text


def test_extract_docx_text_nonexistent_raises(tmp_path):
    from tools.review_paper import extract_docx_text
    with pytest.raises(Exception):
        extract_docx_text(tmp_path / "missing.docx")


def test_extract_pdf_text_nonexistent_raises(tmp_path):
    from tools.review_paper import extract_pdf_text
    with pytest.raises(Exception):
        extract_pdf_text(tmp_path / "missing.pdf")
