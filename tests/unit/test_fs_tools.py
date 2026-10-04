import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from tools.fs_tools import read_pdf

def test_read_pdf_success(tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"dummy pdf content")
    
    # We must patch pypdf.PdfReader
    with patch("pypdf.PdfReader") as mock_pdf_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello World PDF Content!"
        mock_instance = MagicMock()
        mock_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_instance
        
        # Override environment variables for path resolution
        with patch.dict(os.environ, {"FS_READ_ALLOWLIST": str(tmp_path)}):
            res = read_pdf(str(pdf_file))
            
    assert res.success is True
    assert "Hello World PDF Content!" in res.output

def test_read_pdf_not_exist(tmp_path):
    pdf_file = tmp_path / "non_existent.pdf"
    with patch.dict(os.environ, {"FS_READ_ALLOWLIST": str(tmp_path)}):
        res = read_pdf(str(pdf_file))
        
    assert res.success is False
    assert "does not exist" in res.error

def test_read_pdf_not_pdf_extension(tmp_path):
    text_file = tmp_path / "test.txt"
    text_file.write_text("Hello")
    with patch.dict(os.environ, {"FS_READ_ALLOWLIST": str(tmp_path)}):
        res = read_pdf(str(text_file))
        
    assert res.success is False
    assert "is not a .pdf file" in res.error
