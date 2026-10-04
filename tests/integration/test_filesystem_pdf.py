import os
from unittest.mock import patch, MagicMock
import pytest
from core.interfaces import ToolResult
from tools.registry import ToolRegistry
from tools.fs_tools import read_pdf

def test_filesystem_agent_read_pdf_integration(tmp_path):
    # Setup Tool Registry
    mock_guardrails = MagicMock()
    mock_guardrails.check_tool_call.return_value = MagicMock(allowed=True)
    registry = ToolRegistry(guardrails=mock_guardrails)
    registry.register(
        "read_pdf",
        read_pdf,
        {
            "name": "read_pdf",
            "description": "Reads and extracts text from a PDF file.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        }
    )

    pdf_file = tmp_path / "integration_test.pdf"
    pdf_file.write_bytes(b"dummy pdf bytes")

    with patch("pypdf.PdfReader") as mock_pdf_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Integration test PDF content!"
        mock_instance = MagicMock()
        mock_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_instance

        # Execute read_pdf tool through the registry as an agent would
        with patch.dict(os.environ, {"FS_READ_ALLOWLIST": str(tmp_path)}):
            result = registry.dispatch("read_pdf", {"file_path": str(pdf_file)}, caller_agent="filesystem")

    assert result.success is True
    assert "Integration test PDF content!" in result.output
