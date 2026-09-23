"""Unit tests for filesystem tool path boundary resolution."""

import pytest
from tools.fs_tools import _resolve_read_path, _resolve_write_path


def test_write_outside_workspace_blocked(temp_dir):
    with pytest.raises(PermissionError):
        # Attempt writing outside permitted workspace
        _resolve_write_path("C:/Windows/System32/malicious.dll")


def test_sensitive_file_read_blocked():
    with pytest.raises(PermissionError):
        _resolve_read_path("d:/peterAI/.env")
