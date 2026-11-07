"""Sandboxed code and shell execution tools for Peter AI.

These tools allow Peter and his specialists to run Python snippets and shell
commands within the workspace boundary. All executions are:
  - Restricted to D:\\peterAI\\workspace\\ by default
  - Time-limited (default 30s)
  - stdout/stderr captured and returned as ToolResult
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from core.interfaces import ToolResult
from core.logger import logger

# Hard workspace root — all execution CWD is locked here unless explicitly overridden
_WORKSPACE_ROOT = Path(os.getenv("PETER_WORKSPACE", "D:/peterAI/workspace")).resolve()
_EXEC_TIMEOUT = int(os.getenv("CODE_EXEC_TIMEOUT_SEC", "30"))

_ALLOWED_CWD_ROOTS = [
    Path("D:/peterAI/workspace").resolve(),
    Path("D:/peterAI").resolve(),
]


def _is_safe_cwd(path: Path) -> bool:
    """Returns True if the given path is within an allowed workspace root."""
    try:
        resolved = path.resolve()
        return any(
            str(resolved).startswith(str(root))
            for root in _ALLOWED_CWD_ROOTS
        )
    except Exception:
        return False


def run_python_snippet(code: str, cwd: str = "") -> ToolResult:
    """Executes a Python code snippet in a subprocess and returns stdout/stderr.

    Args:
        code: Python source code string to execute.
        cwd: Working directory for execution. Defaults to the workspace root.
             Must be within the allowed workspace boundary.

    Returns:
        ToolResult with stdout as output, or error if execution failed or timed out.
    """
    exec_cwd = Path(cwd).resolve() if cwd else _WORKSPACE_ROOT
    _WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    if not _is_safe_cwd(exec_cwd):
        return ToolResult(
            success=False,
            output="",
            error=f"Execution blocked: cwd '{exec_cwd}' is outside the workspace boundary.",
        )

    if not exec_cwd.exists():
        exec_cwd.mkdir(parents=True, exist_ok=True)

    # Write code to a temp file in workspace
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="peter_exec_",
            dir=exec_cwd,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        logger.info(f"[CodeTools] Running Python snippet from {tmp_path} (cwd={exec_cwd})")
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=_EXEC_TIMEOUT,
            cwd=str(exec_cwd),
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout
        if stderr:
            combined += f"\n[stderr]\n{stderr}"

        if result.returncode != 0:
            return ToolResult(
                success=False,
                output=combined,
                error=f"Python execution exited with code {result.returncode}.\n{stderr[:500]}",
            )

        return ToolResult(success=True, output=combined or "(No output)")

    except subprocess.TimeoutExpired:
        logger.warning(f"[CodeTools] Python snippet timed out after {_EXEC_TIMEOUT}s.")
        return ToolResult(
            success=False,
            output="",
            error=f"Execution timed out after {_EXEC_TIMEOUT} seconds.",
        )
    except Exception as e:
        logger.error(f"[CodeTools] run_python_snippet error: {e}")
        return ToolResult(success=False, output="", error=str(e))
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def run_shell_command(command: str, cwd: str = "") -> ToolResult:
    """Runs a shell command within the workspace boundary.

    Args:
        command: Shell command string to execute (run via cmd on Windows).
        cwd: Working directory. Defaults to the workspace root. Must be within
             the allowed workspace boundary.

    Returns:
        ToolResult with stdout/stderr, or error if execution failed or timed out.

    Security notes:
        - Blocks commands that reference common dangerous patterns.
        - CWD is restricted to the workspace boundary.
        - Execution timeout enforced.
    """
    # Minimal dangerous-command guard (not a full sandbox — relies on guardrails too)
    _BLOCKED_PATTERNS = [
        "rm -rf /", "rmdir /s /q C:\\", "format c:", "del /f /q C:\\",
        "shutdown", "taskkill", "net user", "reg delete",
        "curl -o /tmp", "wget -O /tmp",
    ]
    cmd_lower = command.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern.lower() in cmd_lower:
            return ToolResult(
                success=False,
                output="",
                error=f"Command blocked by safety filter: matches pattern '{pattern}'.",
            )

    exec_cwd = Path(cwd).resolve() if cwd else _WORKSPACE_ROOT
    _WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    if not _is_safe_cwd(exec_cwd):
        return ToolResult(
            success=False,
            output="",
            error=f"Execution blocked: cwd '{exec_cwd}' is outside the workspace boundary.",
        )

    if not exec_cwd.exists():
        exec_cwd.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"[CodeTools] Running shell command: '{command[:80]}' (cwd={exec_cwd})")
        # Use powershell on Windows for richer command support
        if sys.platform == "win32":
            shell_cmd = ["powershell", "-NonInteractive", "-Command", command]
        else:
            shell_cmd = ["bash", "-c", command]

        result = subprocess.run(
            shell_cmd,
            capture_output=True,
            text=True,
            timeout=_EXEC_TIMEOUT,
            cwd=str(exec_cwd),
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout
        if stderr:
            combined += f"\n[stderr]\n{stderr}"

        if result.returncode != 0:
            return ToolResult(
                success=False,
                output=combined,
                error=f"Command exited with code {result.returncode}.\n{stderr[:500]}",
            )

        return ToolResult(success=True, output=combined or "(No output)")

    except subprocess.TimeoutExpired:
        logger.warning(f"[CodeTools] Shell command timed out after {_EXEC_TIMEOUT}s.")
        return ToolResult(
            success=False,
            output="",
            error=f"Shell command timed out after {_EXEC_TIMEOUT} seconds.",
        )
    except Exception as e:
        logger.error(f"[CodeTools] run_shell_command error: {e}")
        return ToolResult(success=False, output="", error=str(e))
