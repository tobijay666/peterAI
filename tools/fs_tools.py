"""Filesystem tool functions enforcing read allowlist and strict workspace write boundaries."""

import os
from pathlib import Path
from core.interfaces import ToolResult

DEFAULT_READ_ROOT = Path(os.getenv("FS_READ_ROOT", "D:\\" if os.name == "nt" else "/"))
DEFAULT_WRITE_ROOT = Path(os.getenv("PETER_WORKSPACE", "D:\\Peter_Workspace" if os.name == "nt" else "./workspace"))

# Sensitive patterns that are blocked even if inside allowlist
BLOCKED_PATTERNS = [
    ".env",
    ".pem",
    ".key",
    "id_rsa",
    ".ssh",
    ".aws",
    ".azure",
    "user data",
    "firefox\\profiles",
]


def _audit_log(action: str, path: str, success: bool, extra: str = "") -> None:
    audit_path = Path("data/fs_audit.log")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(f"[{os.getenv('USER', 'user')}] [{action}] [{path}] [success={success}] {extra}\n")


def _is_path_blocked(target_path: Path) -> bool:
    path_str = str(target_path).lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in path_str:
            return True
    return False


def _resolve_read_path(path_str: str) -> Path:
    target = Path(path_str).resolve()
    if _is_path_blocked(target):
        raise PermissionError(f"Access to sensitive file or directory '{path_str}' is strictly blocked.")

    # Check allowlist
    allowlist_str = os.getenv("FS_READ_ALLOWLIST", "")
    allowlist = [Path(p.strip()).resolve() for p in allowlist_str.split(";") if p.strip()]
    if DEFAULT_READ_ROOT.exists():
        allowlist.append(DEFAULT_READ_ROOT.resolve())

    # Fallback to current working directory if default read root doesn't exist
    cwd = Path.cwd().resolve()
    allowlist.append(cwd)

    allowed = False
    for allowed_root in allowlist:
        try:
            target.relative_to(allowed_root)
            allowed = True
            break
        except ValueError:
            continue

    if not allowed:
        raise PermissionError(f"Read path '{path_str}' is outside allowed read roots: {allowlist}")

    return target


def _resolve_write_path(path_str: str) -> Path:
    target = Path(path_str).resolve()
    if _is_path_blocked(target):
        raise PermissionError(f"Write access to sensitive file '{path_str}' is strictly blocked.")

    write_root = DEFAULT_WRITE_ROOT.resolve()
    try:
        target.relative_to(write_root)
    except ValueError:
        # Fallback to local workspace directory if default workspace path fails
        local_ws = (Path.cwd() / "workspace").resolve()
        try:
            target.relative_to(local_ws)
        except ValueError:
            raise PermissionError(f"Write path '{path_str}' is outside permitted workspace '{write_root}'.")

    return target


def list_directory(directory_path: str = ".") -> ToolResult:
    try:
        target = _resolve_read_path(directory_path)
        if not target.exists() or not target.is_dir():
            return ToolResult(success=False, output="", error=f"Directory '{directory_path}' does not exist.")

        items = []
        for child in target.iterdir():
            if _is_path_blocked(child):
                continue
            kind = "DIR" if child.is_dir() else "FILE"
            size = child.stat().st_size if child.is_file() else 0
            items.append(f"{kind:<4} {child.name} ({size} bytes)")

        _audit_log("list_directory", str(target), True)
        return ToolResult(success=True, output="\n".join(items) or "Directory is empty.")
    except Exception as e:
        _audit_log("list_directory", directory_path, False, str(e))
        return ToolResult(success=False, output="", error=str(e))


def read_file(file_path: str, max_chars: int = 100000) -> ToolResult:
    try:
        target = _resolve_read_path(file_path)
        if not target.exists() or not target.is_file():
            return ToolResult(success=False, output="", error=f"File '{file_path}' does not exist.")

        content = target.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [Truncated at {max_chars} chars]"

        _audit_log("read_file", str(target), True)
        return ToolResult(success=True, output=content)
    except Exception as e:
        _audit_log("read_file", file_path, False, str(e))
        return ToolResult(success=False, output="", error=str(e))


def read_pdf(file_path: str, max_chars: int = 100000) -> ToolResult:
    try:
        import pypdf
        target = _resolve_read_path(file_path)
        if not target.exists() or not target.is_file():
            return ToolResult(success=False, output="", error=f"File '{file_path}' does not exist.")
        
        if target.suffix.lower() != '.pdf':
            return ToolResult(success=False, output="", error=f"File '{file_path}' is not a .pdf file.")

        text_content = []
        with open(target, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content.append(extracted)
                
        content = "\n\n".join(text_content)
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [Truncated at {max_chars} chars]"

        _audit_log("read_pdf", str(target), True)
        return ToolResult(success=True, output=content or "No text could be extracted from PDF.")
    except ImportError:
        _audit_log("read_pdf", file_path, False, "pypdf not installed")
        return ToolResult(success=False, output="", error="pypdf is required to read PDFs. Install it via pip install pypdf")
    except Exception as e:
        _audit_log("read_pdf", file_path, False, str(e))
        return ToolResult(success=False, output="", error=str(e))


def write_file(file_path: str, content: str) -> ToolResult:
    try:
        target = _resolve_write_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        _audit_log("write_file", str(target), True)
        return ToolResult(success=True, output=f"Successfully wrote {len(content)} characters to {target}")
    except Exception as e:
        _audit_log("write_file", file_path, False, str(e))
        return ToolResult(success=False, output="", error=str(e))


def create_directory(directory_path: str) -> ToolResult:
    try:
        target = _resolve_write_path(directory_path)
        target.mkdir(parents=True, exist_ok=True)
        _audit_log("create_directory", str(target), True)
        return ToolResult(success=True, output=f"Directory '{target}' created.")
    except Exception as e:
        _audit_log("create_directory", directory_path, False, str(e))
        return ToolResult(success=False, output="", error=str(e))


def search_files(directory_path: str = ".", pattern: str = "*") -> ToolResult:
    try:
        target = _resolve_read_path(directory_path)
        matches = []
        for path in target.rglob(pattern):
            if _is_path_blocked(path):
                continue
            matches.append(str(path))
            if len(matches) >= 100:
                break
        _audit_log("search_files", str(target), True)
        return ToolResult(success=True, output="\n".join(matches) or "No matching files found.")
    except Exception as e:
        _audit_log("search_files", directory_path, False, str(e))
        return ToolResult(success=False, output="", error=str(e))
