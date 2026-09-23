"""Git operations tools for DevOps agent with mandatory QA gate validation."""

import datetime
import subprocess
from typing import Optional
from core.interfaces import ToolResult
from core.logger import logger


def _run_git_cmd(args: list[str]) -> ToolResult:
    try:
        res = subprocess.run(["git"] + args, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return ToolResult(success=True, output=res.stdout.strip())
        else:
            return ToolResult(success=False, output=res.stdout.strip(), error=res.stderr.strip())
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Git command failed: {e}")


def git_status() -> ToolResult:
    return _run_git_cmd(["status", "--short"])


def git_branch(branch_name: Optional[str] = None) -> ToolResult:
    if branch_name:
        return _run_git_cmd(["checkout", "-b", branch_name])
    return _run_git_cmd(["branch", "-a"])


def create_session_branch(slug: str = "work-session") -> ToolResult:
    """Creates a work session branch auto/<timestamp>-<slug> (FR-5.3)."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"auto/{ts}-{slug}"
    return _run_git_cmd(["checkout", "-b", branch_name])


def git_log(max_count: int = 5) -> ToolResult:
    return _run_git_cmd(["log", f"-n{max_count}", "--oneline"])


def git_revert(commit_hash: str) -> ToolResult:
    """Reverts a specific commit (FR-5.5)."""
    return _run_git_cmd(["revert", "--no-edit", commit_hash])


def git_stash(action: str = "save") -> ToolResult:
    """Stashes uncommitted changes (FR-5.5)."""
    if action == "pop":
        return _run_git_cmd(["stash", "pop"])
    elif action == "list":
        return _run_git_cmd(["stash", "list"])
    return _run_git_cmd(["stash", "push", "-m", "Auto-stashed by DevOps agent"])


def run_pytest(target_path: str = "tests") -> ToolResult:
    """Runs pytest unit test suite (FR-6.1)."""
    try:
        res = subprocess.run(["pytest", target_path, "-v"], capture_output=True, text=True, check=False)
        passed = res.returncode == 0
        return ToolResult(
            success=passed,
            output=res.stdout.strip(),
            error=res.stderr.strip() if not passed else None,
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Pytest runner error: {e}")


def run_hypothesis(target_path: str = "tests/property") -> ToolResult:
    """Runs hypothesis property tests (FR-6.2)."""
    try:
        res = subprocess.run(["pytest", target_path, "-v"], capture_output=True, text=True, check=False)
        passed = res.returncode == 0
        return ToolResult(
            success=passed,
            output=res.stdout.strip(),
            error=res.stderr.strip() if not passed else None,
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Hypothesis runner error: {e}")


def run_tests() -> ToolResult:
    """Runs standard pytest test suite as a QA gate (FR-5.2)."""
    return run_pytest("tests")


def git_commit(message: str, force_bypass_qa: bool = False) -> ToolResult:
    """Commits changes after ensuring QA unit tests pass."""
    if not force_bypass_qa:
        logger.info("DevOps: Evaluating mandatory QA gate before committing...")
        qa_res = run_tests()
        if not qa_res.success:
            err_msg = (
                f"COMMIT REJECTED: QA test suite failed!\nError details:\n{qa_res.error or qa_res.output}\n"
                "Fix failing tests before committing."
            )
            logger.error(err_msg)
            return ToolResult(success=False, output="", error=err_msg)

    # Stage changes
    _run_git_cmd(["add", "."])
    # Execute commit
    commit_res = _run_git_cmd(["commit", "-m", message])
    return commit_res
