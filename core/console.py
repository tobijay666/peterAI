"""Console formatting utilities for REPL visual outputs."""

import os
import sys

NO_COLOR = bool(os.getenv("NO_COLOR", "")) or not sys.stdout.isatty()


class Colors:
    RESET = "" if NO_COLOR else "\033[0m"
    BOLD = "" if NO_COLOR else "\033[1m"
    ITALIC = "" if NO_COLOR else "\033[3m"
    GRAY = "" if NO_COLOR else "\033[90m"
    CYAN = "" if NO_COLOR else "\033[36m"
    YELLOW = "" if NO_COLOR else "\033[33m"
    GREEN = "" if NO_COLOR else "\033[32m"
    RED = "" if NO_COLOR else "\033[31m"
    MAGENTA = "" if NO_COLOR else "\033[35m"


def format_prompt(text: str) -> str:
    return f"{Colors.YELLOW}{Colors.BOLD}{text}{Colors.RESET}"


def format_reasoning(text: str) -> str:
    return f"{Colors.GRAY}{Colors.ITALIC}{text}{Colors.RESET}"


def format_agent(agent_name: str, text: str) -> str:
    color_map = {
        "Peter": Colors.CYAN,
        "Researcher": Colors.MAGENTA,
        "FileSystem": Colors.YELLOW,
        "Librarian": Colors.GREEN,
        "DevOps": Colors.CYAN,
        "QA": Colors.RED,
        "Voice": Colors.GRAY,
    }
    color = color_map.get(agent_name, Colors.CYAN)
    return f"{color}{Colors.BOLD}[{agent_name}]{Colors.RESET} {text}"


def format_error(text: str) -> str:
    return f"{Colors.RED}{Colors.BOLD}Error:{Colors.RESET} {text}"


def format_warn(text: str) -> str:
    return f"{Colors.YELLOW}{Colors.BOLD}Warning:{Colors.RESET} {text}"
