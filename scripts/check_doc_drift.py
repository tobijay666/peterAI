"""Phase I4: Doc drift checker script (check_doc_drift.py).

Verifies that docs/generated/ reference docs are up-to-date with code and config.
Checks env vars and registered tools against catalog and .env.example.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gen_docs import (
    gen_tool_catalog,
    gen_agent_matrix,
    gen_model_ladders,
    gen_env_variables,
    gen_db_schema,
    gen_guardrail_rules,
    gen_event_topics,
)

DOCS_DIR = Path("docs/generated")


def check_doc_drift() -> bool:
    print("Checking doc drift...")
    errors = []

    if not DOCS_DIR.exists():
        errors.append("docs/generated/ directory does not exist")
        print("\n".join(errors))
        return False

    checks = [
        ("tool_catalog.md", gen_tool_catalog),
        ("agent_matrix.md", gen_agent_matrix),
        ("model_ladders.md", gen_model_ladders),
        ("env_variables.md", gen_env_variables),
        ("db_schema.md", gen_db_schema),
        ("guardrail_rules.md", gen_guardrail_rules),
        ("event_topics.md", gen_event_topics),
    ]

    for fname, gen_func in checks:
        doc_file = DOCS_DIR / fname
        if not doc_file.exists():
            errors.append(f"Missing doc file: {fname}")
            continue

        actual = doc_file.read_text(encoding="utf-8")
        expected = gen_func()

        if actual.strip() != expected.strip():
            errors.append(f"Doc drift detected in {fname}")

    if errors:
        print("Doc drift check failed:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("No doc drift detected. Docs are in sync with code and config.")
    return True


if __name__ == "__main__":
    success = check_doc_drift()
    sys.exit(0 if success else 1)
