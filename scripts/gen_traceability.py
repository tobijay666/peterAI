"""Phase I4: Requirement traceability matrix generator (gen_traceability.py).

Scans tests/ for @pytest.mark.req markers and outputs validation/traceability.md.
"""
import ast
import re
from pathlib import Path
from typing import Dict, List, Set

OUTPUT_PATH = Path("validation/traceability.md")
TESTS_DIR = Path("tests")

# All functional requirements FR-1 through FR-20
ALL_REQUIREMENTS = [
    f"FR-{i}.{j}" for i in range(1, 21) for j in range(1, 8)
]


def find_req_markers() -> Dict[str, List[str]]:
    req_map: Dict[str, List[str]] = {}

    for py_file in TESTS_DIR.glob("**/*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            # Regex match pytestmark = pytest.mark.req(...) or @pytest.mark.req(...)
            matches = re.findall(r'req\(["\'](FR-[\d\.]+|E1)["\']\)', content)
            for m in matches:
                if m not in req_map:
                    req_map[m] = []
                req_map[m].append(str(py_file).replace("\\", "/"))
        except Exception:
            pass

    return req_map


def generate_traceability_matrix() -> str:
    req_map = find_req_markers()

    content = "# Requirement Traceability Matrix\n\n"
    content += "Generated automatically by `scripts/gen_traceability.py`.\n\n"
    content += "| Requirement ID | Test File(s) | Status |\n|---|---|---|\n"

    found_count = 0
    for req_id, test_files in sorted(req_map.items()):
        files_str = ", ".join(f"`{f}`" for f in set(test_files))
        content += f"| `{req_id}` | {files_str} | **COVERED** |\n"
        found_count += 1

    content += f"\nTotal Covered Requirements / Scenarios: {found_count}\n"
    return content


def main() -> None:
    matrix = generate_traceability_matrix()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(matrix, encoding="utf-8")
    print(f"Traceability matrix generated at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
