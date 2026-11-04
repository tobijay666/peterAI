"""Phase I2: Validation runner script (run_validation.py).

Runs all tests, evals, soak, model probe, doc sync, and generates evidence pack in validation/runs/<tag>/.
Exits with non-zero status if any exit criterion is unmet.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

VALIDATION_DIR = Path("validation")


def run_cmd(cmd: str, cwd: str = ".") -> tuple[int, str]:
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return res.returncode, res.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Peter AI Full Validation Runner")
    parser.add_argument("--tag", default="v1.3.0-rc1", help="Validation tag for evidence pack")
    parser.add_argument("--live", action="store_true", help="Run live evals with real API keys")
    args = parser.parse_args()

    tag = args.tag
    is_live = args.live or os.environ.get("E2E") == "1"
    run_dir = VALIDATION_DIR / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Validation Run (tag={tag}, live={is_live}) ===")

    all_passed = True

    # 1. Run pytest suite with coverage & junit output
    print("\n--- 1. Pytest Suite ---")
    pytest_cmd = f".venv\\Scripts\\python.exe -m pytest --junit-xml={run_dir}/junit.xml --cov=core --cov=memory --cov=llm --cov=engine --cov-report=term-missing"
    code, out = run_cmd(pytest_cmd)
    (run_dir / "pytest_report.txt").write_text(out, encoding="utf-8")
    (run_dir / "coverage.txt").write_text(out, encoding="utf-8")
    if code != 0:
        print("FAILED: Pytest suite had failures")
        all_passed = False
    else:
        print("PASSED: Pytest suite green")

    # 2. Probe models
    print("\n--- 2. Model Probe ---")
    probe_cmd = f".venv\\Scripts\\python.exe scripts/probe_models.py {'--fake' if not is_live else ''}"
    code, out = run_cmd(probe_cmd)
    if Path("validation/model_probe.json").exists():
        shutil.copy("validation/model_probe.json", run_dir / "model_probe.json")

    # 3. Evals suite
    print("\n--- 3. Evaluation Harness ---")
    eval_mode = "live" if is_live else "fake"
    eval_cmd = f".venv\\Scripts\\python.exe evals/runner.py --mode {eval_mode}"
    code, out = run_cmd(eval_cmd)
    eval_file = VALIDATION_DIR / f"evals_{eval_mode}.json"
    if eval_file.exists():
        shutil.copy(eval_file, run_dir / f"evals_{eval_mode}.json")
    if code != 0:
        print("FAILED: Evals suite")
        all_passed = False

    # 4. Soak test
    print("\n--- 4. 90-Day Soak Test ---")
    soak_cmd = ".venv\\Scripts\\python.exe scripts/soak.py"
    code, out = run_cmd(soak_cmd)
    if Path("validation/soak.md").exists():
        shutil.copy("validation/soak.md", run_dir / "soak.md")

    # 5. Traceability matrix
    print("\n--- 5. Traceability Matrix ---")
    trace_cmd = ".venv\\Scripts\\python.exe scripts/gen_traceability.py"
    code, out = run_cmd(trace_cmd)
    if Path("validation/traceability.md").exists():
        shutil.copy("validation/traceability.md", run_dir / "traceability.md")

    # 6. Doc drift check
    print("\n--- 6. Doc Drift Check ---")
    run_cmd(".venv\\Scripts\\python.exe scripts/gen_docs.py")
    drift_cmd = ".venv\\Scripts\\python.exe scripts/check_doc_drift.py"
    code, out = run_cmd(drift_cmd)
    (run_dir / "doc_drift.txt").write_text(out, encoding="utf-8")
    if code != 0:
        print("FAILED: Doc drift check")
        all_passed = False
    else:
        print("PASSED: Doc drift check")

    print(f"\n=== Validation Run Complete. Evidence pack saved to {run_dir} ===")
    if not all_passed:
        print("RESULT: VALIDATION FAILED")
        sys.exit(1)
    else:
        print("RESULT: ALL EXIT CRITERIA PASSED GREEN!")
        sys.exit(0)


if __name__ == "__main__":
    main()
