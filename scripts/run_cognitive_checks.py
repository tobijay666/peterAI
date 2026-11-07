"""Run every automated check for the cognitive-memory work (Stages 0–8) and save the evidence.

    python scripts/run_cognitive_checks.py            # everything except the UI smoke test
    python scripts/run_cognitive_checks.py --ui       # + headless browser smoke test
    python scripts/run_cognitive_checks.py --quick    # skip the 90-day soak and the web build

Steps (each result is saved under validation/runs/cognitive/<timestamp>/):
  1. cognitive suite      tests/cognitive  (Stages 0–7, ~110 tests)
  2. end-to-end + props   tests/integration/test_cognitive_memory.py, tests/property/test_cognitive_properties.py
  3. full regression      the whole pytest suite (old tests must still pass), slow tests included
  4. soak                 scripts/soak_cognitive.py --days 90
  5. curiosity eval       evals/curiosity_eval.py (offline wiring check)
  6. web                  npm run typecheck && npm run build   (skipped if npm is not installed)
  7. UI smoke (--ui)      scripts/ui_smoke.py                   (needs playwright + chromium)

Exit code 0 only if every step that ran passed. A summary is printed and saved as summary.md.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(name: str, cmd, out_dir: Path, cwd: Path = ROOT, shell: bool = False, timeout: int = 3600):
    print(f"\n=== {name} ===\n$ {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), shell=shell, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8", errors="replace")
        code, output = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        code, output = 124, f"TIMEOUT after {timeout}s\n{e.stdout or ''}{e.stderr or ''}"
    dt = time.time() - t0
    log = out_dir / (name.lower().replace(" ", "_").replace("/", "_") + ".txt")
    log.write_text(output, encoding="utf-8")
    tail = "\n".join(l for l in output.strip().splitlines()[-6:] if l.strip())
    print(tail)
    status = "PASS" if code == 0 else ("SKIP" if code == 2 and "SKIP" in output else "FAIL")
    print(f"--> {status} ({dt:.0f}s, log: {log.relative_to(ROOT)})", flush=True)
    return {"name": name, "status": status, "seconds": round(dt), "log": str(log.relative_to(ROOT)),
            "tail": tail.splitlines()[-1] if tail else ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", action="store_true", help="also run the headless browser smoke test")
    ap.add_argument("--quick", action="store_true", help="skip the 90-day soak and the web build")
    args = ap.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "validation" / "runs" / "cognitive" / stamp
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    pytest = [py, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-q",
              "-W", "ignore::DeprecationWarning"]
    results = []

    results.append(run("1 cognitive suite", pytest + ["tests/cognitive", f"--junit-xml={out / 'junit_cognitive.xml'}"], out))
    results.append(run("2 end-to-end and properties", pytest + [
        "tests/integration/test_cognitive_memory.py", "tests/property/test_cognitive_properties.py",
        f"--junit-xml={out / 'junit_e2e.xml'}"], out))
    results.append(run("3 full regression", pytest + ["tests", f"--junit-xml={out / 'junit_full.xml'}"], out))
    if not args.quick:
        results.append(run("4 soak 90 days", [py, "scripts/soak_cognitive.py", "--days", "90",
                                              "--out", str(out / "soak_90d.json")], out))
    results.append(run("5 curiosity eval", [py, "evals/curiosity_eval.py"], out))

    npm = shutil.which("npm")
    if args.quick:
        pass
    elif npm is None:
        results.append({"name": "6 web build", "status": "SKIP", "seconds": 0, "log": "", "tail": "npm not found"})
        print("\n=== 6 web build ===\nSKIP: npm not on PATH")
    else:
        web = ROOT / "web"
        if not (web / "node_modules").exists():
            results.append(run("6a npm install", f'"{npm}" install --no-audit --no-fund', out, cwd=web, shell=True))
        results.append(run("6b web typecheck", f'"{npm}" run typecheck', out, cwd=web, shell=True))
        results.append(run("6c web build", f'"{npm}" run build', out, cwd=web, shell=True))

    if args.ui:
        results.append(run("7 ui smoke", [py, "scripts/ui_smoke.py"], out, timeout=900))

    failed = [r for r in results if r["status"] == "FAIL"]
    lines = [f"# Cognitive memory checks — {stamp}", "", "| Step | Result | Time | Last line |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['status']} | {r['seconds']}s | {r['tail'].replace('|', '/')[:90]} |")
    lines += ["", f"**Overall: {'PASS' if not failed else 'FAIL'}** — logs in `{out.relative_to(ROOT)}`"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0 if not failed else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
