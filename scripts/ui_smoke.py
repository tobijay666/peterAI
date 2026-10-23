"""Headless UI smoke test for the 3-D mind view (Stage 8, S8.5).

Starts the real server on a free port with a seeded throw-away user, opens the
built web app in headless Chromium, and checks:

  1. the knowledge web renders (nodes from /api/mind/snapshot, no page errors)
  2. high-usage topics are drawn bigger than low-usage ones
  3. clicking a node opens the inspector with usage, records and sources
  4. double-clicking a node opens its vector cloud
  5. "Sleep now" runs a cycle live: the state chip goes sleeping -> awake and the
     stale topic is pruned from the scene

Screenshots go to validation/runs/cognitive/ui/. Exit code 0 = pass.

Needs:  pip install playwright && python -m playwright install chromium
        cd web && npm install && npm run build
Run:    python scripts/ui_smoke.py
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
USER = "ui-smoke"
OUT = ROOT / "validation" / "runs" / "cognitive" / "ui"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def seed() -> None:
    from core.user_context import web_context
    from memory.cognition import CognitiveMemory
    from memory.graph_store import GraphMemory
    from memory.vector_store import VectorMemory

    shutil.rmtree(ROOT / "data" / "users" / USER, ignore_errors=True)
    ctx = web_context(USER, "UI Smoke")
    vector = VectorMemory(str(ctx.chroma_dir), user_id=USER)
    graph = GraphMemory(str(ctx.graph_db_path), user_id=USER)
    cog = CognitiveMemory(vector, graph)
    topics = {
        "Henna": ["Henna dye comes from the leaves of Lawsonia inermis.",
                  "Bridal henna is applied the night before South Asian weddings."],
        "Saree": ["The Kandyan saree is the traditional Sri Lankan bridal dress."],
        "Invoices": ["Invoices are due within thirty days."],
        "Stale trivia": ["A fact nobody has asked about for months."],
    }
    for label, recs in topics.items():
        node, _ = cog.knowledge.ensure_node(label)
        for i, text in enumerate(recs):
            cog.knowledge.add_record(node, text, {"source_url": f"https://{label.split()[0].lower()}.example/{i}",
                                                  "source_title": f"{label} source"})
        cog.knowledge.refresh_stats(node)
    for _ in range(12):
        cog.usage.touch("ent_henna", emit_event=False)
    for _ in range(4):
        cog.usage.touch("ent_saree", emit_event=False)
    graph.set_node_fields("ent_invoices", last_accessed=1.0)
    graph.set_node_fields("ent_stale_trivia", status="fading", fading_since=1.0, last_accessed=1.0)
    graph.close()


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("SKIP: playwright not installed (pip install playwright && python -m playwright install chromium)")
        return 2
    if not (ROOT / "web" / "dist" / "index.html").exists():
        print("FAIL: web/dist missing — run: cd web && npm install && npm run build")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    seed()
    port = free_port()
    secret = "ui-smoke-" + str(os.getpid())
    # Deterministic sleep: no web research, no LLM calls during the smoke run
    smoke_cfg = OUT / "cognition_smoke.yaml"
    smoke_cfg.write_text("research:\n  enabled: false\nresolver:\n  use_llm: false\n"
                         "sleep:\n  research_items_per_cycle: 0\n", encoding="utf-8")
    env = {**os.environ, "JWT_SECRET": secret, "EVOLUTION_MODE": "active", "AUTH_MODE": "jwt",
           "PETER_COGNITION_CONFIG": str(smoke_cfg)}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(port)],
                              cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failures, page_errors = [], []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail and not ok else ''}")
        if not ok:
            failures.append(name)

    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        os.environ["JWT_SECRET"] = secret
        import importlib
        import server.auth as auth
        importlib.reload(auth)
        token = auth.create_access_token(USER, "UI Smoke")
        auth_hdr = "{'Content-Type':'application/json','Authorization':'Bearer '+sessionStorage.getItem('peter_access_token')}"

        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            page = browser.new_page(viewport={"width": 1440, "height": 860})
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.goto(base + "/#/login")
            page.evaluate(f"sessionStorage.setItem('peter_access_token', {json.dumps(token)})")
            page.goto(base + "/#/app")
            page.wait_for_function("window.__peterMind && window.__peterMind.layer.size() > 0", timeout=30000)
            page.wait_for_timeout(2500)
            page.screenshot(path=str(OUT / "1_overview.png"))

            ids = page.evaluate("window.__peterMind.layer.allNodes().map(n => n.id)")
            check("knowledge web renders all seeded topics",
                  {"ent_henna", "ent_saree", "ent_invoices", "ent_stale_trivia"} <= set(ids), str(ids))
            tiers = page.evaluate("Object.fromEntries(window.__peterMind.layer.allNodes().map(n => [n.id, n.style.scale]))")
            check("high-usage topic drawn bigger than a low-usage one", tiers.get("ent_henna", 0) > tiers.get("ent_invoices", 9))

            page.evaluate("window.__peterMind.rig.controls.autoRotate = false")
            xy = page.evaluate("window.__peterMind.screenPos('ent_saree')")
            page.mouse.click(xy[0], xy[1])
            page.wait_for_selector(".mind-inspector-panel .mi-title", timeout=10000)
            page.wait_for_timeout(600)
            title = page.inner_text(".mind-inspector-panel .mi-title")
            panel = page.inner_text(".mind-inspector-panel")
            page.screenshot(path=str(OUT / "2_inspector.png"))
            check("inspector opens for the clicked node", title == "Saree", title)
            check("inspector shows usage, records and sources",
                  "usage score" in panel and "RECORDS" in panel.upper() and "Saree source" in panel)
            page.keyboard.press("Escape")
            page.click("[data-act='home']")
            page.wait_for_timeout(1200)

            xy = page.evaluate("window.__peterMind.screenPos('ent_henna')")
            page.mouse.dblclick(xy[0], xy[1])
            page.wait_for_timeout(1800)
            toast = page.inner_text(".mind-toast")
            page.screenshot(path=str(OUT / "3_vectors.png"))
            check("double-click opens the node's vector cloud", "vectors in k_henna" in toast, toast)
            page.keyboard.press("Escape")

            r = page.evaluate(f"fetch('/api/mind/sleep', {{method:'POST', headers:{auth_hdr}, body: JSON.stringify({{mode:'active'}})}}).then(r => r.json())")
            check("sleep endpoint starts a cycle", bool(r.get("started")), str(r))
            try:
                page.wait_for_selector(".mind-state[data-state='sleeping']", timeout=20000)
                page.screenshot(path=str(OUT / "4_sleeping.png"))
                slept = True
            except Exception:
                slept = False
            check("scene enters sleep mode", slept)
            try:
                page.wait_for_selector(".mind-state[data-state='awake']", timeout=120000)
                page.wait_for_timeout(3000)
                woke = True
            except Exception:
                woke = False
            check("scene wakes after the cycle", woke)
            page.screenshot(path=str(OUT / "5_after_sleep.png"))
            ids_after = page.evaluate("window.__peterMind.layer.allNodes().map(n => n.id)")
            check("stale topic was pruned from the scene", "ent_stale_trivia" not in ids_after, str(ids_after))
            check("no JavaScript errors", not page_errors, "; ".join(page_errors[:3]))
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()
        time.sleep(1)
        shutil.rmtree(ROOT / "data" / "users" / USER, ignore_errors=True)

    print(f"\nScreenshots: {OUT}")
    print("UI SMOKE: " + ("PASS" if not failures else f"FAIL ({len(failures)}): " + ", ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
