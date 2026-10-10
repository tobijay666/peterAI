"""Phase I4: Documentation generator script (gen_docs.py).

Generates reference markdown documents under docs/generated/ directly from code and config:
- tool_catalog.md
- agent_matrix.md
- model_ladders.md
- env_variables.md
- db_schema.md
- guardrail_rules.md
- event_topics.md
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml

DOCS_DIR = Path("docs/generated")


def gen_tool_catalog() -> str:
    from tools import registry
    tools = registry.list_tools() if hasattr(registry, "list_tools") else []
    
    content = "# Tool Catalog\n\nGenerated automatically from registered tools.\n\n"
    content += "| Tool Name | Description | Namespace |\n|---|---|---|\n"
    for t in tools:
        name = t.get("name", "")
        desc = t.get("description", "").replace("\n", " ")
        ns = t.get("namespace", "general")
        content += f"| `{name}` | {desc} | `{ns}` |\n"
    return content


def gen_agent_matrix() -> str:
    agents_path = Path("config/agents.yaml")
    agents = {}
    if agents_path.exists():
        agents = yaml.safe_load(agents_path.read_text(encoding="utf-8")) or {}
        
    content = "# Agent Matrix\n\nGenerated automatically from config/agents.yaml.\n\n"
    content += "| Agent | Description | Tools allowed |\n|---|---|---|\n"
    for name, cfg in agents.get("agents", {}).items():
        desc = cfg.get("description", "")
        tools = ", ".join(cfg.get("allowed_tools", []))
        content += f"| `{name}` | {desc} | {tools} |\n"
    return content


def gen_model_ladders() -> str:
    providers_path = Path("config/providers.yaml")
    providers = {}
    if providers_path.exists():
        providers = yaml.safe_load(providers_path.read_text(encoding="utf-8")) or {}
        
    content = "# Model Ladders\n\nGenerated automatically from config/providers.yaml.\n\n"
    for p_id, p_cfg in providers.get("providers", {}).items():
        content += f"### {p_cfg.get('name', p_id)}\n"
        for m in p_cfg.get("models", []):
            content += f"- `{m}`\n"
        content += "\n"
    return content


def gen_env_variables() -> str:
    env_example = Path(".env.example")
    lines = env_example.read_text(encoding="utf-8").splitlines() if env_example.exists() else []
    
    content = "# Environment Variables Reference\n\nGenerated from .env.example.\n\n"
    content += "```env\n" + "\n".join(lines) + "\n```\n"
    return content


def gen_db_schema() -> str:
    from memory.graph_store import GraphMemory
    import sqlite3
    import tempfile
    
    tmp = tempfile.mkdtemp()
    db_p = Path(tmp) / "schema.db"
    gm = GraphMemory(database_path=str(db_p))
    gm.migrate_schema()
    
    conn = sqlite3.connect(str(db_p))
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type IN ('table', 'view') AND sql IS NOT NULL;")
    sqls = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    content = "# Database Schema Reference (v2)\n\nGenerated from sqlite_master.\n\n```sql\n"
    content += ";\n\n".join(sqls) + ";\n```\n"
    return content


def gen_guardrail_rules() -> str:
    g_path = Path("config/guardrails.json")
    data = {}
    if g_path.exists():
        data = json.loads(g_path.read_text(encoding="utf-8"))
        
    content = "# Guardrail Rules Reference\n\nGenerated from config/guardrails.json.\n\n"
    content += "```json\n" + json.dumps(data, indent=2) + "\n```\n"
    return content


def gen_event_topics() -> str:
    from core import events
    topics = getattr(events, "EVENT_TOPICS", [
        "task.compiled", "task.all_steps_completed", "task.failed",
        "outcome.recorded", "evolution.cycle_started", "evolution.cycle_completed",
        "skill.promoted", "skill.deprecated"
    ])
    
    content = "# Event Topics Reference\n\nGenerated from core/events.py.\n\n"
    for t in topics:
        content += f"- `{t}`\n"
    return content


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    (DOCS_DIR / "tool_catalog.md").write_text(gen_tool_catalog(), encoding="utf-8")
    (DOCS_DIR / "agent_matrix.md").write_text(gen_agent_matrix(), encoding="utf-8")
    (DOCS_DIR / "model_ladders.md").write_text(gen_model_ladders(), encoding="utf-8")
    (DOCS_DIR / "env_variables.md").write_text(gen_env_variables(), encoding="utf-8")
    (DOCS_DIR / "db_schema.md").write_text(gen_db_schema(), encoding="utf-8")
    (DOCS_DIR / "guardrail_rules.md").write_text(gen_guardrail_rules(), encoding="utf-8")
    (DOCS_DIR / "event_topics.md").write_text(gen_event_topics(), encoding="utf-8")
    
    print(f"Generated docs written to {DOCS_DIR}/")


if __name__ == "__main__":
    main()
