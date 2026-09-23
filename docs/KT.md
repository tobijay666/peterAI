# Knowledge Transfer (KT) Guide — Peter AI Harness

## System Overview
Peter AI is a modular, multi-agent AI assistant framework built in Python 3.10+. It extends single-agent designs into a distributed fleet coordinated by **Peter**, the primary orchestrator.

## Subsystem Map

```
d:/peterAI/
├── peter.py                     # System bootstrapper and main entry point
├── config/                      # Declarative configuration (providers, agents, guardrails, memory)
├── core/                        # Data interfaces, logging, console visualizer, security guardrails, events
├── llm/                         # Provider abstractions (NVIDIA, Groq, OpenRouter, HF), router, health prober
├── memory/                      # ChromaDB vector store, SQLite graph store, decay eviction scheduler
├── tools/                       # Tool implementations (filesystem, web, git, memory) & tool registry
├── tts/                         # Non-blocking pyttsx3 voice queue worker
├── agents/                      # Peter orchestrator & 6 specialist agents
├── engine/                      # Interactive REPL loop
├── tests/                       # Unit, integration, property test suite
└── DevNotes/                    # Architecture SRAS and implementation plan (git-ignored)
```

## Core Workflows

1. **User Request Intake**:
   - User inputs natural language prompt in REPL (`engine/loop.py`).
   - Input evaluated against `GuardrailManager` security rules (`core/guardrails.py`).

2. **Agent Delegation**:
   - Peter processes user intent and invokes `delegate_to_<specialist>` function tool call.
   - Delegation records task node and `spawned_by` relationship edge in SQLite graph memory (`memory/graph_store.py`).
   - Specialist agent executes reasoning loop up to `max_steps` threshold.

3. **LLM Execution & Fallback**:
   - Agent delegates prompt completion to `LLMRouter` (`llm/router.py`).
   - Router attempts completion along agent's provider policy ladder (e.g. NVIDIA -> Groq -> OpenRouter -> HuggingFace).
   - If an endpoint fails, circuit breaker (`llm/health.py`) records failure and router falls through to next provider.

4. **Memory Retention & Decay Eviction**:
   - Insights and findings persist to ChromaDB vector store (`memory/vector_store.py`) under agent namespace.
   - `EvictionScheduler` (`memory/eviction.py`) runs every 24h, applying decay rules to prune unpinned, low-importance, stale records.

5. **DevOps & QA Commit Gate**:
   - DevOps agent initiates git commit via `git_tools.py`.
   - Pre-commit hook automatically runs `pytest` suite; commit aborts if any test fails.

## Maintenance Recipes

- **Adding a Specialist Agent**:
  1. Create `agents/<name>.py` subclassing `BaseAgent`.
  2. Add prompt & policy to `config/agents.yaml`.
  3. Register delegation tool on Peter (`agents/peter.py`).
  4. Add namespace to `config/memory.yaml`.

- **Adding an LLM Provider**:
  1. Add provider entry to `config/providers.yaml`.
  2. Create class in `llm/providers/<name>.py` extending `BaseProvider`.
  3. Register in `PROVIDER_CLASSES` in `llm/router.py`.
