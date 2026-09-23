# Peter AI — Multi-Agent AI Assistant Harness

Peter is a locally-executing, multi-agent AI assistant harness designed for Sir Anton. It coordinates a fleet of specialist agents (**Researcher**, **FileSystem**, **Librarian**, **DevOps**, **QA**, **Voice**) orchestrated by a central conversational front-end named **Peter**.

## How to Work in the REPO:
1. Create a branch for your self or for a given ISSUE. 
2. Once the update or the fix has been provided the work can be published as a PR, to the "working" Branch.
3. The "Main" branch is not to be tampered with until the "working" branch is 100% working.
4. Peter will be deployed through the main branch. 
5. in case of a dire situation please don't do anything. 


## Architecture & Features

- **Conversational REPL**: Peter greets and interacts respectfully ("Sir" / "Mr. Anton").
- **Specialist Fleet**: Orchestration via OpenAI function calling and tool delegation.
- **Fail-Soft LLM Routing**: 4 providers (NVIDIA, Groq, OpenRouter, HuggingFace) with 3+ model ladders per provider and automatic circuit breaker fallback.
- **Dual Memory Subsystem**:
  - Vector Store: ChromaDB with 7 namespaced collections and local embeddings (`all-MiniLM-L6-v2`).
  - Graph Store: SQLite graph DB tracking task lineage and cross-agent relationships.
  - Decay Eviction: 24-hour background worker preserving pinned records.
- **Security Guardrails Engine**: Dynamic security rules protecting against path traversal (`../`), credential leakage (`.env`), system directory access, and destructive commands.
- **Non-Blocking TTS**: pyttsx3 voice queue worker with Windows COM support.

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
```bash
# Clone or navigate to directory
cd d:/peterAI

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and fill in your LLM API keys:
```bash
cp .env.example .env
```

### 4. Running Peter AI
```bash
python peter.py
```

## Running Tests
```bash
pytest tests/unit tests/integration -v
```

## Developer Documentation
Developer plans and architecture specifications are saved in `DevNotes/` (ignored by git).
- `DevNotes/ImplementationPlan.md`
- `DevNotes/SRAS.md`
