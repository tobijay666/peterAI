# Changelog

All notable changes to Peter AI will be documented in this file.

## [1.0.0] - 2026-09-19

### Added
- Multi-agent architecture with Peter as primary orchestrator and 6 specialists (Researcher, FileSystem, Librarian, DevOps, QA, Voice).
- Multi-provider LLM router supporting NVIDIA, Groq, OpenRouter, and HuggingFace with model ladders and automatic failover.
- ChromaDB vector memory with 7 namespaced collections and sentence-transformer embeddings.
- SQLite graph store for task lineage tracking.
- 24-hour background decay memory eviction scheduler.
- Security guardrail engine blocking path traversal, credential leakage, system directory access, and destructive commands.
- Non-blocking TTS worker thread using pyttsx3 with COM support.
- FileSystem tool isolation restricting writes to workspace boundary.
- Mandatory QA gate in DevOps git commit tool.
- DevNotes directory structure for architectural documentation.
