# Environment Variables Reference

Generated from .env.example.

```env
# ---- LLM Provider Keys ----
NVIDIA_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=
HF_API_KEY=

# ---- Filesystem Security Scope ----
PETER_WORKSPACE=D:\Peter_Workspace
FS_READ_ROOT=D:\
FS_READ_ALLOWLIST=
FS_AUDIT_LOG=data/fs_audit.log

# ---- Voice & TTS Settings ----
TTS_ENABLED=true
TTS_RATE=150
TTS_VOLUME=1.0
TTS_VOICE_ID=

# ---- Memory Settings ----
MEMORY_EVICTION_MODE=decay
MEMORY_MAX_AGE_DAYS=30
MEMORY_MIN_ACCESS_COUNT=2
MEMORY_IMPORTANCE_THRESHOLD=0.3
MEMORY_EVICTION_INTERVAL_HOURS=24

# ---- Git Workflow Settings ----
GIT_REMOTE_PUSH_ENABLED=false
GIT_DEFAULT_BRANCH=main

# ---- System Logging & Probing ----
LOG_LEVEL=INFO
NO_COLOR=
HEALTH_PROBE_INTERVAL_SEC=600
HEALTH_COOLDOWN_SEC=300

# ---- Async Task Queue & Provider Reliability ----
COMPLETION_TIMEOUT_SEC=90
NUM_TASK_WORKERS=2

# ---- Evolution & Learning Loop (v1.3.0) ----
# EVOLUTION_MODE: shadow (report only) | active (mutate) | off (do nothing)
# Default: shadow. Switch to 'active' deliberately after review.
EVOLUTION_MODE=shadow
# Maximum tokens to spend on auto-recalled context injected into the planner.
RECALL_TOKEN_BUDGET=1200

```
