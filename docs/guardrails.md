# Security & Guardrails Engine Reference

The Guardrail Engine enforces dynamic security policies at three evaluation points:
1. `check_input()`: Filters user prompts before LLM submission.
2. `check_tool_call()`: Validates tool names and parameters before tool execution.
3. `check_output()`: Sanitizes LLM text output before rendering to user.

## Rule Definitions (`config/guardrails.json`)

| Rule ID | Severity | Scope | Description |
|---|---|---|---|
| `block_path_traversal` | `block` | `tool_call` | Rejects directory traversal sequences (`../`, `..\`) |
| `block_system_prompt_leak` | `block` | `input`, `output` | Rejects prompt extraction attempts |
| `block_secret_extraction` | `block` | `tool_call`, `output` | Rejects access to `.env`, `.pem`, `.ssh`, browser data |
| `block_mass_destruction` | `block` | `input`, `tool_call` | Rejects `shutil.rmtree`, `rm -rf`, `format c:` |
| `block_sensitive_directories` | `block` | `tool_call` | Rejects OS system folder access (`C:\Windows`, `/etc/`) |
| `max_file_size` | `warn` | `tool_call` | Triggers warning on payloads > 500,000 characters |

## Dynamic Hot-Reloading
Rule files can be reloaded live during REPL runtime without restarting the process:
```
Sir Anton > reload-guardrails
```
