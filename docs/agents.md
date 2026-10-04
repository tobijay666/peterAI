# Specialist Fleet Agent Reference

## 1. Peter (Orchestrator)
- **Namespace**: `peter`
- **Role**: Primary user-facing assistant.
- **Addressing**: Always addresses user as "Sir" or "Mr. Anton".
- **Primary Tools**: Delegation tools (`delegate_to_researcher`, `delegate_to_filesystem`, `delegate_to_librarian`, `delegate_to_devops`, `delegate_to_qa`, `delegate_to_voice`), memory recall/store, TTS speak.

## 2. Researcher Agent
- **Namespace**: `researcher`
- **Role**: Web search & scholarly research.
- **Primary Tools**: `web_search` (DuckDuckGo), `fetch_url` (trafilatura), `scholarly_search` (arXiv API), `memory_store`.

## 3. FileSystem Agent
- **Namespace**: `filesystem`
- **Role**: File and directory manipulation within safe security boundaries.
- **Read Scope**: Default `D:\` (plus `FS_READ_ALLOWLIST`).
- **Write Scope**: Strictly `D:\Peter_Workspace\` (or `./workspace`).
- **Primary Tools**: `list_directory`, `read_file`, `read_pdf`, `write_file`, `create_directory`, `search_files`.

## 4. Librarian Agent
- **Namespace**: `librarian`
- **Role**: Memory curation and knowledge graph management.
- **Primary Tools**: `memory_recall`, `memory_store`, `memory_prune`, `memory_pin`, `graph_link`, `graph_query`.

## 5. DevOps Agent
- **Namespace**: `devops`
- **Role**: Workspace Git operations & checkpoint commits.
- **Gate**: Requires passing QA test suite before executing commit.
- **Primary Tools**: `git_status`, `git_commit`, `git_branch`, `git_log`, `run_tests`.

## 6. QA Agent
- **Namespace**: `qa`
- **Role**: Testing specialist authoring and executing pytest and hypothesis suites.
- **Primary Tools**: `read_file`, `write_file`, `run_pytest`, `run_hypothesis`, `list_directory`.

## 7. Voice Agent
- **Namespace**: `voice`
- **Role**: Text-to-speech audio synthesis wrapper.
- **Primary Tools**: `tts_speak`.
