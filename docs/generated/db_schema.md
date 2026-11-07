# Database Schema Reference (v2)

Generated from sqlite_master.

```sql
CREATE TABLE nodes (
                    id          TEXT PRIMARY KEY,
                    kind        TEXT NOT NULL,
                    label       TEXT NOT NULL,
                    data        TEXT,
                    created_at  REAL NOT NULL,
                    last_used   REAL NOT NULL,
                    importance  REAL DEFAULT 0.5,
                    pinned      INTEGER DEFAULT 0
                , access_count INTEGER DEFAULT 0, last_accessed REAL, usage_score REAL DEFAULT 0, collection TEXT, status TEXT DEFAULT 'active', confidence REAL DEFAULT 0.5, fading_since REAL, pos_x REAL, pos_y REAL, pos_z REAL);

CREATE TABLE edges (
                    id          TEXT PRIMARY KEY,
                    src         TEXT NOT NULL,
                    dst         TEXT NOT NULL,
                    relation    TEXT NOT NULL,
                    data        TEXT,
                    created_at  REAL NOT NULL,
                    FOREIGN KEY(src) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(dst) REFERENCES nodes(id) ON DELETE CASCADE
                );

CREATE TABLE schema_version (
                version INTEGER NOT NULL,
                applied_at REAL NOT NULL
            );

CREATE TABLE outcomes (
                    id          TEXT PRIMARY KEY,
                    task_id     TEXT NOT NULL,
                    step_id     TEXT,
                    level       TEXT NOT NULL,
                    agent       TEXT NOT NULL,
                    tool        TEXT,
                    skill_id    TEXT,
                    provider    TEXT,
                    model       TEXT,
                    status      TEXT NOT NULL,
                    latency_ms  INTEGER,
                    tokens_in   INTEGER,
                    tokens_out  INTEGER,
                    memory_ids  TEXT,
                    created_at  REAL NOT NULL
                );

CREATE TABLE feedback (
                    id         TEXT PRIMARY KEY,
                    task_id    TEXT NOT NULL,
                    rating     INTEGER NOT NULL,
                    note       TEXT,
                    created_at REAL NOT NULL
                );

CREATE VIEW v_skill_stats AS
                SELECT o.skill_id,
                       COUNT(*) AS uses,
                       SUM(CASE WHEN o.status = 'success' AND COALESCE(f.rating, 0) >= 0 THEN 1 ELSE 0 END) AS successes
                FROM outcomes o LEFT JOIN feedback f ON f.task_id = o.task_id
                WHERE o.level = 'task' AND o.skill_id IS NOT NULL
                GROUP BY o.skill_id;

CREATE TABLE node_aliases (
                alias_norm TEXT NOT NULL,
                node_id    TEXT NOT NULL,
                PRIMARY KEY (alias_norm, node_id),
                FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
            );

CREATE TABLE collections (
                name             TEXT PRIMARY KEY,
                node_id          TEXT,
                kind             TEXT NOT NULL,
                created_at       REAL NOT NULL,
                created_cycle_id TEXT,
                record_count     INTEGER DEFAULT 0,
                last_accessed    REAL
            );

CREATE TABLE curiosity_queue (
                id             TEXT PRIMARY KEY,
                question       TEXT NOT NULL,
                topic_node_id  TEXT,
                priority       REAL NOT NULL DEFAULT 0,
                status         TEXT NOT NULL DEFAULT 'pending',
                origin_turn_id TEXT,
                attempts       INTEGER NOT NULL DEFAULT 0,
                gap            INTEGER NOT NULL DEFAULT 0,
                created_at     REAL NOT NULL,
                updated_at     REAL NOT NULL
            , topic TEXT);
```
