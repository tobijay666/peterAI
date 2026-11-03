"""SQLite graph store for task lineage and cross-agent relationship tracking."""

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from core.logger import logger
from core.clock import Clock, SystemClock

VALID_NODE_KINDS = {
    "task", "agent", "artifact", "skill", "finding", "episode", "semantic",
    # Cognitive memory (schema v3)
    "entity", "abstraction", "source", "curiosity",
    # Peter himself: the root every main subject hangs off (memory/hierarchy.py)
    "self",
}
VALID_RELATIONS = {
    "produced", "used", "delegated_to", "depends_on",
    "exercised",
    "derived_from",
    "supersedes",
    "requires",
    "spawned_by",
    # Cognitive memory (schema v3)
    "has_collection", "abstracts", "bridges", "cites", "supports",
    "curious_about", "related_to",
    # Hierarchy (memory/hierarchy.py)
    "has_subject",   # Peter -> main subject
    "helper_of",     # helper agent -> Peter (plumbing, never drawn)
    "similar_to",    # topic <-> topic close in meaning (stored when a topic is created)
}

# Columns added to `nodes` in schema v3 (name -> SQL type/default)
_V3_NODE_COLUMNS = {
    "access_count":  "INTEGER DEFAULT 0",
    "last_accessed": "REAL",
    "usage_score":   "REAL DEFAULT 0",
    "collection":    "TEXT",
    "status":        "TEXT DEFAULT 'active'",
    "confidence":    "REAL DEFAULT 0.5",
    "fading_since":  "REAL",
    "pos_x":         "REAL",
    "pos_y":         "REAL",
    "pos_z":         "REAL",
}
_SETTABLE_NODE_FIELDS = set(_V3_NODE_COLUMNS) | {"importance", "pinned", "label", "last_used"}
_NODE_SELECT = (
    "SELECT id, kind, label, data, created_at, last_used, importance, pinned, "
    "access_count, last_accessed, usage_score, collection, status, confidence, "
    "fading_since, pos_x, pos_y, pos_z FROM nodes"
)

class _ClosingConnection(sqlite3.Connection):
    """`with conn:` commits/rolls back AND closes (plain sqlite3 only commits).

    Unclosed connections pile up as ResourceWarnings and, on Windows, keep the
    database file locked until garbage collection.
    """

    def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


class GraphMemory:
    """Manages relationship nodes and edges in SQLite with WAL mode."""

    def __init__(self, database_path: str = "data/peter_graph.db", clock: Optional[Clock] = None,
                 user_id: str = "local"):
        self.db_path = Path(database_path)
        self._clock = clock or SystemClock()
        self.user_id = user_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), factory=_ClosingConnection)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def close(self) -> None:
        """Checkpoints and closes any open WAL connections to release file locks."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.close()
        except Exception:
            pass

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id          TEXT PRIMARY KEY,
                    kind        TEXT NOT NULL,
                    label       TEXT NOT NULL,
                    data        TEXT,
                    created_at  REAL NOT NULL,
                    last_used   REAL NOT NULL,
                    importance  REAL DEFAULT 0.5,
                    pinned      INTEGER DEFAULT 0
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    id          TEXT PRIMARY KEY,
                    src         TEXT NOT NULL,
                    dst         TEXT NOT NULL,
                    relation    TEXT NOT NULL,
                    data        TEXT,
                    created_at  REAL NOT NULL,
                    FOREIGN KEY(src) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(dst) REFERENCES nodes(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);")
            self._migrate_to_v2(conn)
            self._migrate_to_v3(conn)
            conn.commit()
        logger.info(f"GraphMemory initialized at {self.db_path}")

    def migrate_schema(self) -> None:
        """Public entry point: (re)run all idempotent schema migrations."""
        self._init_db()

    def _migrate_to_v3(self, conn: sqlite3.Connection) -> None:
        """Schema v3: usage tracking on nodes, aliases, collection registry, curiosity queue."""
        row = conn.execute("SELECT MAX(version) as version FROM schema_version").fetchone()
        version = row["version"] if row and row["version"] is not None else 0
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}
        for col, decl in _V3_NODE_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE nodes ADD COLUMN {col} {decl};")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS node_aliases (
                alias_norm TEXT NOT NULL,
                node_id    TEXT NOT NULL,
                PRIMARY KEY (alias_norm, node_id),
                FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alias_norm ON node_aliases(alias_norm);")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS collections (
                name             TEXT PRIMARY KEY,
                node_id          TEXT,
                kind             TEXT NOT NULL,
                created_at       REAL NOT NULL,
                created_cycle_id TEXT,
                record_count     INTEGER DEFAULT 0,
                last_accessed    REAL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS curiosity_queue (
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
            );
            """
        )
        cq_cols = {r["name"] for r in conn.execute("PRAGMA table_info(curiosity_queue)").fetchall()}
        if "topic" not in cq_cols:
            conn.execute("ALTER TABLE curiosity_queue ADD COLUMN topic TEXT;")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_curiosity_status ON curiosity_queue(status, priority);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);")
        if version < 3:
            conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (?, ?)", (3, self._clock.now()))

    def _migrate_to_v2(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL,
                applied_at REAL NOT NULL
            );
            """
        )
        row = conn.execute("SELECT MAX(version) as version FROM schema_version").fetchone()
        version = row["version"] if row and row["version"] is not None else 0
        if version < 2:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outcomes (
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
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_task ON outcomes(task_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_skill ON outcomes(skill_id);")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id         TEXT PRIMARY KEY,
                    task_id    TEXT NOT NULL,
                    rating     INTEGER NOT NULL,
                    note       TEXT,
                    created_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE VIEW IF NOT EXISTS v_skill_stats AS
                SELECT o.skill_id,
                       COUNT(*) AS uses,
                       SUM(CASE WHEN o.status = 'success' AND COALESCE(f.rating, 0) >= 0 THEN 1 ELSE 0 END) AS successes
                FROM outcomes o LEFT JOIN feedback f ON f.task_id = o.task_id
                WHERE o.level = 'task' AND o.skill_id IS NOT NULL
                GROUP BY o.skill_id;
                """
            )
            conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (?, ?)", (2, self._clock.now()))

    def upsert_node(
        self,
        node_id: str,
        kind: str,
        label: str,
        data: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        pinned: bool = False,
    ) -> str:
        if kind not in VALID_NODE_KINDS:
            logger.warning(f"GraphMemory.upsert_node: Unknown node kind '{kind}'.")
        if kind == 'skill' and not data:
            data = {}
        now = self._clock.now()
        data_str = json.dumps(data) if data else "{}"
        pinned_int = 1 if pinned else 0

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO nodes (id, kind, label, data, created_at, last_used, importance, pinned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind,
                    label=excluded.label,
                    data=excluded.data,
                    last_used=excluded.last_used,
                    importance=excluded.importance,
                    pinned=excluded.pinned;
                """,
                (node_id, kind, label, data_str, now, now, importance, pinned_int),
            )
            conn.commit()
        return node_id

    def link(
        self,
        src_id: str,
        dst_id: str,
        relation: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        if relation not in VALID_RELATIONS:
            logger.warning(f"GraphMemory.link: Unknown relation '{relation}'.")
        edge_id = str(uuid.uuid4())
        now = self._clock.now()
        data_str = json.dumps(data) if data else "{}"

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO edges (id, src, dst, relation, data, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (edge_id, src_id, dst_id, relation, data_str, now),
            )
            conn.commit()
        return edge_id

    def neighbors(self, node_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Returns direct and n-hop connected graph neighbors (deduped by edge_id)."""
        results = []
        visited_nodes = set()
        seen_edges: set = set()
        queue = [(node_id, 0)]

        with self._get_connection() as conn:
            while queue:
                curr_id, depth = queue.pop(0)
                if curr_id in visited_nodes or depth > max_depth:
                    continue
                visited_nodes.add(curr_id)

                # Outgoing edges: curr -> neighbor
                out_cursor = conn.execute(
                    """
                    SELECT e.id as edge_id, e.src, e.dst, e.relation,
                           n.id as node_id, n.kind, n.label, n.data as node_data
                    FROM edges e
                    JOIN nodes n ON n.id = e.dst
                    WHERE e.src = ?;
                    """,
                    (curr_id,),
                )
                # Incoming edges: neighbor -> curr
                in_cursor = conn.execute(
                    """
                    SELECT e.id as edge_id, e.src, e.dst, e.relation,
                           n.id as node_id, n.kind, n.label, n.data as node_data
                    FROM edges e
                    JOIN nodes n ON n.id = e.src
                    WHERE e.dst = ?;
                    """,
                    (curr_id,),
                )

                for r in list(out_cursor.fetchall()) + list(in_cursor.fetchall()):
                    edge_id = r["edge_id"]
                    if edge_id in seen_edges:
                        continue
                    seen_edges.add(edge_id)

                    nbr_id = r["node_id"]
                    results.append(
                        {
                            "edge_id": edge_id,
                            "src": r["src"],
                            "dst": r["dst"],
                            "relation": r["relation"],
                            "node": {
                                "id": nbr_id,
                                "kind": r["kind"],
                                "label": r["label"],
                                "data": json.loads(r["node_data"] or "{}"),
                            },
                            "depth": depth + 1,
                        }
                    )
                    if nbr_id not in visited_nodes and depth + 1 < max_depth:
                        queue.append((nbr_id, depth + 1))
        return results

    def prune(
        self,
        max_age_days: float = 30,
        min_importance: float = 0.3,
    ) -> int:
        """Legacy decay prune. Cognitive nodes (entity/abstraction/source/curiosity)
        own vector collections and are pruned only by the sleep cycle's usage audit."""
        now = self._clock.now()
        cutoff = now - (max_age_days * 86400)
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM nodes
                WHERE pinned = 0
                  AND last_used < ?
                  AND importance < ?
                  AND kind NOT IN ('entity', 'abstraction', 'source', 'curiosity');
                """,
                (cutoff, min_importance),
            )
            count = cursor.rowcount
            conn.commit()
            logger.info(f"GraphMemory: Pruned {count} stale graph nodes.")
            return count

    @staticmethod
    def _row_to_node(r: sqlite3.Row) -> Dict[str, Any]:
        keys = r.keys()
        node = {
            "id": r["id"],
            "kind": r["kind"],
            "label": r["label"],
            "data": json.loads(r["data"] or "{}"),
            "created_at": r["created_at"],
            "last_used": r["last_used"],
            "importance": r["importance"],
            "pinned": bool(r["pinned"]),
        }
        for col in _V3_NODE_COLUMNS:
            if col in keys:
                node[col] = r[col]
        node["access_count"] = int(node.get("access_count") or 0)
        node["status"] = node.get("status") or "active"
        return node

    def get_nodes_by_kind(self, kind: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(f"{_NODE_SELECT} WHERE kind = ?;", (kind,))
            return [self._row_to_node(r) for r in cursor.fetchall()]

    def all_nodes(self, kinds: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if kinds:
                marks = ",".join("?" for _ in kinds)
                cursor = conn.execute(f"{_NODE_SELECT} WHERE kind IN ({marks});", tuple(kinds))
            else:
                cursor = conn.execute(f"{_NODE_SELECT};")
            return [self._row_to_node(r) for r in cursor.fetchall()]

    def get_skill_source_tasks(self, skill_id: str) -> List[str]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT src FROM edges WHERE dst = ? AND relation = 'derived_from';",
                (skill_id,)
            )
            return [r["src"] for r in cursor.fetchall()]

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            r = conn.execute(f"{_NODE_SELECT} WHERE id = ?;", (node_id,)).fetchone()
            return self._row_to_node(r) if r else None

    # ------------------------------------------------------------------ v3: usage + fields

    def set_node_fields(self, node_id: str, **fields: Any) -> bool:
        """Update whitelisted node columns. Returns True if the node exists."""
        cols = {k: v for k, v in fields.items() if k in _SETTABLE_NODE_FIELDS}
        if not cols:
            return False
        if "pinned" in cols:
            cols["pinned"] = 1 if cols["pinned"] else 0
        assignments = ", ".join(f"{k} = ?" for k in cols)
        with self._get_connection() as conn:
            cur = conn.execute(f"UPDATE nodes SET {assignments} WHERE id = ?;", (*cols.values(), node_id))
            conn.commit()
            return cur.rowcount > 0

    def touch(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Activation: increment access_count, set last_accessed, restore 'active' status.

        One SQL statement so concurrent touches never lose a count.
        Returns the updated node, or None if it does not exist.
        """
        now = self._clock.now()
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE nodes
                   SET access_count = COALESCE(access_count, 0) + 1,
                       last_accessed = ?,
                       last_used = ?,
                       status = CASE WHEN status = 'fading' THEN 'active' ELSE COALESCE(status, 'active') END,
                       fading_since = CASE WHEN status = 'fading' THEN NULL ELSE fading_since END
                 WHERE id = ?;
                """,
                (now, now, node_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_node(node_id)

    # ------------------------------------------------------------------ v3: edges

    def list_edges(self, node_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if node_ids is None:
                rows = conn.execute("SELECT id, src, dst, relation, data, created_at FROM edges;").fetchall()
            else:
                ids = list(node_ids)
                if not ids:
                    return []
                marks = ",".join("?" for _ in ids)
                rows = conn.execute(
                    f"SELECT id, src, dst, relation, data, created_at FROM edges "
                    f"WHERE src IN ({marks}) AND dst IN ({marks});",
                    (*ids, *ids),
                ).fetchall()
            return [
                {"id": r["id"], "src": r["src"], "dst": r["dst"], "relation": r["relation"],
                 "data": json.loads(r["data"] or "{}"), "created_at": r["created_at"]}
                for r in rows
            ]

    def edges_of(self, node_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, src, dst, relation, created_at FROM edges WHERE src = ? OR dst = ?;",
                (node_id, node_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def find_edge(self, src_id: str, dst_id: str, relation: str) -> Optional[str]:
        with self._get_connection() as conn:
            r = conn.execute(
                "SELECT id FROM edges WHERE src = ? AND dst = ? AND relation = ? LIMIT 1;",
                (src_id, dst_id, relation),
            ).fetchone()
            return r["id"] if r else None

    def link_once(self, src_id: str, dst_id: str, relation: str,
                  data: Optional[Dict[str, Any]] = None) -> Tuple[str, bool]:
        """Create the edge only if an identical one does not exist. Returns (edge_id, created)."""
        existing = self.find_edge(src_id, dst_id, relation)
        if existing:
            return existing, False
        return self.link(src_id, dst_id, relation, data), True

    def degree(self, node_id: str) -> int:
        with self._get_connection() as conn:
            r = conn.execute(
                "SELECT COUNT(*) AS c FROM edges WHERE src = ? OR dst = ?;", (node_id, node_id)
            ).fetchone()
            return int(r["c"])

    def delete_node(self, node_id: str) -> bool:
        """Delete a node; its edges and aliases cascade. Pinned nodes are refused."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM nodes WHERE id = ? AND pinned = 0;", (node_id,))
            conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------ v3: aliases

    @staticmethod
    def normalize_alias(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))

    def add_alias(self, node_id: str, alias: str) -> None:
        norm = self.normalize_alias(alias)
        if not norm:
            return
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO node_aliases (alias_norm, node_id) VALUES (?, ?);", (norm, node_id)
            )
            conn.commit()

    def resolve_alias(self, text: str) -> List[str]:
        norm = self.normalize_alias(text)
        if not norm:
            return []
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT a.node_id FROM node_aliases a JOIN nodes n ON n.id = a.node_id "
                "WHERE a.alias_norm = ? AND COALESCE(n.status, 'active') != 'pruned';",
                (norm,),
            ).fetchall()
            return [r["node_id"] for r in rows]

    def aliases_of(self, node_id: str) -> List[str]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT alias_norm FROM node_aliases WHERE node_id = ?;", (node_id,)).fetchall()
            return [r["alias_norm"] for r in rows]

    # ------------------------------------------------------------------ v3: collection registry

    def register_collection(self, name: str, node_id: Optional[str], kind: str,
                            cycle_id: Optional[str] = None) -> None:
        now = self._clock.now()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO collections (name, node_id, kind, created_at, created_cycle_id, record_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(name) DO UPDATE SET node_id = excluded.node_id, kind = excluded.kind;
                """,
                (name, node_id, kind, now, cycle_id, now),
            )
            conn.commit()

    def unregister_collection(self, name: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM collections WHERE name = ?;", (name,))
            conn.commit()

    def list_collections(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if kind:
                rows = conn.execute("SELECT * FROM collections WHERE kind = ?;", (kind,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM collections;").fetchall()
            return [dict(r) for r in rows]

    def get_collection(self, name: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            r = conn.execute("SELECT * FROM collections WHERE name = ?;", (name,)).fetchone()
            return dict(r) if r else None

    def update_collection_stats(self, name: str, record_count: Optional[int] = None,
                                accessed: bool = False) -> None:
        sets, args = [], []
        if record_count is not None:
            sets.append("record_count = ?")
            args.append(int(record_count))
        if accessed:
            sets.append("last_accessed = ?")
            args.append(self._clock.now())
        if not sets:
            return
        with self._get_connection() as conn:
            conn.execute(f"UPDATE collections SET {', '.join(sets)} WHERE name = ?;", (*args, name))
            conn.commit()
