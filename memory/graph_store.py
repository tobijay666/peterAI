"""SQLite graph store for task lineage and cross-agent relationship tracking."""

import json
import re
import sqlite3
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
    # position in the force-directed "graph" layout (memory/graph_layout.py), kept between snapshots
    "gpos_x":        "REAL",
    "gpos_y":        "REAL",
    "gpos_z":        "REAL",
}
_SETTABLE_NODE_FIELDS = set(_V3_NODE_COLUMNS) | {"importance", "pinned", "label", "last_used"}
_NODE_SELECT = (
    "SELECT id, kind, label, data, created_at, last_used, importance, pinned, "
    "access_count, last_accessed, usage_score, collection, status, confidence, "
    "fading_since, pos_x, pos_y, pos_z, gpos_x, gpos_y, gpos_z FROM nodes"
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
            self._migrate_vectors(conn)
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

    def _migrate_vectors(self, conn: sqlite3.Connection) -> None:
        """Vector-grounded graph (idempotent, additive):
        node_vectors  one vector per topic node (its collection centroid, or its label while the
                      collection is empty) and Peter's own (mean of his subjects)
        vector_links  record-level links of a topic-topic edge: the most similar record pairs
                      across the two collections (deleted with the edge)."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS node_vectors (
                node_id    TEXT PRIMARY KEY,
                vector     TEXT NOT NULL,
                basis      TEXT NOT NULL DEFAULT 'centroid',
                n          INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_links (
                edge_id    TEXT NOT NULL,
                src_node   TEXT NOT NULL,
                dst_node   TEXT NOT NULL,
                src_record TEXT NOT NULL,
                dst_record TEXT NOT NULL,
                similarity REAL NOT NULL,
                PRIMARY KEY (edge_id, src_record, dst_record),
                FOREIGN KEY(edge_id) REFERENCES edges(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vlinks_src ON vector_links(src_record);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vlinks_dst ON vector_links(dst_record);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS graph_meta (key TEXT PRIMARY KEY, value TEXT, updated_at REAL);"
        )

    # ------------------------------------------------------------------ small key/value meta

    def get_meta(self, key: str) -> Optional[str]:
        with self._get_connection() as conn:
            r = conn.execute("SELECT value FROM graph_meta WHERE key = ?;", (key,)).fetchone()
            return r["value"] if r else None

    def set_meta(self, key: str, value: Optional[str]) -> None:
        with self._get_connection() as conn:
            if value is None:
                conn.execute("DELETE FROM graph_meta WHERE key = ?;", (key,))
            else:
                conn.execute(
                    "INSERT INTO graph_meta (key, value, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;",
                    (key, value, self._clock.now()),
                )
            conn.commit()

    # ------------------------------------------------------------------ vectors of nodes / record links

    def set_node_vector(self, node_id: str, vector: List[float], basis: str, n: int) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO node_vectors (node_id, vector, basis, n, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(node_id) DO UPDATE SET vector = excluded.vector, basis = excluded.basis, "
                "n = excluded.n, updated_at = excluded.updated_at;",
                (node_id, json.dumps([round(float(x), 6) for x in vector]), basis, int(n), self._clock.now()),
            )
            conn.commit()

    def get_node_vector(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            r = conn.execute("SELECT * FROM node_vectors WHERE node_id = ?;", (node_id,)).fetchone()
            if not r:
                return None
            return {"node_id": r["node_id"], "vector": json.loads(r["vector"]), "basis": r["basis"],
                    "n": int(r["n"]), "updated_at": r["updated_at"]}

    def replace_vector_links(self, edge_id: str, rows: List[Tuple[str, str, str, str, float]]) -> None:
        """rows: (src_node, dst_node, src_record, dst_record, similarity)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM vector_links WHERE edge_id = ?;", (edge_id,))
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO vector_links (edge_id, src_node, dst_node, src_record, dst_record, "
                    "similarity) VALUES (?, ?, ?, ?, ?, ?);",
                    [(edge_id, a, b, ra, rb, float(sim)) for a, b, ra, rb, sim in rows],
                )
            conn.commit()

    def replace_vector_links_many(self, by_edge: Dict[str, List[Tuple[str, str, str, str, float]]]) -> None:
        """replace_vector_links for many edges in one transaction."""
        if not by_edge:
            return
        with self._get_connection() as conn:
            ids = list(by_edge)
            for i in range(0, len(ids), 400):
                chunk = ids[i:i + 400]
                conn.execute(f"DELETE FROM vector_links WHERE edge_id IN ({','.join('?' for _ in chunk)});",
                             tuple(chunk))
            rows = [(eid, a, b, ra, rb, float(sim)) for eid, rs in by_edge.items() for a, b, ra, rb, sim in rs]
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO vector_links (edge_id, src_node, dst_node, src_record, dst_record, "
                    "similarity) VALUES (?, ?, ?, ?, ?, ?);", rows)
            conn.commit()

    def vector_links_for_records(self, record_ids: List[str]) -> List[Dict[str, Any]]:
        ids = list(dict.fromkeys(record_ids))
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM vector_links WHERE src_record IN ({marks}) OR dst_record IN ({marks});",
                (*ids, *ids),
            ).fetchall()
            return [dict(r) for r in rows]

    def vector_links_of_edges(self, edge_ids: List[str]) -> List[Dict[str, Any]]:
        ids = list(dict.fromkeys(edge_ids))
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        with self._get_connection() as conn:
            rows = conn.execute(f"SELECT * FROM vector_links WHERE edge_id IN ({marks});", tuple(ids)).fetchall()
            return [dict(r) for r in rows]

    def delete_vector_links_for_records(self, record_ids: List[str]) -> int:
        ids = list(dict.fromkeys(record_ids))
        if not ids:
            return 0
        marks = ",".join("?" for _ in ids)
        with self._get_connection() as conn:
            cur = conn.execute(f"DELETE FROM vector_links WHERE src_record IN ({marks}) OR dst_record IN ({marks});",
                               (*ids, *ids))
            conn.commit()
            return cur.rowcount

    def count_vector_links(self) -> int:
        with self._get_connection() as conn:
            return int(conn.execute("SELECT COUNT(*) AS c FROM vector_links;").fetchone()["c"])

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
        importance: Optional[float] = None,
        pinned: Optional[bool] = None,
    ) -> str:
        """Insert a node, or update kind/label/last_used of an existing one.

        ``data``, ``importance`` and ``pinned`` are only changed on an existing node when
        passed explicitly (not None): a plain re-upsert never wipes a subject's role, a
        pinned flag or a learned importance. New nodes default to {} / 0.5 / False.
        """
        if kind not in VALID_NODE_KINDS:
            logger.warning(f"GraphMemory.upsert_node: Unknown node kind '{kind}'.")
        now = self._clock.now()
        has_data, has_imp, has_pin = data is not None, importance is not None, pinned is not None
        data_str = json.dumps(data) if data else "{}"
        imp = float(importance) if has_imp else 0.5
        pinned_int = 1 if pinned else 0

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO nodes (id, kind, label, data, created_at, last_used, importance, pinned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind,
                    label=excluded.label,
                    data=CASE WHEN ? THEN excluded.data ELSE data END,
                    last_used=excluded.last_used,
                    importance=CASE WHEN ? THEN excluded.importance ELSE importance END,
                    pinned=CASE WHEN ? THEN excluded.pinned ELSE pinned END;
                """,
                (node_id, kind, label, data_str, now, now, imp, pinned_int,
                 1 if has_data else 0, 1 if has_imp else 0, 1 if has_pin else 0),
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

    def link_many(self, triples: List[Tuple[str, str, str, Optional[Dict[str, Any]]]]) -> List[str]:
        """Insert many edges in ONE transaction (bulk sleep work; the caller has already
        deduped and validated). Returns the new edge ids in order."""
        if not triples:
            return []
        now = self._clock.now()
        rows, ids = [], []
        for src_id, dst_id, relation, data in triples:
            eid = str(uuid.uuid4())
            ids.append(eid)
            rows.append((eid, src_id, dst_id, relation, json.dumps(data) if data else "{}", now))
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT INTO edges (id, src, dst, relation, data, created_at) VALUES (?, ?, ?, ?, ?, ?);", rows
            )
            conn.commit()
        return ids

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

    def set_graph_positions(self, positions: Dict[str, Tuple[float, float, float]]) -> None:
        """Store graph-layout positions of many nodes in one transaction."""
        if not positions:
            return
        with self._get_connection() as conn:
            conn.executemany("UPDATE nodes SET gpos_x = ?, gpos_y = ?, gpos_z = ? WHERE id = ?;",
                             [(p[0], p[1], p[2], nid) for nid, p in positions.items()])
            conn.commit()

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

    def edge_pairs(self, relation: str) -> List[Tuple[str, str]]:
        """(src, dst) of every edge with this relation (one cheap query)."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT src, dst FROM edges WHERE relation = ?;", (relation,)).fetchall()
            return [(r["src"], r["dst"]) for r in rows]

    def edges_of(self, node_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, src, dst, relation, data, created_at FROM edges WHERE src = ? OR dst = ?;",
                (node_id, node_id),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                raw = d.get("data")
                try:
                    d["data"] = json.loads(raw) if raw and raw != "{}" else {}
                except Exception:
                    d["data"] = {}
                out.append(d)
            return out

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

    def delete_edge(self, edge_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM edges WHERE id = ?;", (edge_id,))
            conn.commit()
            return cur.rowcount > 0

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            r = conn.execute("SELECT id, src, dst, relation, data, created_at FROM edges WHERE id = ?;",
                             (edge_id,)).fetchone()
            if not r:
                return None
            return {"id": r["id"], "src": r["src"], "dst": r["dst"], "relation": r["relation"],
                    "data": json.loads(r["data"] or "{}"), "created_at": r["created_at"]}

    def update_edge_data(self, edge_id: str, **fields: Any) -> bool:
        """Merge fields into an edge's JSON data (None removes a key)."""
        e = self.get_edge(edge_id)
        if e is None:
            return False
        data = dict(e["data"] or {})
        for k, v in fields.items():
            if v is None:
                data.pop(k, None)
            else:
                data[k] = v
        with self._get_connection() as conn:
            conn.execute("UPDATE edges SET data = ? WHERE id = ?;", (json.dumps(data), edge_id))
            conn.commit()
        return True

    def set_edge_data_many(self, items: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Replace the JSON data of many edges in one transaction: [(edge_id, data)]."""
        if not items:
            return
        with self._get_connection() as conn:
            conn.executemany("UPDATE edges SET data = ? WHERE id = ?;",
                             [(json.dumps(d), eid) for eid, d in items])
            conn.commit()

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
        """Alias key. Keeps meaningful symbols so 'C', 'C++' and 'C#' are three keys."""
        return " ".join(re.findall(r"[a-z0-9]+[+#]*|[+#]+", (text or "").lower()))

    @staticmethod
    def legacy_normalize_alias(text: str) -> str:
        """The alias key of schema v3 (symbols dropped); used only to upgrade old rows."""
        return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))

    def remove_alias(self, node_id: str, alias_norm: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM node_aliases WHERE alias_norm = ? AND node_id = ?;", (alias_norm, node_id))
            conn.commit()

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

    def bump_collection_count(self, name: str, delta: int) -> None:
        """Add `delta` to a collection's record_count in one statement (no re-count).
        A later authoritative recount (sleep reconcile / delete) corrects any drift."""
        if not delta:
            return
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE collections SET record_count = MAX(0, COALESCE(record_count, 0) + ?) WHERE name = ?;",
                (int(delta), name),
            )
            conn.commit()

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
