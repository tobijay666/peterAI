"""SQLite graph store for task lineage and cross-agent relationship tracking."""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.logger import logger


class GraphMemory:
    """Manages relationship nodes and edges in SQLite with WAL mode."""

    def __init__(self, database_path: str = "data/peter_graph.db"):
        self.db_path = Path(database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
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
            conn.commit()
        logger.info(f"GraphMemory initialized at {self.db_path}")

    def upsert_node(
        self,
        node_id: str,
        kind: str,
        label: str,
        data: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        pinned: bool = False,
    ) -> str:
        now = time.time()
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
        edge_id = str(uuid.uuid4())
        now = time.time()
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
        now = time.time()
        cutoff = now - (max_age_days * 86400)
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM nodes
                WHERE pinned = 0
                  AND last_used < ?
                  AND importance < ?;
                """,
                (cutoff, min_importance),
            )
            count = cursor.rowcount
            conn.commit()
            logger.info(f"GraphMemory: Pruned {count} stale graph nodes.")
            return count
