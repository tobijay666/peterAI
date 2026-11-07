"""Pre-cycle snapshot of Chroma + graph (FR-20.3).

Keeps the last 3 snapshots.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Optional
from core.clock import Clock, SystemClock
from core.logger import logger

SNAPSHOT_DIR = Path("data/snapshots")
MAX_SNAPSHOTS = 3


class SnapshotManager:
    """Takes and manages pre-cycle snapshots of the memory stores."""

    def __init__(
        self,
        chroma_dir: str = "data/chroma",
        graph_db: str = "data/peter_graph.db",
        snapshot_dir: Optional[Path] = None,
        clock: Optional[Clock] = None,
        max_snapshots: int = MAX_SNAPSHOTS,
    ) -> None:
        self.chroma_dir = Path(chroma_dir)
        self.graph_db = Path(graph_db)
        self.snapshot_dir = snapshot_dir or SNAPSHOT_DIR
        self._clock = clock or SystemClock()
        self.max_snapshots = max_snapshots

    def take_snapshot(self) -> Optional[Path]:
        """Copy Chroma dir and graph DB to a timestamped snapshot folder.
        Never raises. Returns the snapshot path or None on failure.
        """
        try:
            ts = int(self._clock.now())
            snap_path = self.snapshot_dir / f"snap_{ts}"
            snap_path.mkdir(parents=True, exist_ok=True)

            if self.chroma_dir.exists():
                shutil.copytree(str(self.chroma_dir), str(snap_path / "chroma"), dirs_exist_ok=True)
            if self.graph_db.exists():
                # sqlite backup API: a consistent copy even with WAL pages not yet checkpointed
                import sqlite3
                src = sqlite3.connect(str(self.graph_db))
                dst = sqlite3.connect(str(snap_path / "peter_graph.db"))
                try:
                    src.backup(dst)
                finally:
                    dst.close()
                    src.close()

            # Write manifest
            (snap_path / "manifest.json").write_text(
                json.dumps({"timestamp": ts, "chroma_dir": str(self.chroma_dir), "graph_db": str(self.graph_db)}),
                encoding="utf-8",
            )
            logger.info(f"SnapshotManager: Snapshot taken at {snap_path}")
            self._prune_old()
            return snap_path
        except Exception as e:
            logger.error(f"SnapshotManager.take_snapshot failed (non-fatal): {e}")
            return None

    def list_snapshots(self):
        """Return sorted list of snapshot paths (oldest first)."""
        try:
            return sorted([p for p in self.snapshot_dir.iterdir() if p.is_dir() and p.name.startswith("snap_")])
        except Exception:
            return []

    def _prune_old(self) -> None:
        """Keep only the last max_snapshots snapshots."""
        snaps = self.list_snapshots()
        for old in snaps[:-self.max_snapshots]:
            try:
                shutil.rmtree(str(old))
                logger.info(f"SnapshotManager: Removed old snapshot {old}")
            except Exception as e:
                logger.warning(f"SnapshotManager: Could not remove {old}: {e}")


def snapshot_manager_for(vector: object, graph: object, clock: Optional[Clock] = None,
                         max_snapshots: int = 7) -> "SnapshotManager":
    """Per-user snapshots (D6: keep 7) next to that user's own graph DB.

    The default SnapshotManager() points at the REPL's data/chroma and
    data/peter_graph.db, which is wrong for web users under data/users/<id>/.
    """
    graph_db = Path(getattr(graph, "db_path"))
    return SnapshotManager(
        chroma_dir=str(getattr(vector, "persist_dir")),
        graph_db=str(graph_db),
        snapshot_dir=graph_db.parent / "snapshots",
        clock=clock,
        max_snapshots=max_snapshots,
    )


def restore_snapshot(snap_path: Path, chroma_dir: Path, graph_db: Path) -> None:
    """Copy a snapshot back over the live stores. Peter must be stopped first."""
    snap_path, chroma_dir, graph_db = Path(snap_path), Path(chroma_dir), Path(graph_db)
    if (snap_path / "chroma").exists():
        if chroma_dir.exists():
            shutil.rmtree(str(chroma_dir))
        shutil.copytree(str(snap_path / "chroma"), str(chroma_dir))
    if (snap_path / "peter_graph.db").exists():
        for suffix in ("-wal", "-shm"):
            side = Path(str(graph_db) + suffix)
            if side.exists():
                side.unlink()
        shutil.copy2(str(snap_path / "peter_graph.db"), str(graph_db))
