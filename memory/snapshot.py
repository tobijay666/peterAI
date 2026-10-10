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
    ) -> None:
        self.chroma_dir = Path(chroma_dir)
        self.graph_db = Path(graph_db)
        self.snapshot_dir = snapshot_dir or SNAPSHOT_DIR
        self._clock = clock or SystemClock()

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
                shutil.copy2(str(self.graph_db), str(snap_path / "peter_graph.db"))

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
        """Keep only the last MAX_SNAPSHOTS snapshots."""
        snaps = self.list_snapshots()
        for old in snaps[:-MAX_SNAPSHOTS]:
            try:
                shutil.rmtree(str(old))
                logger.info(f"SnapshotManager: Removed old snapshot {old}")
            except Exception as e:
                logger.warning(f"SnapshotManager: Could not remove {old}: {e}")
