"""Unit tests for GraphMemory schema migration (B1)."""
from memory.graph_store import GraphMemory
import sqlite3

def test_schema_migration_fresh_db(temp_dir, fake_clock):
    db_path = str(temp_dir / "graph.db")
    gm = GraphMemory(db_path, clock=fake_clock)
    
    with gm._get_connection() as conn:
        row = conn.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
        assert row["v"] == 2
        
        # Verify tables exist
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t["name"] for t in tables}
        assert "outcomes" in table_names
        assert "feedback" in table_names
        
    gm.close()

def test_schema_migration_idempotency(temp_dir, fake_clock):
    db_path = str(temp_dir / "graph.db")
    gm1 = GraphMemory(db_path, clock=fake_clock)
    gm1.close()
    
    # Init again
    gm2 = GraphMemory(db_path, clock=fake_clock)
    
    with gm2._get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) as c FROM schema_version").fetchone()
        # Should still be just 1 row (or the logic just inserts another row but version is same, wait logic does insert).
        # Ah, the logic inserts another row with version 2? Wait! 
        # Actually in _migrate_to_v2:
        # if version < 2:
        #    ...
        #    conn.execute("INSERT INTO schema_version ...")
        # Since version is 2 on second run, it won't enter the `if version < 2:` block!
        # So only 1 row should exist.
        assert row["c"] == 1
    gm2.close()

def test_nodes_edges_survive_migration(temp_dir, fake_clock):
    db_path = str(temp_dir / "graph.db")
    # Simulate an older schema manually
    conn = sqlite3.connect(db_path)
    conn.execute(
        '''
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
        '''
    )
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL,
            applied_at REAL NOT NULL
        );
        '''
    )
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (1, 0.0)")
    conn.execute("INSERT INTO nodes (id, kind, label, created_at, last_used) VALUES ('n1', 'test', 'l', 0, 0)")
    conn.commit()
    conn.close()
    
    # Now init GraphMemory to trigger migration
    gm = GraphMemory(db_path, clock=fake_clock)
    
    with gm._get_connection() as conn:
        row = conn.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
        assert row["v"] == 2
        
        n_row = conn.execute("SELECT * FROM nodes WHERE id = 'n1'").fetchone()
        assert n_row is not None
        assert n_row["kind"] == "test"
    gm.close()
