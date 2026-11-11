import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from ..validation import reject, canonical, digest, parse_json

DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS schemas(id TEXT NOT NULL,version INTEGER NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(id,version));
CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,title TEXT NOT NULL,schema_id TEXT NOT NULL,owner TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS versions(dataset TEXT NOT NULL,version INTEGER NOT NULL,schema_version INTEGER NOT NULL,
 artifact TEXT NOT NULL,row_count INTEGER NOT NULL,classification TEXT NOT NULL,scopes TEXT NOT NULL,state TEXT NOT NULL,
 payload TEXT NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(dataset,version));
CREATE TABLE IF NOT EXISTS lineage(dataset TEXT NOT NULL,version INTEGER NOT NULL,parent TEXT NOT NULL,parent_version INTEGER NOT NULL,
 PRIMARY KEY(dataset,version,parent,parent_version));
CREATE TABLE IF NOT EXISTS ingestion(id TEXT PRIMARY KEY,digest TEXT NOT NULL,result TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS releases(id TEXT PRIMARY KEY,request TEXT NOT NULL,manifest TEXT NOT NULL,digest TEXT NOT NULL,
 state TEXT NOT NULL,revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS approvals(release TEXT PRIMARY KEY,reviewer TEXT NOT NULL,manifest_digest TEXT NOT NULL,reason TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS exports(id TEXT PRIMARY KEY,release TEXT NOT NULL,target TEXT NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit(sequence INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,entity TEXT NOT NULL,payload TEXT NOT NULL,previous TEXT NOT NULL,digest TEXT NOT NULL);
"""


class CatalogDatabase:
    def __init__(self, path):
        if Path(path).is_symlink():
            reject("path", "catalog database cannot be symbolic link")
        self.connection = sqlite3.connect(
            path, isolation_level=None, timeout=5, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.executescript(DDL)
        self.lock = threading.RLock()
        self.depth = 0
        old = self.one("SELECT value FROM meta WHERE key='version'")
        if old and old["value"] != "1":
            reject("schema", "unsupported catalog database version")
        self.connection.execute("INSERT OR IGNORE INTO meta VALUES('version','1')")

    @contextmanager
    def transaction(self):
        with self.lock:
            level = self.depth
            self.connection.execute(
                "BEGIN IMMEDIATE" if level == 0 else f"SAVEPOINT tx{level}"
            )
            self.depth += 1
            try:
                yield self.connection
            except BaseException:
                if level == 0:
                    self.connection.execute("ROLLBACK")
                else:
                    self.connection.execute(f"ROLLBACK TO tx{level}")
                    self.connection.execute(f"RELEASE tx{level}")
                raise
            else:
                self.connection.execute(
                    "COMMIT" if level == 0 else f"RELEASE tx{level}"
                )
            finally:
                self.depth -= 1

    def rows(self, sql, args=()):
        with self.lock:
            return [dict(row) for row in self.connection.execute(sql, args)]

    def one(self, sql, args=()):
        result = self.rows(sql, args)
        return result[0] if result else None

    def audit(self, kind, entity, payload):
        old = self.one(
            "SELECT sequence,digest FROM audit ORDER BY sequence DESC LIMIT 1"
        )
        previous = old["digest"] if old else "0" * 64
        sequence = old["sequence"] + 1 if old else 1
        content = {
            "sequence": sequence,
            "kind": kind,
            "entity": entity,
            "payload": payload,
            "previous": previous,
        }
        self.connection.execute(
            "INSERT INTO audit VALUES(?,?,?,?,?,?)",
            (sequence, kind, entity, canonical(payload), previous, digest(content)),
        )

    def audit_check(self):
        previous = "0" * 64
        rows = self.rows("SELECT * FROM audit ORDER BY sequence")
        for sequence, row in enumerate(rows, 1):
            content = {
                "sequence": sequence,
                "kind": row["kind"],
                "entity": row["entity"],
                "payload": parse_json(row["payload"]),
                "previous": previous,
            }
            if (
                row["sequence"] != sequence
                or row["previous"] != previous
                or digest(content) != row["digest"]
            ):
                reject("integrity", "catalog audit mismatch", sequence=sequence)
            previous = row["digest"]
        return {"entries": len(rows), "head": previous}

    def close(self):
        if self.depth:
            reject("state", "cannot close active transaction")
        self.connection.close()
