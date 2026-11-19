"""SQLite storage: idempotent upserts, change history, closing of vanished postings."""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .models import Posting

COLUMNS = [
    "posting_id", "company", "title", "seniority", "city", "country", "remote_policy",
    "salary_min", "salary_max", "currency", "employment_type", "skills",
    "visa_sponsorship", "posted_date", "status", "source_urls", "evidence",
    "first_seen", "last_seen",
]
JSON_COLUMNS = {"skills", "source_urls", "evidence"}
# changes to these fields are written to the history table
TRACKED = ["title", "seniority", "city", "country", "remote_policy", "salary_min",
           "salary_max", "currency", "employment_type", "visa_sponsorship", "status"]
SOURCE_BY_PREFIX = {"gh": "greenhouse", "le": "lever", "hn": "hn"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS postings (
    posting_id TEXT PRIMARY KEY, company TEXT, title TEXT, seniority TEXT, city TEXT,
    country TEXT, remote_policy TEXT, salary_min REAL, salary_max REAL, currency TEXT,
    employment_type TEXT, skills TEXT, visa_sponsorship INTEGER, posted_date TEXT,
    status TEXT, source_urls TEXT, evidence TEXT, first_seen TEXT, last_seen TEXT
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, posting_id TEXT, changed_at TEXT,
    field TEXT, old_value TEXT, new_value TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT,
    new INTEGER, updated INTEGER, closed INTEGER, reopened INTEGER, total_open INTEGER
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str = "data/postings.db"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def _log(self, pid: str, field: str, old, new, now: str) -> None:
        self.db.execute(
            "INSERT INTO history (posting_id, changed_at, field, old_value, new_value) "
            "VALUES (?,?,?,?,?)",
            (pid, now, field, None if old is None else str(old), None if new is None else str(new)),
        )

    @staticmethod
    def _row(p: Posting, now: str) -> dict:
        row = p.model_dump(mode="json")
        for c in JSON_COLUMNS:
            row[c] = json.dumps(row[c])
        row["first_seen"] = row["last_seen"] = now
        return {c: row[c] for c in COLUMNS}

    def sync(self, postings: Iterable[Posting], complete_sources: set[str],
             now: Optional[str] = None) -> dict:
        """Bring the DB in line with this run's postings.

        complete_sources: sources that were scraped successfully this run. A posting
        is only marked closed if its source was complete, so a failed scrape can
        never wipe out data.
        """
        now = now or _now()
        stats: Counter = Counter()
        seen: set[str] = set()
        for p in postings:
            row = self._row(p, now)
            pid = p.posting_id
            seen.add(pid)
            cur = self.db.execute("SELECT * FROM postings WHERE posting_id=?", (pid,)).fetchone()
            if cur is None:
                self.db.execute(
                    f"INSERT INTO postings ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})",
                    [row[c] for c in COLUMNS],
                )
                stats["new"] += 1
                continue
            changes = {f: (cur[f], row[f]) for f in TRACKED if cur[f] != row[f]}
            for f, (old, new) in changes.items():
                self._log(pid, f, old, new, now)
            if "status" in changes:
                stats["reopened"] += 1
            if any(f != "status" for f in changes):
                stats["updated"] += 1
            cols = [c for c in COLUMNS if c not in ("posting_id", "first_seen")]
            self.db.execute(
                f"UPDATE postings SET {','.join(c + '=?' for c in cols)} WHERE posting_id=?",
                [row[c] for c in cols] + [pid],
            )
        for r in self.db.execute("SELECT posting_id FROM postings WHERE status='open'").fetchall():
            pid = r["posting_id"]
            if pid not in seen and SOURCE_BY_PREFIX.get(pid[:2]) in complete_sources:
                self.db.execute("UPDATE postings SET status='closed' WHERE posting_id=?", (pid,))
                self._log(pid, "status", "open", "closed", now)
                stats["closed"] += 1
        total_open = self.db.execute("SELECT COUNT(*) FROM postings WHERE status='open'").fetchone()[0]
        self.db.execute(
            "INSERT INTO runs (started_at,new,updated,closed,reopened,total_open) VALUES (?,?,?,?,?,?)",
            (now, stats["new"], stats["updated"], stats["closed"], stats["reopened"], total_open),
        )
        self.db.commit()
        result = {k: stats[k] for k in ("new", "updated", "closed", "reopened")}
        result["total_open"] = total_open
        return result

    def export_csv(self, path: str = "data/postings.csv") -> int:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = self.db.execute("SELECT * FROM postings ORDER BY company, title").fetchall()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for r in rows:
                out = []
                for c in COLUMNS:
                    v = r[c]
                    if c == "skills":
                        v = "; ".join(json.loads(v or "[]"))
                    elif c == "source_urls":
                        v = " | ".join(json.loads(v or "[]"))
                    out.append(v)
                w.writerow(out)
        return len(rows)

    def count(self, table: str = "postings") -> int:
        return self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
