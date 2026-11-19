import pytest

from pipeline.models import Posting
from pipeline.store import Store


def post(pid="gh_1", **kw):
    base = dict(posting_id=pid, company="Stripe", title="Data Engineer", city="New York",
                salary_min=150000, salary_max=180000, currency="USD", source_urls=["https://x"])
    base.update(kw)
    return Posting(**base)


@pytest.fixture
def store():
    return Store(":memory:")


def test_rerun_creates_no_duplicates(store):
    store.sync([post()], {"greenhouse"})
    stats = store.sync([post()], {"greenhouse"})
    assert store.count() == 1
    assert stats["new"] == 0 and stats["updated"] == 0
    assert store.count("history") == 0


def test_salary_change_updates_record_and_logs_history(store):
    store.sync([post()], {"greenhouse"})
    stats = store.sync([post(salary_max=200000)], {"greenhouse"})
    assert stats["updated"] == 1 and store.count() == 1
    h = store.db.execute("SELECT field, old_value, new_value FROM history").fetchall()
    assert [(r["field"], r["old_value"], r["new_value"]) for r in h] == [("salary_max", "180000.0", "200000.0")]
    assert store.db.execute("SELECT salary_max FROM postings").fetchone()[0] == 200000


def test_location_change_is_tracked(store):
    store.sync([post()], {"greenhouse"})
    store.sync([post(city="Seattle")], {"greenhouse"})
    fields = [r["field"] for r in store.db.execute("SELECT field FROM history")]
    assert fields == ["city"]


def test_missing_posting_is_closed_then_reopened(store):
    store.sync([post()], {"greenhouse"})
    assert store.sync([], {"greenhouse"})["closed"] == 1
    assert store.db.execute("SELECT status FROM postings").fetchone()[0] == "closed"
    assert store.sync([post()], {"greenhouse"})["reopened"] == 1
    assert store.db.execute("SELECT status FROM postings").fetchone()[0] == "open"


def test_failed_source_never_closes_its_postings(store):
    store.sync([post(), post("hn_1", company="Acme")], {"greenhouse", "hn"})
    stats = store.sync([post("hn_1", company="Acme")], {"hn"})  # greenhouse scrape failed
    assert stats["closed"] == 0
    assert store.db.execute("SELECT COUNT(*) FROM postings WHERE status='open'").fetchone()[0] == 2


def test_first_seen_is_preserved(store):
    store.sync([post()], {"greenhouse"}, now="2026-01-01T00:00:00")
    store.sync([post()], {"greenhouse"}, now="2026-02-01T00:00:00")
    r = store.db.execute("SELECT first_seen, last_seen FROM postings").fetchone()
    assert (r["first_seen"], r["last_seen"]) == ("2026-01-01T00:00:00", "2026-02-01T00:00:00")
