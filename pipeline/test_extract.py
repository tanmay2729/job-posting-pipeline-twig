import json
from types import SimpleNamespace as NS

import pytest

import pipeline.extract as ex
from pipeline.models import RawPosting


class FakeRateLimit(Exception):
    def __init__(self, retry_after):
        self.response = NS(headers={"retry-after": str(retry_after)})


GOOD = json.dumps({"postings": [{"title": "Data Engineer", "company": "Acme"}]})


def fake_client(behaviour):
    def create(**kw):
        out = behaviour(kw["model"])
        return NS(usage=NS(prompt_tokens=1000, completion_tokens=100),
                  choices=[NS(message=NS(content=out))])
    return NS(chat=NS(completions=NS(create=create)))


def raw(i):
    return RawPosting("hn", None, None, None, f"Acme | Data Engineer | posting {i}", f"u{i}", str(i))


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(ex, "_RETRY_ERRORS", (FakeRateLimit,))
    monkeypatch.setattr(ex.time, "sleep", lambda s: None)


def test_falls_back_when_primary_daily_limit_hit(tmp_path):
    def behaviour(model):
        if model == "primary":
            raise FakeRateLimit(5000)  # daily limit: huge retry-after
        return GOOD
    e = ex.Extractor(model="primary", cache_dir=str(tmp_path), client=fake_client(behaviour),
                     fallback_models=["openai/gpt-oss-120b"])
    assert len(e.extract(raw(1))) == 1
    assert len(e.extract(raw(2))) == 1          # primary is now skipped without calling it
    assert e.usage.fresh_calls == 2 and e.usage.failures == 0
    assert e.usage.cost_usd == pytest.approx(2 * (1000 * 0.15 + 100 * 0.60) / 1e6)


def test_cache_is_shared_across_models(tmp_path):
    e1 = ex.Extractor(model="primary", cache_dir=str(tmp_path), client=fake_client(lambda m: GOOD),
                      fallback_models=["backup"])
    e1.extract(raw(1))
    def boom(m):
        raise AssertionError("must not call the API: result is cached")
    e2 = ex.Extractor(model="backup", cache_dir=str(tmp_path), client=fake_client(boom),
                      fallback_models=["primary"])
    assert len(e2.extract(raw(1))) == 1 and e2.usage.fresh_calls == 0


def test_circuit_breaker_stops_api_calls_after_repeated_failures(tmp_path):
    calls = []
    def behaviour(model):
        calls.append(model)
        raise FakeRateLimit(5000)
    e = ex.Extractor(model="primary", cache_dir=str(tmp_path), client=fake_client(behaviour),
                     fallback_models=[])
    for i in range(10):
        assert e.extract(raw(i)) == []
    assert e.usage.failures == 10 and len(calls) == 1  # one call marked the model dead; rest were instant
