"""Polite HTTP fetching.

Rules from the assignment, all handled here:
  * respect robots.txt
  * at most 1 request per second per domain
  * retry with exponential backoff
  * log failures instead of crashing the run
  * cache raw responses on disk (so re-runs are cheap and debuggable)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

log = logging.getLogger(__name__)

USER_AGENT = "job-pipeline-takehome/0.1 (data engineering assignment)"


@dataclass
class FetchResult:
    url: str
    status: int
    content_type: str
    text: str
    from_cache: bool = False

    def json(self):
        return json.loads(self.text)

    @property
    def is_json(self) -> bool:
        return "json" in self.content_type.lower()


class Fetcher:
    def __init__(
        self,
        cache_dir: str = "cache",
        min_interval: float = 1.0,
        max_retries: int = 3,
        timeout: int = 20,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last_hit: dict[str, float] = {}
        self._intervals: dict[str, float] = {}  # per-domain delay (robots Crawl-delay)
        self._robots: dict[str, Optional[RobotFileParser]] = {}

    # ---------- robots.txt ----------
    def _allowed(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self._robots:
            self._robots[base] = self._load_robots(base)
        rp = self._robots[base]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

    def _load_robots(self, base: str) -> Optional[RobotFileParser]:
        try:
            self._wait(urlparse(base).netloc)
            r = self.session.get(f"{base}/robots.txt", timeout=self.timeout)
            if r.status_code == 200:
                rp = RobotFileParser()
                rp.parse(r.text.splitlines())
                delay = rp.crawl_delay(USER_AGENT)
                if delay:
                    domain = urlparse(base).netloc
                    self._intervals[domain] = max(float(delay), self.min_interval)
                    log.info("Honoring Crawl-delay %ss for %s", delay, domain)
                return rp
            # no robots.txt (404 etc.) -> nothing is disallowed
            return None
        except requests.RequestException as e:
            log.warning("robots.txt unreachable for %s (%s); proceeding", base, e)
            return None

    # ---------- rate limiting ----------
    def _wait(self, domain: str) -> None:
        interval = self._intervals.get(domain, self.min_interval)
        elapsed = time.monotonic() - self._last_hit.get(domain, 0.0)
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_hit[domain] = time.monotonic()

    # ---------- cache ----------
    def _cache_path(self, url: str) -> Path:
        key = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, url: str) -> Optional[FetchResult]:
        path = self._cache_path(url)
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return FetchResult(d["url"], d["status"], d["content_type"], d["text"], True)

    def _write_cache(self, res: FetchResult) -> None:
        self._cache_path(res.url).write_text(
            json.dumps(
                {
                    "url": res.url,
                    "status": res.status,
                    "content_type": res.content_type,
                    "text": res.text,
                    "fetched_at": time.time(),
                }
            ),
            encoding="utf-8",
        )

    # ---------- main entry ----------
    def get(self, url: str, refresh: bool = False) -> Optional[FetchResult]:
        """Return a FetchResult, or None if it failed / is disallowed.

        refresh=False -> use the cached copy if we have one.
        refresh=True  -> hit the network (still writes the cache after).
        Never raises for network problems: failures are logged.
        """
        if not refresh:
            cached = self._read_cache(url)
            if cached:
                return cached

        if not self._allowed(url):
            log.warning("Blocked by robots.txt: %s", url)
            return None

        domain = urlparse(url).netloc
        for attempt in range(self.max_retries):
            try:
                self._wait(domain)
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    res = FetchResult(
                        url, 200, r.headers.get("Content-Type", ""), r.text
                    )
                    self._write_cache(res)
                    return res
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"retryable status {r.status_code}")
                log.error("Giving up on %s (HTTP %s)", url, r.status_code)
                return None
            except requests.RequestException as e:
                backoff = 2 ** attempt
                log.warning(
                    "Attempt %d/%d failed for %s: %s (sleep %ss)",
                    attempt + 1, self.max_retries, url, e, backoff,
                )
                time.sleep(backoff)
        log.error("Failed after %d attempts: %s", self.max_retries, url)
        return None
