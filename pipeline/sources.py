"""Source adapters: turn each source's format into a list of RawPosting.

Each adapter returns the SAME shape (RawPosting) with clean plain text, so
everything after this point doesn't care where a posting came from.
"""
from __future__ import annotations

import html as html_lib
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .fetch import Fetcher
from .models import RawPosting

log = logging.getLogger(__name__)

GREENHOUSE_TOKENS = ["stripe", "airbnb", "databricks", "figma"]
LEVER_SITES = ["palantir", "spotify"]
HN_BASE = "https://news.ycombinator.com/"
HN_SUBMITTED = HN_BASE + "submitted?id=whoishiring"


# ---------------------------------------------------------------- helpers
def html_to_text(html: str, escaped: bool = False, links_as_urls: bool = False) -> str:
    """HTML -> clean plain text.

    escaped=True      Greenhouse sends HTML as an escaped string (&lt;p&gt;...), so
                      we unescape once first.
    links_as_urls     HN shortens link text ("https://foo.com/care..."), so we
                      swap each <a> for its real href.
    """
    if not html:
        return ""
    if escaped:
        html = html_lib.unescape(html)
    soup = BeautifulSoup(html, "html.parser")
    if links_as_urls:
        for a in soup.find_all("a", href=True):
            a.replace_with(a["href"])
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _ms_to_date(ms) -> Optional[str]:
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return None


# -------------------------------------------------------------- greenhouse
def fetch_greenhouse(
    fetcher: Fetcher, tokens=GREENHOUSE_TOKENS, refresh: bool = False
) -> list[RawPosting]:
    out: list[RawPosting] = []
    for token in tokens:
        url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
        res = fetcher.get(url, refresh=refresh)
        if res is None or not res.is_json:
            log.error("Greenhouse %s: no usable response, skipping", token)
            continue
        jobs = res.json().get("jobs", [])
        if not jobs:
            log.warning("Greenhouse %s returned 0 jobs: swap the token, note in README", token)
        for j in jobs:
            posted = j.get("first_published") or j.get("updated_at")
            out.append(
                RawPosting(
                    source="greenhouse",
                    company=j.get("company_name") or token.title(),
                    title=j.get("title"),
                    location=(j.get("location") or {}).get("name"),
                    text=html_to_text(j.get("content", ""), escaped=True),
                    url=j.get("absolute_url", url),
                    external_id=f"{token}:{j.get('id')}",
                    posted_date=posted[:10] if posted else None,
                )
            )
        log.info("Greenhouse %s: %d jobs", token, len(jobs))
    return out


# ------------------------------------------------------------------- lever
def fetch_lever(
    fetcher: Fetcher, sites=LEVER_SITES, refresh: bool = False
) -> list[RawPosting]:
    out: list[RawPosting] = []
    for site in sites:
        url = f"https://api.lever.co/v0/postings/{site}?mode=json"
        res = fetcher.get(url, refresh=refresh)
        if res is None:
            log.error("Lever %s: fetch failed, skipping", site)
            continue
        if not res.is_json:  # the assignment warns this endpoint sometimes returns HTML
            log.error("Lever %s returned %s, not JSON; skipping", site, res.content_type)
            continue
        postings = res.json()
        if not isinstance(postings, list) or not postings:
            log.warning("Lever %s returned no postings: swap the site, note in README", site)
            continue
        for p in postings:
            cats = p.get("categories") or {}
            parts = [p.get("descriptionPlain") or html_to_text(p.get("description", ""))]
            for block in p.get("lists", []) or []:
                parts.append(f"{block.get('text', '')}\n{html_to_text(block.get('content', ''))}")
            parts.append(p.get("additionalPlain") or html_to_text(p.get("additional", "")))
            # structured fields that exist outside the text: append as text so
            # the LLM can use them AND evidence snippets can point at them
            if p.get("workplaceType"):
                parts.append(f"Workplace type: {p['workplaceType']}")
            sal = p.get("salaryRange")
            if isinstance(sal, dict) and sal.get("min") is not None:
                parts.append(
                    f"Salary range: {sal.get('min')} - {sal.get('max')} "
                    f"{sal.get('currency', '')} per {sal.get('interval', '')}"
                )
            location = cats.get("location") or ", ".join(cats.get("allLocations") or []) or None
            out.append(
                RawPosting(
                    source="lever",
                    company=site.title(),
                    title=p.get("text"),
                    location=location,
                    text="\n\n".join(x for x in parts if x).strip(),
                    url=p.get("hostedUrl", url),
                    external_id=f"{site}:{p.get('id')}",
                    posted_date=_ms_to_date(p.get("createdAt")),
                )
            )
        log.info("Lever %s: %d postings", site, len(postings))
    return out


# ------------------------------------------------------------ hacker news
def find_latest_hn_thread(fetcher: Fetcher, refresh: bool = False) -> Optional[str]:
    """Find the newest 'Ask HN: Who is hiring?' thread on the whoishiring page."""
    res = fetcher.get(HN_SUBMITTED, refresh=refresh)
    if res is None:
        return None
    soup = BeautifulSoup(res.text, "html.parser")
    for a in soup.select("span.titleline > a"):
        if re.match(r"Ask HN: Who is hiring\?", a.get_text(), re.I):
            log.info("Found HN thread: %s", a.get_text())
            return urljoin(HN_BASE, a["href"])
    return None


def parse_hn_page(page_html: str, base_url: str) -> tuple[list[RawPosting], Optional[str]]:
    """Parse one HN item page -> (top-level comments, URL of next page or None)."""
    soup = BeautifulSoup(page_html, "html.parser")
    out: list[RawPosting] = []
    for row in soup.select("tr.athing.comtr"):
        ind = row.select_one("td.ind")
        indent = ind.get("indent") if ind else None
        if indent is None:
            img = ind.find("img") if ind else None
            indent = str(int(img["width"]) // 40) if img and img.get("width") else "0"
        if indent != "0":  # replies to postings, not postings
            continue
        body = row.select_one("div.commtext, span.commtext")
        if body is None:  # deleted / flagged comment
            continue
        cid = row.get("id")
        age = row.select_one("span.age")
        posted = age.get("title", "")[:10] if age else None
        out.append(
            RawPosting(
                source="hn",
                company=None,
                title=None,
                location=None,
                text=html_to_text(str(body), links_as_urls=True),
                url=f"{HN_BASE}item?id={cid}",
                external_id=str(cid),
                posted_date=posted or None,
            )
        )
    more = soup.select_one("a.morelink")
    return out, (urljoin(base_url, more["href"]) if more else None)


def fetch_hn(
    fetcher: Fetcher,
    thread_id: Optional[str] = None,
    max_pages: int = 25,
    refresh: bool = False,
) -> list[RawPosting]:
    """Scrape the HTML pages of the current 'Who is hiring?' thread.

    thread_id lets you skip discovery (useful if robots.txt blocks the
    /submitted page: open the thread in a browser and copy the id from the URL).
    """
    url = f"{HN_BASE}item?id={thread_id}" if thread_id else find_latest_hn_thread(fetcher, refresh)
    if not url:
        log.error("HN: could not find the thread. Set HN_THREAD_ID and retry.")
        return []
    out: list[RawPosting] = []
    for page in range(1, max_pages + 1):
        res = fetcher.get(url, refresh=refresh)
        if res is None:
            log.error("HN page %d failed, stopping with what we have", page)
            break
        items, url = parse_hn_page(res.text, res.url)
        out.extend(items)
        log.info("HN page %d: %d top-level comments", page, len(items))
        if not url:
            break
    return out
