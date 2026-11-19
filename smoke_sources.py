"""Quick check: run every source and print counts before/after filtering."""
import logging
import os

from dotenv import load_dotenv

from pipeline.fetch import Fetcher
from pipeline.filter import keep_ats_posting, keep_hn_comment
from pipeline.sources import fetch_greenhouse, fetch_hn, fetch_lever

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

f = Fetcher()
gh = fetch_greenhouse(f)
lv = fetch_lever(f)
print("greenhouse:", len(gh), "->", sum(keep_ats_posting(p.title, p.location) for p in gh))
print("lever:", len(lv), "->", sum(keep_ats_posting(p.title, p.location) for p in lv))

# HN is slow if robots.txt sets a crawl delay, so test it on 2 pages first
hn = fetch_hn(f, thread_id=os.getenv("HN_THREAD_ID"), max_pages=2)
print("hn (2 pages):", len(hn), "->", sum(keep_hn_comment(p.text) for p in hn))
if gh:
    print("\nSAMPLE:", gh[0].title, "|", gh[0].location, "\n", gh[0].text[:300])
