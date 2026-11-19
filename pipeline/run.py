"""Entry point:  python -m pipeline.run   (add --refresh to re-download everything)"""
from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from .dedup import dedup
from .extract import Extractor
from .fetch import Fetcher
from .filter import keep_after_extraction, keep_ats_posting, keep_hn_comment
from .sources import fetch_greenhouse, fetch_hn, fetch_lever
from .store import Store

log = logging.getLogger("pipeline")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore the raw cache and re-download")
    ap.add_argument("--hn-pages", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0, help="debug: only process N postings per source")
    args = ap.parse_args(argv)

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetcher, extractor, store = Fetcher(), Extractor(), Store()

    # 1. scrape + cheap pre-filter (a source that returns nothing is not "complete")
    batches = {
        "greenhouse": [p for p in fetch_greenhouse(fetcher, refresh=args.refresh)
                       if keep_ats_posting(p.title, p.location)],
        "lever": [p for p in fetch_lever(fetcher, refresh=args.refresh)
                  if keep_ats_posting(p.title, p.location)],
        "hn": [p for p in fetch_hn(fetcher, thread_id=os.getenv("HN_THREAD_ID"),
                                   max_pages=args.hn_pages, refresh=args.refresh)
               if keep_hn_comment(p.text)],
    }
    complete = {s for s, raws in batches.items() if raws}
    log.info("After pre-filter: %s", {s: len(r) for s, r in batches.items()})

    # 2. LLM extraction
    candidates = []
    for source, raws in batches.items():
        before = extractor.usage.failures
        for raw in (raws[:args.limit] if args.limit else raws):
            candidates += extractor.extract(raw)
        if extractor.usage.failures > before:
            log.warning("%s: %d extraction failures, will NOT close its missing postings",
                        source, extractor.usage.failures - before)
            complete.discard(source)
    candidates = [c for c in candidates if keep_after_extraction(c.data.title, c.data.country)]

    # 3. dedup across sources, 4. store (idempotent) + CSV
    postings = dedup(candidates)
    stats = store.sync(postings, complete)
    n = store.export_csv()

    u = extractor.usage
    print("\n=== RUN SUMMARY ===")
    print(f"candidates after extraction: {len(candidates)}  ->  unique postings: {len(postings)}")
    print(f"db changes: {stats}")
    print(f"csv rows: {n}")
    print(f"LLM: model={extractor.model} postings={u.calls} fresh_calls={u.fresh_calls} "
          f"tokens={u.prompt_tokens}+{u.completion_tokens} "
          f"cost=${u.cost(extractor.model):.4f} (${u.per_posting(extractor.model):.5f}/posting)")


if __name__ == "__main__":
    main()
