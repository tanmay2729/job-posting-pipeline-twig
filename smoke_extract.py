"""Try the LLM on 1 Greenhouse, 1 Lever and 1 HN posting and show the result + cost."""
import logging
import os

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from pipeline.extract import Extractor
from pipeline.fetch import Fetcher
from pipeline.filter import keep_ats_posting, keep_hn_comment
from pipeline.sources import fetch_greenhouse, fetch_hn, fetch_lever

f = Fetcher()
gh = [p for p in fetch_greenhouse(f) if keep_ats_posting(p.title, p.location)]
lv = [p for p in fetch_lever(f) if keep_ats_posting(p.title, p.location)]
hn = [p for p in fetch_hn(f, thread_id=os.getenv("HN_THREAD_ID"), max_pages=1) if keep_hn_comment(p.text)]

ex = Extractor()
for raw in [gh[0], lv[0], hn[0]]:
    print("\n=====", raw.source, "|", raw.title, "|", raw.url)
    for c in ex.extract(raw):
        print(c.data.model_dump_json(indent=1, exclude_none=True))
u = ex.usage
print(f"\nmodel={ex.model} tokens in/out={u.prompt_tokens}/{u.completion_tokens} "
      f"cost=${u.cost(ex.model):.5f} per posting=${u.per_posting(ex.model):.5f}")
