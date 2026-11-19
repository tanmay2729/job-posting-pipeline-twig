"""Pick 20 postings (12 from Hacker News) and write the files you label by hand.

    python eval/make_labeling_sheet.py

Creates:
  eval/to_label.md    the text of each posting (what the LLM saw), numbered
  eval/labels.json    a template: fill in the TRUE value of each field by reading the text
                      (use null when the posting does not state it)
Uses only cached data, no network.
"""
import json
import os
import random
import sqlite3
import sys

sys.path.insert(0, ".")
from dotenv import load_dotenv  # noqa: E402

from pipeline.extract import shrink  # noqa: E402
from pipeline.fetch import Fetcher  # noqa: E402
from pipeline.sources import fetch_greenhouse, fetch_hn, fetch_lever  # noqa: E402

load_dotenv()
FIELDS = ["company", "title", "seniority", "city", "country", "remote_policy",
          "salary_min", "salary_max", "currency", "employment_type", "visa_sponsorship"]

db = sqlite3.connect("data/postings.db")
db.row_factory = sqlite3.Row
rows = db.execute("SELECT * FROM postings WHERE status='open'").fetchall()
hn = [r for r in rows if r["posting_id"].startswith("hn_")]
ats = [r for r in rows if not r["posting_id"].startswith("hn_")]
rnd = random.Random(42)  # fixed seed -> same sample every time
n_hn = min(12, len(hn))
sample = rnd.sample(hn, n_hn) + rnd.sample(ats, min(20 - n_hn, len(ats)))

f = Fetcher()
raws = fetch_greenhouse(f) + fetch_lever(f) + fetch_hn(f, thread_id=os.getenv("HN_THREAD_ID"))
by_url = {r.url: r for r in raws}

md = ["# Label these 20 postings\n",
      "Edit eval/labels.json. For each posting fill the TRUE value of every field from the text below.",
      "Use null when the text does not state it. Allowed values:",
      "- seniority: junior | mid | senior | staff_plus | null",
      "- remote_policy: onsite | hybrid | remote | null",
      "- employment_type: full_time | contract | intern | null",
      "- salary_*: ANNUAL number or null; currency e.g. USD; visa_sponsorship: true | false | null",
      "- country: United States (or null); city: as written\n"]
template = []
for i, r in enumerate(sample, 1):
    raw = by_url.get(json.loads(r["source_urls"])[0])
    text = shrink(raw.text) if raw else "(text not found in cache)"
    md.append(f"\n---\n## {i}. {r['posting_id']}  ({'HN' if r['posting_id'].startswith('hn_') else 'ATS'})")
    md.append(f"Role to label: **{r['title']}** at **{r['company']}**\n")
    md.append(f"Source: {json.loads(r['source_urls'])[0]}\n")
    md.append("```\n" + text + "\n```")
    template.append({"posting_id": r["posting_id"], "role": r["title"], **{k: None for k in FIELDS}})

open("eval/to_label.md", "w", encoding="utf-8").write("\n".join(md))
json.dump(template, open("eval/labels.json", "w", encoding="utf-8"), indent=1)
print(f"Wrote eval/to_label.md and eval/labels.json  ({n_hn} HN + {len(sample) - n_hn} ATS)")
