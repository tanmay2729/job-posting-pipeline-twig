# Job Postings Pipeline: Data & ML Engineering Roles at US Tech Companies

A small Python pipeline that scrapes job postings from three sources, uses an LLM to extract structured records, deduplicates them across sources, and keeps them current across re-runs.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set GROQ_API_KEY=YOUR_API_KEY and HN_THREAD_ID
python -m pipeline.run    
pytest                        # dedup + change-history tests
python eval/evaluate.py       # accuracy vs. hand labels -> eval/results.md
```

Output goes to `data/postings.db` (SQLite) and `data/postings.csv`.

## Architecture

```
sources.py  ->  fetch.py (cache, robots, rate limit)  ->  extract.py (shrink + LLM + Pydantic)  ->  dedup.py  ->  SQLite (postings / history / runs)  ->  CSV
```

| Stage | What it does |
|---|---|
| **Sources** | Greenhouse JSON (`stripe`, `airbnb`, `databricks`, `figma`), Lever JSON (`palantir`, plus `spotify`; see note below), and HN "Who is hiring?" HTML pages, following the "More" pagination. The HN Firebase API is not used. |
| **Fetch** | Checks robots.txt, allows at most 1 request/sec per domain, and retries with exponential backoff. Raw responses are cached on disk, so re-runs and the eval scripts work offline. A failed page is logged and skipped; it never crashes the run. Lever responses are checked by content type, because the endpoint sometimes returns HTML. |
| **Pre-filter** | Titles are matched against a keyword list (data / ML / analytics / platform) *before* any LLM call. This cuts LLM spend. |
| **Extract** | `shrink()` strips HTML and boilerplate so the LLM never sees a whole page. A cheap model (`openai/gpt-oss-20b` on Groq) returns JSON, which is validated by a Pydantic schema with enums for seniority, remote policy and employment type. Missing fields are `null`, never guessed. Each field stores its source snippet in `evidence`. **Anti-hallucination check:** for salary, currency, city, country, visa and posted date, if the evidence snippet is not literally in the source text, the field is set to `null`. A 429 from Groq is retried with backoff, and a model that hits its rate limit is skipped for 600 seconds. LLM responses are cached, so a re-run only calls the LLM for new or changed postings (7 fresh calls out of 124 on the last run). |
| **Dedup** | Two candidates merge only if all of these hold: same company after normalization (`Stripe, Inc.` = `Stripe`); they come from *different* sources, because two Greenhouse posts with the same title are different jobs; their core titles match fuzzily (rapidfuzz ≥ 85 after stripping level words and team suffixes, so `Sr. Data Engineer` ≈ `Senior Data Engineer, Platform`); and their seniority and city don't conflict (`null` is compatible with anything). The ATS record is primary, and HN only fills gaps. Merged records keep every `source_url`, and skills and evidence are combined. |
| **Persist** | `posting_id` = `sha1(source + external_id)`, plus the core title for HN, since one comment can hold several roles. Same input means same id, so re-runs upsert instead of duplicating rows. A change to any tracked field (title, seniority, location, remote policy, salary, employment type, visa, status) writes an old → new row to `history`. A posting missing from a fresh fetch is marked `closed`, **but only if its source scraped successfully that run**, so a failed scrape can't wipe out data. A closed posting that reappears is reopened. Each run logs new, updated, closed and reopened counts to `runs`. |

**Note on sources:** `plaid` returned no jobs on Lever, so `spotify` (also on Lever) was used instead.

## Results

The current dataset holds 90 open postings: 48 from Greenhouse, 13 from Lever and 29 from HN. Three runs were made. The second run created 0 new rows, which shows re-runs are idempotent.

Accuracy on 20 hand-labeled postings (12 from HN, 8 from ATS feeds); full table in `eval/results.md`:

| Field group | Accuracy |
|---|---|
| Overall | 90% (198/220 fields) |
| Salary, employment type, visa, title | 100% |
| City | 55% (4 hallucinated) |
| Country | 70% |
| HN / ATS | 88% / 93% |

LLM cost (last run, `openai/gpt-oss-20b` on Groq):

| Postings extracted | Tokens (in + out) | Total cost | Cost per posting |
|---|---|---|---|
| 124 | 151,600 + 128,470 | **$0.061** | **$0.00049** |

On average that is about 1.2k input and 1.0k output tokens per posting. The output count is high because gpt-oss is a reasoning model and its reasoning tokens are billed as output.

## Design trade-offs

- **SQLite over Postgres or DuckDB.** There are no extra services to run, and it is easy to inspect. Upserts and the `history` table give idempotency without an orchestrator.
- **Pre-filter before the LLM.** Running cheap regex on titles first means we only pay to extract relevant roles. The downside is that unusual titles can be missed.
- **Fuzzy dedup over exact matching.** This catches pairs like "Sr. Data Engineer" vs "Senior Data Engineer, Platform". The risk is merging two genuinely different roles at the same company, so the threshold is kept conservative.
- **`null` over a guess.** The prompt and schema prefer a missing value to a wrong one, which keeps hallucinated fields low.

## Known limitations and what breaks first

1. **Location fields** (city 55%, country 70%). HN comments list several cities or say "SF/NYC/DC", and the extractor does not normalize these. Some non-US roles (Berlin, Amsterdam/London, Toronto) got through the US filter because `country` came back `null`.
2. **HN multi-role comments.** One comment can describe many roles that share one header (salary, location). Attributing those details to the right role is the weakest step.
3. **Cross-source dedup is barely exercised by real data.** In this sample no posting appeared on both HN and an ATS feed, so the merge logic is covered by unit tests only (`tests/test_dedup.py`). An HN post often names a different team title than the ATS listing, which falls below the 85 threshold. That keeps false merges low but misses some true ones.
4. **Change history** is covered by tests (`tests/test_store.py`). No tracked field changed between the three live runs, so the `history` table is still empty in the shipped DB.
5. **Skills** come back as free-text phrases instead of a clean skill list.

## Scaling to 5,000 sources refreshed weekly

| Concern | Approach |
|---|---|
| **Crawl** | Run a per-domain token-bucket queue with concurrency across domains (asyncio/httpx), keeping 1 request/sec per domain. 5,000 sources finish in well under an hour. |
| **LLM cost** | Hash the shrunk text and **only re-extract postings whose text changed**. Most postings are unchanged week to week. |
| **Cost estimate** | Uses the measured $0.00049 per posting and assumes about 20 postings per source, so roughly 100k postings. With about 15% new or changed each week, that is 15k extractions, about **$7 a week**. Re-extracting all 100k from scratch would cost about **$49**. The title pre-filter and text hashing keep the weekly cost near the lower figure. |
| **Quality** | Re-label a random sample each month and alert when field accuracy drops. Route low-confidence or schema-failing records to a stronger model. |
| **Storage** | Move to Postgres with a `(company_norm, core_title)` index for dedup. Keep raw responses in object storage. |

## AI tools used

- **LLM extraction:** `openai/gpt-oss-20b` via Groq
- **Coding assistance:**Claude for code scaffolding, label drafting and the README
