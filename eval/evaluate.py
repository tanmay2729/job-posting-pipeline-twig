"""Compare the pipeline's output with your hand labels.

    python eval/evaluate.py

Prints field-level accuracy, hallucinations (value filled in where the true label
is null) and LLM cost per posting; also writes eval/results.md for the README.
"""
import json
import sqlite3
import sys

sys.path.insert(0, ".")
from pipeline.dedup import core_title, norm_company  # noqa: E402

FIELDS = ["company", "title", "seniority", "city", "country", "remote_policy",
          "salary_min", "salary_max", "currency", "employment_type", "visa_sponsorship"]


def norm(field, v):
    if v is None:
        return None
    if field == "company":
        return norm_company(str(v))
    if field == "title":
        return core_title(str(v))
    if field == "visa_sponsorship":
        return bool(v)
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else float(v)
    return str(v).strip().casefold()


def evaluate(labels, rows):
    """labels: list of dicts, rows: {posting_id: row dict}. Returns the stats."""
    stats = {f: {"ok": 0, "n": 0, "halluc": 0} for f in FIELDS}
    by_src = {"HN": [0, 0], "ATS": [0, 0]}
    missing = []
    for lab in labels:
        row = rows.get(lab["posting_id"])
        if row is None:
            missing.append(lab["posting_id"])
            continue
        src = "HN" if lab["posting_id"].startswith("hn_") else "ATS"
        for f in FIELDS:
            gold, pred = norm(f, lab.get(f)), norm(f, row[f])
            ok = gold == pred
            stats[f]["n"] += 1
            stats[f]["ok"] += ok
            stats[f]["halluc"] += (gold is None and pred is not None)
            by_src[src][0] += ok
            by_src[src][1] += 1
    return stats, by_src, missing


def main():
    labels = json.load(open("eval/labels.json", encoding="utf-8"))
    db = sqlite3.connect("data/postings.db")
    db.row_factory = sqlite3.Row
    rows = {r["posting_id"]: dict(r) for r in db.execute("SELECT * FROM postings")}
    stats, by_src, missing = evaluate(labels, rows)

    lines = ["| field | accuracy | correct/total | hallucinated |", "|---|---|---|---|"]
    tot_ok = tot_n = tot_h = 0
    for f, s in stats.items():
        if s["n"]:
            lines.append(f"| {f} | {s['ok'] / s['n']:.0%} | {s['ok']}/{s['n']} | {s['halluc']} |")
        tot_ok, tot_n, tot_h = tot_ok + s["ok"], tot_n + s["n"], tot_h + s["halluc"]
    lines.append(f"| **overall** | **{tot_ok / tot_n:.0%}** | {tot_ok}/{tot_n} | {tot_h} |")
    for k, (ok, n) in by_src.items():
        if n:
            lines.append(f"\n{k} postings: {ok / n:.0%} field accuracy ({ok}/{n})")
    try:
        run = json.load(open("data/last_run.json"))
        lines.append(f"\nLLM: {run['model']}, {run['postings_extracted']} postings, "
                     f"${run['cost_usd']} total, ${run['cost_per_posting_usd']} per posting")
    except FileNotFoundError:
        pass
    if missing:
        lines.append(f"\nWARNING: labeled postings not in DB: {missing}")
    out = "\n".join(lines)
    print(out)
    open("eval/results.md", "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
