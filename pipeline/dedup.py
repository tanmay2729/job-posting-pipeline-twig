"""Cross-source dedup: merge the same real-world job seen on several sources.

Matching rules (all must hold):
  * same company after normalization ("Stripe, Inc." == "Stripe")
  * postings come from DIFFERENT sources (two Greenhouse postings with the same
    title are different jobs: different teams/cities; we never merge those)
  * core titles are similar: seniority words and team suffixes are stripped,
    then fuzzy matched ("Sr. Data Engineer" ~ "Senior Data Engineer, Platform")
  * seniority doesn't conflict (None is compatible with anything)
  * cities don't conflict (None is compatible with anything)

The ATS record is the "primary" (more reliable); HN only fills in gaps.
"""
from __future__ import annotations

import hashlib
import re

from rapidfuzz import fuzz

from .extract import Candidate
from .models import Posting, Status

SOURCE_PRIORITY = {"greenhouse": 0, "lever": 0, "hn": 1}
ID_PREFIX = {"greenhouse": "gh", "lever": "le", "hn": "hn"}
TITLE_THRESHOLD = 85

_COMPANY_SUFFIX = re.compile(r"\b(inc|llc|ltd|corp|corporation|co|company|gmbh|plc)\b")
_LEVEL_WORDS = {"senior", "sr", "staff", "principal", "lead", "junior", "jr",
                "associate", "ii", "iii", "iv"}
_SYNONYMS = {"ml": "machine learning", "engineering": "engineer", "engineers": "engineer"}


def norm_company(name: str) -> str:
    n = re.sub(r"\(.*?\)", " ", name.lower())
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = _COMPANY_SUFFIX.sub(" ", n)
    return " ".join(n.split())


def core_title(title: str) -> str:
    """'Senior Data Engineer, Platform' -> 'data engineer'."""
    t = re.split(r",| - | – | — |:|\(|\|", title.lower())[0]
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    words = [_SYNONYMS.get(w, w) for w in t.split() if w not in _LEVEL_WORDS]
    return " ".join(words)


def _same_job(a: Candidate, b: Candidate) -> float:
    """0 = not the same job, otherwise a match score (higher = better)."""
    if norm_company(a.data.company) != norm_company(b.data.company):
        return 0.0
    sa, sb = a.data.seniority, b.data.seniority
    if sa and sb and sa != sb:
        return 0.0
    bonus = 0.0
    ca, cb = a.data.city, b.data.city
    if ca and cb:
        if ca.strip().casefold() != cb.strip().casefold():
            return 0.0
        bonus = 10.0
    ta, tb = core_title(a.data.title), core_title(b.data.title)
    if not ta or not tb:
        return 0.0
    score = fuzz.token_sort_ratio(ta, tb)
    return score + bonus if score >= TITLE_THRESHOLD else 0.0


def make_posting_id(c: Candidate) -> str:
    key = f"{c.source}|{c.external_id}"
    if c.source == "hn":  # one HN comment can hold several roles
        key += f"|{core_title(c.data.title)}"
    return f"{ID_PREFIX.get(c.source, 'xx')}_{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def _merge(cluster: list[Candidate]) -> Posting:
    primary = cluster[0]
    d = primary.data.model_dump(mode="python")
    evidence = dict(d.pop("evidence"))
    skills = list(d["skills"])
    for m in cluster[1:]:
        o = m.data
        if d["salary_min"] is None and d["salary_max"] is None and (o.salary_min or o.salary_max):
            for k in ("salary_min", "salary_max", "currency"):
                d[k] = getattr(o, k)
                if k in o.evidence:
                    evidence.setdefault(k, o.evidence[k])
        for k in ("seniority", "city", "country", "remote_policy",
                  "employment_type", "visa_sponsorship", "posted_date"):
            if d[k] is None and getattr(o, k) is not None:
                d[k] = getattr(o, k)
                if k in o.evidence:
                    evidence.setdefault(k, o.evidence[k])
        have = {s.lower() for s in skills}
        skills += [s for s in o.skills if s.lower() not in have]
    d["skills"] = skills
    return Posting(
        posting_id=make_posting_id(primary),
        status=Status.open,
        source_urls=list(dict.fromkeys(m.url for m in cluster)),
        evidence=evidence,
        **d,
    )


def dedup(candidates: list[Candidate]) -> list[Posting]:
    ordered = sorted(candidates, key=lambda c: SOURCE_PRIORITY.get(c.source, 9))
    clusters: list[list[Candidate]] = []
    for cand in ordered:
        best, best_score = None, 0.0
        for cl in clusters:
            if any(m.source == cand.source for m in cl):
                continue
            score = max(_same_job(cand, m) for m in cl)
            if score > best_score:
                best, best_score = cl, score
        if best is not None:
            best.append(cand)
        else:
            clusters.append([cand])
    postings = [_merge(cl) for cl in clusters]
    # Two roles in one HN comment can normalize to the same id (same company/title, e.g.
    # different cities). Never let one overwrite the other: add a counter. Candidate order
    # is deterministic (cached extractions, stable sorts), so ids stay stable across runs.
    used: dict[str, int] = {}
    for p in postings:
        n = used.get(p.posting_id, 0)
        used[p.posting_id] = n + 1
        if n:
            p.posting_id = f"{p.posting_id}-{n + 1}"
    return postings
