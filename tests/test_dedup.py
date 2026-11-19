from pipeline.dedup import core_title, dedup, norm_company
from pipeline.extract import Candidate
from pipeline.models import ExtractedPosting


def cand(source, company, title, city=None, seniority=None, ext="1", skills=None, salary=None):
    d = ExtractedPosting(company=company, title=title, city=city, seniority=seniority,
                         skills=skills or [], salary_min=salary, salary_max=salary)
    return Candidate(d, source, f"https://{source}.example/{ext}", ext)


def test_title_normalization():
    assert core_title("Sr. Data Engineer") == core_title("Senior Data Engineer, Platform") == "data engineer"
    assert core_title("ML Engineer") == "machine learning engineer"


def test_company_normalization():
    assert norm_company("Stripe, Inc.") == norm_company("Stripe (YC S09)") == "stripe"


def test_merge_across_sources_keeps_all_links_and_fills_gaps():
    gh = cand("greenhouse", "Stripe", "Data Engineer, Platform", city="New York", skills=["Python"], ext="g1")
    hn = cand("hn", "Stripe, Inc.", "Sr. Data Engineer", city="new york", skills=["python", "Spark"], salary=150000, ext="h1")
    out = dedup([hn, gh])  # order must not matter
    assert len(out) == 1
    p = out[0]
    assert p.posting_id.startswith("gh_")          # ATS is the primary record
    assert len(p.source_urls) == 2
    assert p.salary_min == 150000                  # gap filled from HN
    assert p.skills == ["Python", "Spark"]         # union, no case-duplicates


def test_different_city_is_a_different_job():
    gh = cand("greenhouse", "Stripe", "Data Engineer", city="Seattle", ext="g1")
    hn = cand("hn", "Stripe", "Data Engineer", city="New York", ext="h1")
    assert len(dedup([gh, hn])) == 2


def test_same_source_never_merges():
    a = cand("greenhouse", "Stripe", "Data Engineer", city="New York", ext="g1")
    b = cand("greenhouse", "Stripe", "Data Engineer", city="New York", ext="g2")
    assert len(dedup([a, b])) == 2


def test_conflicting_seniority_or_company_does_not_merge():
    gh = cand("greenhouse", "Stripe", "Data Engineer", seniority="senior", ext="g1")
    assert len(dedup([gh, cand("hn", "Stripe", "Data Engineer", seniority="junior", ext="h1")])) == 2
    assert len(dedup([gh, cand("hn", "Airbnb", "Data Engineer", seniority="senior", ext="h2")])) == 2


def test_ids_are_stable_between_runs():
    cs = [cand("greenhouse", "Stripe", "Data Engineer", ext="g1"), cand("hn", "Acme", "ML Engineer", ext="h1")]
    assert {p.posting_id for p in dedup(cs)} == {p.posting_id for p in dedup(list(reversed(cs)))}


def test_two_roles_in_one_hn_comment_get_distinct_stable_ids():
    a = cand("hn", "Acme", "Data Engineer", city="New York", ext="h1")
    b = cand("hn", "Acme", "Data Engineer", city="London", ext="h1")  # same comment, same title
    first = dedup([a, b])
    second = dedup([a, b])
    assert len(first) == 2 and len({p.posting_id for p in first}) == 2
    assert [p.posting_id for p in first] == [p.posting_id for p in second]
