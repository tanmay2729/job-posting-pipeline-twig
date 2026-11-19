"""Cheap filters that run BEFORE the LLM, so we only pay for relevant postings.

Order of operations: filter by title/location (free) -> LLM (costs money).
After the LLM we re-check the title and country on the cleaned fields.
"""
from __future__ import annotations

import re
from typing import Optional

TITLE_INCLUDE = re.compile(
    r"data engineer|data platform|data infrastructure|analytics engineer|"
    r"machine learning engineer|ml engineer|mlops|ml platform|ml infra|"
    r"\bai engineer|applied ai|"
    r"software engineer.{0,40}(\bdata\b|machine learning|\bml\b)|"
    r"(\bdata\b|\bml\b|machine learning).{0,40}(engineer|infrastructure|platform)",
    re.I,
)
TITLE_EXCLUDE = re.compile(
    r"\bsales\b|recruit|account (exec|manager)|marketing|counsel|designer|"
    r"data center|support engineer|customer|\bintern\b.*\bmarketing",
    re.I,
)

# HN comments have no title, so we just look for relevant keywords in the text
HN_KEYWORDS = re.compile(
    r"data engineer|data platform|data infrastructure|analytics engineer|"
    r"machine learning|\bml\b|mlops|ai engineer|applied ai|data pipeline|\betl\b",
    re.I,
)

US_STATES = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
    "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC"
).split()
US_MARKER = re.compile(
    r"\bunited states\b|\busa\b|\bu\.s\.|\bUS\b|\bnyc\b|new york|san francisco|"
    r"seattle|austin|boston|chicago|los angeles|denver|atlanta|"
    r",\s*(" + "|".join(US_STATES) + r")\b"
)
NON_US_MARKER = re.compile(
    r"india|canada|toronto|vancouver|united kingdom|\buk\b|london|ireland|dublin|"
    r"germany|berlin|munich|france|paris|spain|madrid|barcelona|netherlands|"
    r"amsterdam|poland|warsaw|singapore|japan|tokyo|australia|sydney|brazil|"
    r"mexico|israel|tel aviv|bangalore|bengaluru|hyderabad|emea|apac|europe",
    re.I,
)


def is_target_title(title: Optional[str]) -> bool:
    if not title:
        return False
    return bool(TITLE_INCLUDE.search(title)) and not TITLE_EXCLUDE.search(title)


def us_location(location: Optional[str]) -> Optional[bool]:
    """True = clearly US, False = clearly not US, None = can't tell.

    None means we KEEP the posting for now (e.g. plain "Remote") and let the
    LLM-extracted country decide later.
    """
    if not location:
        return None
    if US_MARKER.search(location):
        return True
    if NON_US_MARKER.search(location):
        return False
    return None


def keep_ats_posting(title: Optional[str], location: Optional[str]) -> bool:
    return is_target_title(title) and us_location(location) is not False


def keep_hn_comment(text: str) -> bool:
    return bool(HN_KEYWORDS.search(text))


def keep_after_extraction(title: str, country: Optional[str]) -> bool:
    """Final check on LLM output: relevant title, and US (or unknown) country."""
    if not is_target_title(title):
        return False
    if country is None:
        return True
    return country.strip().lower() in {"us", "usa", "united states", "u.s."}
