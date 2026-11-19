"""Data shapes used across the pipeline.

Three layers:
  RawPosting         -> what a source (Greenhouse/Lever/HN) gives us, before the LLM
  ExtractedPosting   -> what the LLM must return (validated by Pydantic)
  Posting            -> final canonical record stored in the DB
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Seniority(str, Enum):
    junior = "junior"
    mid = "mid"
    senior = "senior"
    staff_plus = "staff_plus"


class RemotePolicy(str, Enum):
    onsite = "onsite"
    hybrid = "hybrid"
    remote = "remote"


class EmploymentType(str, Enum):
    full_time = "full_time"
    contract = "contract"
    intern = "intern"


class Status(str, Enum):
    open = "open"
    closed = "closed"


@dataclass
class RawPosting:
    """Common shape every source converts into."""
    source: str            # "greenhouse" | "lever" | "hn"
    company: Optional[str]  # known for ATS sources, None for HN (LLM extracts it)
    title: Optional[str]    # None for HN comments (they have no title)
    location: Optional[str]
    text: str               # clean plain text description (NOT raw HTML)
    url: str
    external_id: str        # id from the source, used for stable posting ids
    posted_date: Optional[str] = None


class ExtractedPosting(BaseModel):
    """LLM output. Rule: if the text doesn't state it, it MUST be null."""
    company: Optional[str] = None
    title: str
    seniority: Optional[Seniority] = None
    city: Optional[str] = None
    country: Optional[str] = None
    remote_policy: Optional[RemotePolicy] = None
    salary_min: Optional[float] = None   # annualized
    salary_max: Optional[float] = None
    currency: Optional[str] = None
    employment_type: Optional[EmploymentType] = None
    skills: list[str] = Field(default_factory=list)
    visa_sponsorship: Optional[bool] = None
    posted_date: Optional[date] = None
    # field name -> exact snippet from the source text that supports it
    evidence: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_salary(self):
        if self.salary_min and self.salary_max and self.salary_min > self.salary_max:
            self.salary_min, self.salary_max = self.salary_max, self.salary_min
        return self

    def verify_evidence(self, source_text: str) -> "ExtractedPosting":
        """Anti-hallucination check.

        Any evidence snippet that is not literally in the source text is
        dropped, and its field is nulled (we don't trust unsupported claims).
        """
        haystack = " ".join(source_text.lower().split())
        guarded = [
            "salary_min", "salary_max", "currency", "visa_sponsorship",
            "posted_date", "city", "country",
        ]
        for field in guarded:
            if getattr(self, field) is None:
                continue
            snippet = " ".join(self.evidence.get(field, "").lower().split())
            if not snippet or snippet not in haystack:
                setattr(self, field, None)
                self.evidence.pop(field, None)
        return self


class ExtractionResult(BaseModel):
    """One HN comment can contain several roles, so the LLM returns a list."""
    postings: list[ExtractedPosting] = Field(default_factory=list)


class Posting(BaseModel):
    """Final canonical record (one per real-world job, merged across sources)."""
    posting_id: str
    company: str
    title: str
    seniority: Optional[Seniority] = None
    city: Optional[str] = None
    country: Optional[str] = None
    remote_policy: Optional[RemotePolicy] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    currency: Optional[str] = None
    employment_type: Optional[EmploymentType] = None
    skills: list[str] = Field(default_factory=list)
    visa_sponsorship: Optional[bool] = None
    posted_date: Optional[date] = None
    status: Status = Status.open
    source_urls: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
