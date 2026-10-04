from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class SelectedStudent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    branch: str | None = None
    degree: str | None = None
    batch: str | None = None
    role: str | None = None


class SelectionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    company: str | None = None
    selection_date: str | None = None
    batch: str | None = None
    opportunity_type: str | None = None
    role: str | None = None
    job_category: str | None = None

    selected_count: int | None = Field(default=None, ge=0)
    selected_students: list[SelectedStudent] = Field(default_factory=list)

    eligible_degree: str | None = None
    eligible_branches: list[str] = Field(default_factory=list)
    eligible_batches: list[str] = Field(default_factory=list)

    minimum_cgpa: float | None = None
    maximum_backlogs: int | None = None
    minimum_10th_percentage: float | None = None
    minimum_12th_percentage: float | None = None

    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)

    ctc_lpa: float | None = None
    base_salary_lpa: float | None = None
    stipend_monthly: float | None = None

    location: str | None = None
    city: str | None = None
    state: str | None = None
    work_mode: str | None = None

    selection_process: list[str] = Field(default_factory=list)
    aptitude_test: bool | None = None
    coding_test: bool | None = None
    technical_interview: bool | None = None
    hr_interview: bool | None = None
    group_discussion: bool | None = None

    has_bond: bool | None = None
    bond_duration_months: float | None = None

    source_subject: str | None = None
    source_message_id: str | None = None


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_selection_email: bool
    records: list[SelectionRecord] = Field(default_factory=list)
