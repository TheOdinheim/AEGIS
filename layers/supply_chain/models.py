"""
Supply chain verification data models.

Shared Pydantic models used across all four verification stages and the
orchestrator.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class StageResult(BaseModel):
    """Result from a single verification stage."""

    stage: str = Field(description="Stage name (e.g., 'cryptographic_integrity')")
    status: str = Field(description="Stage outcome: 'pass', 'fail', or 'warn'")
    details: dict[str, Any] = Field(default_factory=dict, description="Stage-specific details")
    duration_ms: float = Field(default=0.0, description="Stage execution time in milliseconds")


class ModelVerificationReport(BaseModel):
    """Complete verification report for a model."""

    model_id: str = Field(description="Unique identifier for the model under verification")
    overall_status: str = Field(description="Overall verdict: 'pass', 'fail', or 'warn'")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stages: list[StageResult] = Field(default_factory=list, description="Individual stage results")
    total_duration_ms: float = Field(default=0.0, description="Total pipeline execution time")
    summary: str = Field(default="", description="Human-readable summary of findings")
    risk_score: float = Field(
        default=0.0,
        description="Aggregate risk score (0.0-1.0), weighted across all stages",
    )
    stage_risk_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage risk scores (0.0-1.0)",
    )
    critical_findings: list[str] = Field(
        default_factory=list,
        description="Critical findings requiring immediate attention",
    )
