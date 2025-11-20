"""Evaluation run model for experiment evaluations."""
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import JSON, DateTime, Enum as SQLEnum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from platform.core.database import Base


class EvaluationStatus(str, Enum):
    """Evaluation execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationRun(Base):
    """Evaluation execution for an experiment."""

    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id"), nullable=False, index=True
    )
    job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("jobs.id"), nullable=True)

    # Evaluator info
    evaluator_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # w&b_weave, custom_function
    evaluator_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Status
    status: Mapped[EvaluationStatus] = mapped_column(
        SQLEnum(EvaluationStatus), default=EvaluationStatus.PENDING, nullable=False
    )

    # Results
    pass_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    fail_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    average_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Results storage
    results_gcs_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="evaluation_runs"
    )

    def __repr__(self) -> str:
        return f"<EvaluationRun(id={self.id}, type='{self.evaluator_type}', status='{self.status}')>"
