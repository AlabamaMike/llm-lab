"""Experiment model for LLM experiments."""
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from platform.core.database import Base


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""

    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Experiment(Base):
    """LLM experiment configuration and metadata."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    # Basic metadata
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[ExperimentStatus] = mapped_column(
        SQLEnum(ExperimentStatus), default=ExperimentStatus.DRAFT, nullable=False, index=True
    )

    # LLM provider configuration
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # openai, anthropic
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)  # gpt-4, claude-3-opus

    # Model parameters (stored as JSON)
    model_params: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict
    )  # {temperature: 0.7, max_tokens: 1000, ...}

    # Prompt reference
    prompt_repo_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("prompt_repositories.id"), nullable=True
    )
    prompt_file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    prompt_commit_sha: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)

    # Dataset reference
    dataset_gcs_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    dataset_inline: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Evaluation configuration
    evaluation_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # W&B integration
    wandb_run_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Results summary
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="experiments")
    prompt_repository: Mapped[Optional["PromptRepository"]] = relationship(
        "PromptRepository", back_populates="experiments"
    )
    jobs: Mapped[list["Job"]] = relationship(
        "Job", back_populates="experiment", cascade="all, delete-orphan"
    )
    evaluation_runs: Mapped[list["EvaluationRun"]] = relationship(
        "EvaluationRun", back_populates="experiment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name='{self.name}', status='{self.status}')>"
