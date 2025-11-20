"""Prompt repository model for Git-based prompt management."""
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from platform.core.database import Base


class PromptRepository(Base):
    """Git repository for prompt templates."""

    __tablename__ = "prompt_repositories"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    # Repository info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    git_url: Mapped[str] = mapped_column(String(500), nullable=False)
    default_branch: Mapped[str] = mapped_column(String(100), default="main", nullable=False)

    # Authentication (encrypted)
    auth_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # ssh_key, oauth_token, https
    auth_credential: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Sync info
    last_synced_commit: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="prompt_repositories")
    experiments: Mapped[list["Experiment"]] = relationship(
        "Experiment", back_populates="prompt_repository"
    )

    def __repr__(self) -> str:
        return f"<PromptRepository(id={self.id}, name='{self.name}')>"
