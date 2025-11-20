"""Provider configuration model for LLM providers."""
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from platform.core.database import Base


class ProviderConfig(Base):
    """Configuration for LLM providers (global or per-user)."""

    __tablename__ = "provider_configs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Provider info
    provider_name: Mapped[str] = mapped_column(
        String(50), nullable=False, unique=True, index=True
    )
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # API configuration
    api_base_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    api_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Rate limits
    requests_per_minute: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_per_minute: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Pricing (per 1K tokens)
    default_input_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    default_output_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Model-specific pricing
    model_pricing: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # {model_name: {input: X, output: Y}}

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<ProviderConfig(provider='{self.provider_name}')>"
