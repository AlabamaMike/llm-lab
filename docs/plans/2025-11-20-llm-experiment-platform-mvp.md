# LLM Experiment Platform MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundational MVP of the LLM experiment platform with core experiment execution, OpenAI integration, W&B tracking, and basic Streamlit UI.

**Architecture:** Python modular monolith with FastAPI for REST APIs, Streamlit for web UI, Celery for async job execution, PostgreSQL for data persistence, Redis for job queue, and W&B SDK for experiment tracking. Deployed on GCP using Cloud Run, Cloud SQL, and Memorystore.

**Tech Stack:** Python 3.11+, FastAPI, Streamlit, Celery, SQLAlchemy, PostgreSQL, Redis, W&B SDK, OpenAI SDK, Docker, Terraform (GCP)

---

## Task 1: Project Structure and Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `.gitignore`

**Step 1: Create project structure**

```bash
mkdir -p platform/{api,ui,services,workers,models,core}
mkdir -p platform/api/routes
mkdir -p platform/ui/{pages,components}
mkdir -p platform/workers/evaluators
mkdir -p tests/{unit,integration}
mkdir -p infrastructure/terraform
mkdir -p scripts
touch platform/__init__.py
touch platform/api/__init__.py
touch platform/ui/__init__.py
touch platform/services/__init__.py
touch platform/workers/__init__.py
touch platform/models/__init__.py
touch platform/core/__init__.py
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "llm-experiment-platform"
version = "0.1.0"
description = "LLM experiment platform with LLMOps capabilities"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "streamlit>=1.28.0",
    "celery[redis]>=5.3.0",
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "alembic>=1.12.0",
    "redis>=5.0.0",
    "wandb>=0.16.0",
    "openai>=1.3.0",
    "anthropic>=0.7.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "python-multipart>=0.0.6",
    "jinja2>=3.1.0",
    "gitpython>=3.1.0",
    "google-cloud-storage>=2.10.0",
    "google-cloud-secret-manager>=2.16.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.25.0",
    "black>=23.11.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
select = ["E", "F", "I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 3: Create .env.example**

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/llm_experiments
DATABASE_POOL_SIZE=5

# Redis
REDIS_URL=redis://localhost:6379/0

# GCP
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_ARTIFACTS=llm-platform-artifacts
GCS_BUCKET_DATASETS=llm-platform-datasets

# API
API_SECRET_KEY=your-secret-key-change-in-production
API_ACCESS_TOKEN_EXPIRE_MINUTES=15
API_REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS
CORS_ORIGINS=http://localhost:8501

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# W&B (defaults, users override in settings)
WANDB_PROJECT_PREFIX=llm-experiments

# Feature Flags
ENABLE_ANTHROPIC=true
ENABLE_CUSTOM_EVALUATORS=true

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
```

**Step 4: Create Dockerfile**

```dockerfile
# Multi-stage build for optimal image size
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy application code
COPY platform/ ./platform/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (override in docker-compose or Cloud Run)
CMD ["uvicorn", "platform.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Step 5: Create .dockerignore**

```
.git
.env
.env.*
!.env.example
__pycache__
*.pyc
*.pyo
*.pyd
.pytest_cache
.coverage
htmlcov/
dist/
build/
*.egg-info
.venv
venv/
.mypy_cache
.ruff_cache
tests/
docs/
*.md
!README.md
infrastructure/
scripts/
```

**Step 6: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.env.local
.env.*.local

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Logs
*.log

# GCP
*.json
!infrastructure/**/*.json

# OS
.DS_Store
Thumbs.db

# Terraform
infrastructure/terraform/.terraform/
infrastructure/terraform/*.tfstate
infrastructure/terraform/*.tfstate.backup
infrastructure/terraform/.terraform.lock.hcl
```

**Step 7: Verify structure**

Run: `tree -L 3 -I '__pycache__|*.pyc'`

Expected: Should show the complete directory structure with all folders created

**Step 8: Commit**

```bash
git add .
git commit -m "feat: initialize project structure and dependencies

- Add pyproject.toml with all required dependencies
- Create project directory structure for modular monolith
- Add Dockerfile for containerized deployment
- Add environment configuration examples
- Configure linting and formatting tools"
```

---

## Task 2: Core Configuration and Database Setup

**Files:**
- Create: `platform/core/config.py`
- Create: `platform/core/database.py`
- Create: `platform/core/security.py`
- Create: `alembic.ini`
- Create: `platform/migrations/env.py`
- Create: `platform/migrations/versions/001_initial_schema.py`

**Step 1: Create core configuration**

File: `platform/core/config.py`

```python
"""Application configuration using pydantic-settings."""
from functools import lru_cache
from typing import Optional

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Database
    database_url: PostgresDsn
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # Redis
    redis_url: RedisDsn

    # GCP
    gcp_project_id: str
    gcs_bucket_artifacts: str
    gcs_bucket_datasets: str

    # API Security
    api_secret_key: str
    api_algorithm: str = "HS256"
    api_access_token_expire_minutes: int = 15
    api_refresh_token_expire_days: int = 7

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])

    # Celery
    celery_broker_url: str
    celery_result_backend: str

    # W&B
    wandb_project_prefix: str = "llm-experiments"

    # Feature Flags
    enable_anthropic: bool = True
    enable_custom_evaluators: bool = True

    # Environment
    environment: str = "development"
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

**Step 2: Create database utilities**

File: `platform/core/database.py`

```python
"""Database connection and session management."""
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from platform.core.config import get_settings

settings = get_settings()

# Create SQLAlchemy engine
engine = create_engine(
    str(settings.database_url),
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,  # Verify connections before using
    echo=settings.environment == "development",
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI routes to get database session.

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions in non-FastAPI code.

    Usage:
        with get_db_context() as db:
            items = db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable foreign keys for SQLite (used in tests)."""
    if "sqlite" in str(settings.database_url):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
```

**Step 3: Create security utilities**

File: `platform/core/security.py`

```python
"""Security utilities for authentication and encryption."""
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from platform.core.config import get_settings

settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload to encode in token (typically {"sub": user_id})
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.api_access_token_expire_minutes
        )

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(
        to_encode, settings.api_secret_key, algorithm=settings.api_algorithm
    )
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token with longer expiration."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.api_refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})

    encoded_jwt = jwt.encode(
        to_encode, settings.api_secret_key, algorithm=settings.api_algorithm
    )
    return encoded_jwt


def decode_token(token: str) -> dict:
    """
    Decode and verify a JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded payload

    Raises:
        JWTError: If token is invalid or expired
    """
    return jwt.decode(token, settings.api_secret_key, algorithms=[settings.api_algorithm])
```

**Step 4: Initialize Alembic for migrations**

Run: `alembic init platform/migrations`

Expected: Creates alembic.ini and platform/migrations directory

**Step 5: Configure Alembic**

File: `alembic.ini` (modify the sqlalchemy.url line)

```ini
[alembic]
script_location = platform/migrations
prepend_sys_path = .
version_path_separator = os

sqlalchemy.url =

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

**Step 6: Update Alembic env.py**

File: `platform/migrations/env.py`

```python
"""Alembic migration environment configuration."""
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from platform.core.config import get_settings
from platform.core.database import Base

# Import all models so Alembic can detect them
from platform.models.user import User  # noqa: F401
from platform.models.experiment import Experiment  # noqa: F401
from platform.models.job import Job  # noqa: F401

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set SQLAlchemy URL from settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", str(settings.database_url))

# Target metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Step 7: Commit**

```bash
git add platform/core/ alembic.ini platform/migrations/
git commit -m "feat: add core configuration and database setup

- Add settings management with pydantic-settings
- Create database session management with SQLAlchemy
- Add security utilities for JWT and password hashing
- Initialize Alembic for database migrations
- Configure connection pooling and health checks"
```

---

## Task 3: Database Models

**Files:**
- Create: `platform/models/user.py`
- Create: `platform/models/experiment.py`
- Create: `platform/models/job.py`
- Create: `platform/models/prompt_repository.py`
- Create: `platform/models/provider_config.py`
- Create: `platform/models/evaluation_run.py`

**Step 1: Create User model**

File: `platform/models/user.py`

```python
"""User model for authentication and settings."""
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from platform.core.database import Base


class User(Base):
    """User account model."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # W&B integration
    wandb_api_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Provider API keys (encrypted)
    openai_api_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    anthropic_api_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    experiments: Mapped[list["Experiment"]] = relationship(
        "Experiment", back_populates="user", cascade="all, delete-orphan"
    )
    prompt_repositories: Mapped[list["PromptRepository"]] = relationship(
        "PromptRepository", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}')>"
```

**Step 2: Create Experiment model**

File: `platform/models/experiment.py`

```python
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
```

**Step 3: Create Job model**

File: `platform/models/job.py`

```python
"""Job model for asynchronous task execution."""
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


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Type of job to execute."""

    LLM_INFERENCE = "llm_inference"
    EVALUATION = "evaluation"
    BULK_RUN = "bulk_run"


class Job(Base):
    """Asynchronous job for experiment execution."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id"), nullable=False, index=True
    )

    # Job metadata
    job_type: Mapped[JobType] = mapped_column(
        SQLEnum(JobType), default=JobType.LLM_INFERENCE, nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True
    )

    # Celery task info
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    worker_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Results
    success_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="jobs")

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, type='{self.job_type}', status='{self.status}')>"
```

**Step 4: Create PromptRepository model**

File: `platform/models/prompt_repository.py`

```python
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
```

**Step 5: Create ProviderConfig model**

File: `platform/models/provider_config.py`

```python
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
```

**Step 6: Create EvaluationRun model**

File: `platform/models/evaluation_run.py`

```python
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
```

**Step 7: Create initial migration**

Run: `alembic revision --autogenerate -m "initial schema"`

Expected: Creates a new migration file in `platform/migrations/versions/`

**Step 8: Review migration**

Run: `cat platform/migrations/versions/*_initial_schema.py | head -50`

Expected: Should show upgrade() and downgrade() functions with table creation statements

**Step 9: Commit**

```bash
git add platform/models/ platform/migrations/versions/
git commit -m "feat: add database models for core entities

- Add User model with authentication and API keys
- Add Experiment model with status tracking
- Add Job model for async task execution
- Add PromptRepository for Git-based prompts
- Add ProviderConfig for LLM provider settings
- Add EvaluationRun for evaluation tracking
- Generate initial Alembic migration"
```

---

## Task 4: OpenAI Provider Integration

**Files:**
- Create: `platform/providers/__init__.py`
- Create: `platform/providers/base.py`
- Create: `platform/providers/openai_provider.py`
- Create: `platform/providers/pricing.py`
- Create: `platform/providers/factory.py`
- Create: `tests/unit/test_openai_provider.py`

**Step 1: Write test for OpenAI provider (TDD)**

File: `tests/unit/test_openai_provider.py`

```python
"""Tests for OpenAI provider integration."""
import pytest
from unittest.mock import Mock, patch

from platform.providers.openai_provider import OpenAIProvider
from platform.providers.base import LLMResponse


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    with patch("platform.providers.openai_provider.OpenAI") as mock:
        yield mock


def test_openai_complete_basic(mock_openai_client):
    """Test basic completion with OpenAI."""
    # Arrange
    provider = OpenAIProvider(api_key="test-key")
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Hello!"))]
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
    mock_response.model = "gpt-4"
    mock_openai_client.return_value.chat.completions.create.return_value = mock_response

    # Act
    result = provider.complete("Say hello", {"temperature": 0.7})

    # Assert
    assert result.content == "Hello!"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.model == "gpt-4"
    assert result.cost_usd > 0


def test_openai_count_tokens():
    """Test token counting."""
    provider = OpenAIProvider(api_key="test-key")

    # Simple approximation: ~4 chars per token
    text = "Hello world"
    tokens = provider.count_tokens(text)

    assert tokens > 0
    assert tokens < len(text)  # Should be less than character count
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_openai_provider.py -v`

Expected: FAIL - ImportError or AttributeError (provider doesn't exist yet)

**Step 3: Create base provider interface**

File: `platform/providers/base.py`

```python
"""Base interface for LLM providers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str  # The completion text
    model: str  # Model that generated the response
    input_tokens: int  # Input/prompt tokens
    output_tokens: int  # Output/completion tokens
    latency_ms: float  # Response time in milliseconds
    cost_usd: float  # Estimated cost in USD
    provider_metadata: dict  # Provider-specific extras


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, prompt: str, config: dict) -> LLMResponse:
        """
        Execute a completion request.

        Args:
            prompt: The input prompt
            config: Model configuration (temperature, max_tokens, etc.)

        Returns:
            Standardized LLMResponse
        """
        pass

    @abstractmethod
    def stream_complete(self, prompt: str, config: dict) -> Iterator[str]:
        """
        Stream completion chunks.

        Args:
            prompt: The input prompt
            config: Model configuration

        Yields:
            Content chunks as they arrive
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text for cost estimation.

        Args:
            text: Text to count

        Returns:
            Approximate token count
        """
        pass

    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """
        Calculate cost based on token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name for pricing lookup

        Returns:
            Cost in USD
        """
        pass
```

**Step 4: Create pricing configuration**

File: `platform/providers/pricing.py`

```python
"""Pricing information for LLM providers."""

# Pricing per 1,000 tokens (as of Nov 2024)
PRICING = {
    "openai": {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    },
    "anthropic": {
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    },
}


def get_pricing(provider: str, model: str) -> dict[str, float]:
    """
    Get pricing for a specific provider and model.

    Args:
        provider: Provider name (openai, anthropic)
        model: Model name

    Returns:
        Dict with 'input' and 'output' prices per 1K tokens

    Raises:
        ValueError: If provider or model not found
    """
    if provider not in PRICING:
        raise ValueError(f"Unknown provider: {provider}")

    provider_pricing = PRICING[provider]

    # Try exact match first
    if model in provider_pricing:
        return provider_pricing[model]

    # Try partial match (e.g., "gpt-4-0613" matches "gpt-4")
    for model_prefix, pricing in provider_pricing.items():
        if model.startswith(model_prefix):
            return pricing

    raise ValueError(f"Unknown model '{model}' for provider '{provider}'")
```

**Step 5: Create OpenAI provider implementation**

File: `platform/providers/openai_provider.py`

```python
"""OpenAI provider implementation."""
import time
from typing import Iterator

from openai import OpenAI

from platform.providers.base import BaseLLMProvider, LLMResponse
from platform.providers.pricing import get_pricing


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self, api_key: str, base_url: str = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            base_url: Optional custom API base URL
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt: str, config: dict) -> LLMResponse:
        """Execute a completion request with OpenAI."""
        start_time = time.time()

        # Map config to OpenAI parameters
        model = config.get("model", "gpt-4o-mini")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 1000)
        top_p = config.get("top_p", 1.0)

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response data
        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Calculate cost
        cost_usd = self.calculate_cost(input_tokens, output_tokens, model)

        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            provider_metadata={
                "finish_reason": response.choices[0].finish_reason,
            },
        )

    def stream_complete(self, prompt: str, config: dict) -> Iterator[str]:
        """Stream completion chunks from OpenAI."""
        model = config.get("model", "gpt-4o-mini")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 1000)

        stream = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count.

        Note: This is a rough approximation. For exact counts,
        use tiktoken library (adds dependency).
        """
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost based on OpenAI pricing."""
        pricing = get_pricing("openai", model)

        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost
```

**Step 6: Create provider factory**

File: `platform/providers/factory.py`

```python
"""Factory for creating LLM provider instances."""
from typing import Optional

from platform.providers.base import BaseLLMProvider
from platform.providers.openai_provider import OpenAIProvider


def create_provider(
    provider_name: str, api_key: str, base_url: Optional[str] = None
) -> BaseLLMProvider:
    """
    Create an LLM provider instance.

    Args:
        provider_name: Name of provider (openai, anthropic)
        api_key: API key for the provider
        base_url: Optional custom API base URL

    Returns:
        Provider instance

    Raises:
        ValueError: If provider name is unknown
    """
    if provider_name == "openai":
        return OpenAIProvider(api_key=api_key, base_url=base_url)
    # elif provider_name == "anthropic":
    #     return AnthropicProvider(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/unit/test_openai_provider.py -v`

Expected: Tests should PASS

**Step 8: Commit**

```bash
git add platform/providers/ tests/unit/test_openai_provider.py
git commit -m "feat: add OpenAI provider integration

- Create base provider interface with LLMResponse
- Implement OpenAI provider with completion support
- Add pricing configuration for cost tracking
- Create provider factory for instantiation
- Add unit tests for OpenAI provider
- Support both streaming and non-streaming modes"
```

---

## Task 5: W&B Tracking Service

**Files:**
- Create: `platform/services/tracking_service.py`
- Create: `tests/unit/test_tracking_service.py`

**Step 1: Write test for W&B tracking (TDD)**

File: `tests/unit/test_tracking_service.py`

```python
"""Tests for W&B tracking service."""
import pytest
from unittest.mock import Mock, patch

from platform.services.tracking_service import TrackingService


@pytest.fixture
def mock_wandb():
    """Mock wandb module."""
    with patch("platform.services.tracking_service.wandb") as mock:
        yield mock


def test_create_run(mock_wandb):
    """Test creating a W&B run."""
    # Arrange
    service = TrackingService(api_key="test-key", project="test-project")
    mock_run = Mock()
    mock_run.id = "test-run-123"
    mock_wandb.init.return_value = mock_run

    # Act
    run_id = service.create_run(
        name="test-experiment",
        config={"model": "gpt-4", "temperature": 0.7},
    )

    # Assert
    assert run_id == "test-run-123"
    mock_wandb.init.assert_called_once()


def test_log_metrics(mock_wandb):
    """Test logging metrics to W&B."""
    # Arrange
    service = TrackingService(api_key="test-key", project="test-project")
    mock_run = Mock()
    service._current_run = mock_run

    # Act
    service.log_metrics({"accuracy": 0.95, "cost": 0.01})

    # Assert
    mock_run.log.assert_called_once_with({"accuracy": 0.95, "cost": 0.01})
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tracking_service.py -v`

Expected: FAIL - ImportError (service doesn't exist yet)

**Step 3: Create tracking service implementation**

File: `platform/services/tracking_service.py`

```python
"""W&B tracking service for experiment metrics."""
import os
from typing import Any, Optional

import wandb


class TrackingService:
    """Service for tracking experiments with Weights & Biases."""

    def __init__(self, api_key: str, project: str, entity: Optional[str] = None):
        """
        Initialize tracking service.

        Args:
            api_key: W&B API key
            project: W&B project name
            entity: Optional W&B entity (username or team)
        """
        self.api_key = api_key
        self.project = project
        self.entity = entity
        self._current_run = None

        # Set API key in environment
        os.environ["WANDB_API_KEY"] = api_key

    def create_run(
        self,
        name: str,
        config: dict,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Create a new W&B run.

        Args:
            name: Run name
            config: Configuration dict (model params, etc.)
            tags: Optional list of tags
            notes: Optional notes

        Returns:
            W&B run ID
        """
        self._current_run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config,
            tags=tags or [],
            notes=notes,
        )

        return self._current_run.id

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to current run.

        Args:
            metrics: Dict of metric name -> value
            step: Optional step number for time series
        """
        if not self._current_run:
            raise RuntimeError("No active run. Call create_run() first.")

        self._current_run.log(metrics, step=step)

    def log_artifact(
        self, artifact_path: str, artifact_type: str, name: str
    ) -> None:
        """
        Log an artifact (file, dataset, model) to W&B.

        Args:
            artifact_path: Path to artifact file
            artifact_type: Type of artifact (dataset, model, etc.)
            name: Artifact name
        """
        if not self._current_run:
            raise RuntimeError("No active run. Call create_run() first.")

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(artifact_path)
        self._current_run.log_artifact(artifact)

    def finish_run(self, exit_code: int = 0) -> None:
        """
        Finish the current run.

        Args:
            exit_code: Exit code (0 for success, non-zero for failure)
        """
        if self._current_run:
            self._current_run.finish(exit_code=exit_code)
            self._current_run = None

    def update_summary(self, summary: dict[str, Any]) -> None:
        """
        Update run summary (final metrics).

        Args:
            summary: Dict of summary metrics
        """
        if not self._current_run:
            raise RuntimeError("No active run. Call create_run() first.")

        for key, value in summary.items():
            self._current_run.summary[key] = value

    @property
    def run_url(self) -> Optional[str]:
        """Get URL for current run."""
        if self._current_run:
            return self._current_run.url
        return None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_tracking_service.py -v`

Expected: Tests should PASS

**Step 5: Commit**

```bash
git add platform/services/tracking_service.py tests/unit/test_tracking_service.py
git commit -m "feat: add W&B tracking service

- Create TrackingService for W&B integration
- Support run creation, metrics logging, and artifacts
- Add run summary and URL access
- Include unit tests with mocked wandb"
```

---

## Task 6: Experiment Service (Business Logic)

**Files:**
- Create: `platform/services/experiment_service.py`
- Create: `tests/unit/test_experiment_service.py`

**Step 1: Write test for experiment service (TDD)**

File: `tests/unit/test_experiment_service.py`

```python
"""Tests for experiment service."""
import pytest
from unittest.mock import Mock

from platform.models.experiment import Experiment, ExperimentStatus
from platform.services.experiment_service import ExperimentService


@pytest.fixture
def mock_db():
    """Mock database session."""
    return Mock()


def test_create_experiment(mock_db):
    """Test creating an experiment."""
    # Arrange
    service = ExperimentService(db=mock_db)

    # Act
    experiment = service.create_experiment(
        user_id=1,
        name="Test Experiment",
        provider="openai",
        model_name="gpt-4",
        model_params={"temperature": 0.7},
    )

    # Assert
    assert experiment.name == "Test Experiment"
    assert experiment.status == ExperimentStatus.DRAFT
    assert experiment.provider == "openai"
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


def test_submit_experiment(mock_db):
    """Test submitting an experiment for execution."""
    # Arrange
    service = ExperimentService(db=mock_db)
    experiment = Experiment(
        id=1,
        user_id=1,
        name="Test",
        status=ExperimentStatus.DRAFT,
        provider="openai",
        model_name="gpt-4",
    )

    # Act
    service.submit_experiment(experiment)

    # Assert
    assert experiment.status == ExperimentStatus.QUEUED
    mock_db.commit.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_experiment_service.py -v`

Expected: FAIL - ImportError (service doesn't exist yet)

**Step 3: Create experiment service implementation**

File: `platform/services/experiment_service.py`

```python
"""Business logic for experiment management."""
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from platform.models.experiment import Experiment, ExperimentStatus


class ExperimentService:
    """Service for managing LLM experiments."""

    def __init__(self, db: Session):
        """
        Initialize experiment service.

        Args:
            db: Database session
        """
        self.db = db

    def create_experiment(
        self,
        user_id: int,
        name: str,
        provider: str,
        model_name: str,
        model_params: dict,
        description: Optional[str] = None,
        prompt_repo_id: Optional[int] = None,
        prompt_file_path: Optional[str] = None,
        dataset_gcs_path: Optional[str] = None,
        dataset_inline: Optional[dict] = None,
        evaluation_config: Optional[dict] = None,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            user_id: ID of user creating the experiment
            name: Experiment name
            provider: LLM provider (openai, anthropic)
            model_name: Model name (gpt-4, etc.)
            model_params: Model configuration (temperature, max_tokens, etc.)
            description: Optional description
            prompt_repo_id: Optional prompt repository ID
            prompt_file_path: Optional path to prompt file
            dataset_gcs_path: Optional GCS path to dataset
            dataset_inline: Optional inline dataset
            evaluation_config: Optional evaluation configuration

        Returns:
            Created experiment
        """
        experiment = Experiment(
            user_id=user_id,
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            provider=provider,
            model_name=model_name,
            model_params=model_params,
            prompt_repo_id=prompt_repo_id,
            prompt_file_path=prompt_file_path,
            dataset_gcs_path=dataset_gcs_path,
            dataset_inline=dataset_inline,
            evaluation_config=evaluation_config,
        )

        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)

        return experiment

    def get_experiment(self, experiment_id: int, user_id: int) -> Optional[Experiment]:
        """
        Get an experiment by ID (with user access check).

        Args:
            experiment_id: Experiment ID
            user_id: User ID for access check

        Returns:
            Experiment or None if not found/not authorized
        """
        return (
            self.db.query(Experiment)
            .filter(Experiment.id == experiment_id, Experiment.user_id == user_id)
            .first()
        )

    def list_experiments(
        self,
        user_id: int,
        status: Optional[ExperimentStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Experiment]:
        """
        List experiments for a user.

        Args:
            user_id: User ID
            status: Optional status filter
            limit: Max number of results
            offset: Pagination offset

        Returns:
            List of experiments
        """
        query = self.db.query(Experiment).filter(Experiment.user_id == user_id)

        if status:
            query = query.filter(Experiment.status == status)

        return query.order_by(Experiment.created_at.desc()).limit(limit).offset(offset).all()

    def submit_experiment(self, experiment: Experiment) -> None:
        """
        Submit an experiment for execution.

        Args:
            experiment: Experiment to submit

        Raises:
            ValueError: If experiment is not in DRAFT status
        """
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot submit experiment with status {experiment.status}")

        experiment.status = ExperimentStatus.QUEUED
        self.db.commit()

    def update_experiment_status(
        self,
        experiment: Experiment,
        status: ExperimentStatus,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """
        Update experiment status and timestamps.

        Args:
            experiment: Experiment to update
            status: New status
            started_at: Optional start time
            completed_at: Optional completion time
        """
        experiment.status = status

        if started_at:
            experiment.started_at = started_at

        if completed_at:
            experiment.completed_at = completed_at

        self.db.commit()

    def update_experiment_results(
        self,
        experiment: Experiment,
        total_cost_usd: float,
        total_input_tokens: int,
        total_output_tokens: int,
    ) -> None:
        """
        Update experiment with execution results.

        Args:
            experiment: Experiment to update
            total_cost_usd: Total cost
            total_input_tokens: Total input tokens
            total_output_tokens: Total output tokens
        """
        experiment.total_cost_usd = total_cost_usd
        experiment.total_input_tokens = total_input_tokens
        experiment.total_output_tokens = total_output_tokens

        self.db.commit()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_experiment_service.py -v`

Expected: Tests should PASS

**Step 5: Commit**

```bash
git add platform/services/experiment_service.py tests/unit/test_experiment_service.py
git commit -m "feat: add experiment service with business logic

- Create ExperimentService for CRUD operations
- Add methods for creating, listing, and submitting experiments
- Include status and result updates
- Add user-based access control
- Include unit tests"
```

---

---

## Task 7: FastAPI Application Setup

**Files:**
- Create: `platform/api/main.py`
- Create: `platform/api/dependencies.py`
- Create: `tests/integration/test_api.py`

**Step 1: Create FastAPI app**

File: `platform/api/main.py`

```python
"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from platform.core.config import get_settings

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="LLM Experiment Platform API",
    description="API for running and managing LLM experiments",
    version="0.1.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "environment": settings.environment}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LLM Experiment Platform API",
        "version": "0.1.0",
        "docs": "/docs" if not settings.is_production else "disabled",
    }


# Import routers (will add later)
# from platform.api.routes import auth, experiments, jobs, prompts
# app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
# app.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
# app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
# app.include_router(prompts.router, prefix="/prompts", tags=["Prompts"])
```

**Step 2: Create API dependencies**

File: `platform/api/dependencies.py`

```python
"""FastAPI dependencies for authentication and database."""
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.orm import Session

from platform.core.database import get_db
from platform.core.security import decode_token
from platform.models.user import User

# Security scheme for JWT
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Get the current authenticated user from JWT token.

    Args:
        credentials: HTTP Authorization header with Bearer token
        db: Database session

    Returns:
        Current user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = decode_token(token)

        # Check token type
        if payload.get("type") != "access":
            raise credentials_exception

        # Extract user ID
        user_id: Optional[int] = payload.get("sub")
        if user_id is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    return user
```

**Step 3: Create integration test**

File: `tests/integration/test_api.py`

```python
"""Integration tests for FastAPI application."""
import pytest
from fastapi.testclient import TestClient

from platform.api.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
```

**Step 4: Run tests**

Run: `pytest tests/integration/test_api.py -v`

Expected: Tests should PASS

**Step 5: Test app locally**

Run: `uvicorn platform.api.main:app --reload`

Expected: Server starts on http://localhost:8000, can access /docs

**Step 6: Commit**

```bash
git add platform/api/ tests/integration/test_api.py
git commit -m "feat: create FastAPI application structure

- Add FastAPI app with CORS middleware
- Create health check and root endpoints
- Add authentication dependencies for JWT
- Include integration tests
- Configure OpenAPI docs (dev only)"
```

---

## Task 8: Authentication API Endpoints

**Files:**
- Create: `platform/api/routes/auth.py`
- Create: `platform/api/schemas/auth.py`
- Modify: `platform/api/main.py`
- Create: `tests/integration/test_auth_api.py`

**Step 1: Create auth schemas**

File: `platform/api/schemas/auth.py`

```python
"""Pydantic schemas for authentication."""
from pydantic import BaseModel, EmailStr


class UserRegister(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """User login request."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """User info response."""

    id: int
    email: str
    is_active: bool

    class Config:
        from_attributes = True
```

**Step 2: Create auth routes**

File: `platform/api/routes/auth.py`

```python
"""Authentication routes."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from platform.api.dependencies import get_current_user
from platform.api.schemas.auth import TokenResponse, UserLogin, UserRegister, UserResponse
from platform.core.database import get_db
from platform.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from platform.models.user import User

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user.

    Args:
        user_data: Registration data
        db: Database session

    Returns:
        Created user info

    Raises:
        HTTPException: If email already registered
    """
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create user
    user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        is_active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Login and receive access token.

    Args:
        credentials: Login credentials
        db: Database session

    Returns:
        JWT tokens

    Raises:
        HTTPException: If credentials are invalid
    """
    # Find user
    user = db.query(User).filter(User.email == credentials.email).first()

    # Verify password
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    # Create tokens
    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user info.

    Args:
        current_user: Current authenticated user

    Returns:
        User info
    """
    return current_user
```

**Step 3: Update main.py to include router**

File: `platform/api/main.py` (update the commented section)

```python
# Import routers
from platform.api.routes import auth

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
```

**Step 4: Create integration test**

File: `tests/integration/test_auth_api.py`

```python
"""Integration tests for authentication API."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from platform.api.main import app
from platform.core.database import Base, get_db

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


def test_register_user():
    """Test user registration."""
    response = client.post(
        "/auth/register",
        json={"email": "test@example.com", "password": "testpassword123"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data


def test_register_duplicate_email():
    """Test registering with duplicate email."""
    # First registration
    client.post(
        "/auth/register",
        json={"email": "duplicate@example.com", "password": "testpassword123"},
    )

    # Second registration (should fail)
    response = client.post(
        "/auth/register",
        json={"email": "duplicate@example.com", "password": "testpassword123"},
    )
    assert response.status_code == 400


def test_login():
    """Test user login."""
    # Register user
    client.post(
        "/auth/register",
        json={"email": "login@example.com", "password": "testpassword123"},
    )

    # Login
    response = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "testpassword123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_get_current_user():
    """Test getting current user info."""
    # Register and login
    client.post(
        "/auth/register",
        json={"email": "current@example.com", "password": "testpassword123"},
    )
    login_response = client.post(
        "/auth/login",
        json={"email": "current@example.com", "password": "testpassword123"},
    )
    token = login_response.json()["access_token"]

    # Get current user
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "current@example.com"
```

**Step 5: Run tests**

Run: `pytest tests/integration/test_auth_api.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add platform/api/routes/auth.py platform/api/schemas/auth.py tests/integration/test_auth_api.py platform/api/main.py
git commit -m "feat: add authentication API endpoints

- Create user registration endpoint
- Add login with JWT token generation
- Include get current user endpoint
- Add Pydantic schemas for auth
- Include comprehensive integration tests"
```

---

## Task 9: Experiment API Endpoints

**Files:**
- Create: `platform/api/routes/experiments.py`
- Create: `platform/api/schemas/experiment.py`
- Modify: `platform/api/main.py`
- Create: `tests/integration/test_experiments_api.py`

**Step 1: Create experiment schemas**

File: `platform/api/schemas/experiment.py`

```python
"""Pydantic schemas for experiments."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from platform.models.experiment import ExperimentStatus


class ExperimentCreate(BaseModel):
    """Experiment creation request."""

    name: str
    description: Optional[str] = None
    provider: str  # openai, anthropic
    model_name: str
    model_params: dict = {}
    prompt_repo_id: Optional[int] = None
    prompt_file_path: Optional[str] = None
    dataset_gcs_path: Optional[str] = None
    dataset_inline: Optional[dict] = None
    evaluation_config: Optional[dict] = None


class ExperimentResponse(BaseModel):
    """Experiment response."""

    id: int
    user_id: int
    name: str
    description: Optional[str]
    status: ExperimentStatus
    provider: str
    model_name: str
    model_params: dict
    wandb_run_id: Optional[str]
    total_cost_usd: Optional[float]
    total_input_tokens: Optional[int]
    total_output_tokens: Optional[int]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class ExperimentListResponse(BaseModel):
    """List of experiments response."""

    experiments: list[ExperimentResponse]
    total: int
    limit: int
    offset: int
```

**Step 2: Create experiment routes**

File: `platform/api/routes/experiments.py`

```python
"""Experiment management routes."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from platform.api.dependencies import get_current_user
from platform.api.schemas.experiment import (
    ExperimentCreate,
    ExperimentListResponse,
    ExperimentResponse,
)
from platform.core.database import get_db
from platform.models.experiment import ExperimentStatus
from platform.models.user import User
from platform.services.experiment_service import ExperimentService

router = APIRouter()


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    experiment_data: ExperimentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new experiment.

    Args:
        experiment_data: Experiment configuration
        current_user: Current authenticated user
        db: Database session

    Returns:
        Created experiment
    """
    service = ExperimentService(db)

    experiment = service.create_experiment(
        user_id=current_user.id,
        name=experiment_data.name,
        description=experiment_data.description,
        provider=experiment_data.provider,
        model_name=experiment_data.model_name,
        model_params=experiment_data.model_params,
        prompt_repo_id=experiment_data.prompt_repo_id,
        prompt_file_path=experiment_data.prompt_file_path,
        dataset_gcs_path=experiment_data.dataset_gcs_path,
        dataset_inline=experiment_data.dataset_inline,
        evaluation_config=experiment_data.evaluation_config,
    )

    return experiment


@router.get("/", response_model=ExperimentListResponse)
async def list_experiments(
    status: Optional[ExperimentStatus] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List experiments for current user.

    Args:
        status: Optional status filter
        limit: Max results
        offset: Pagination offset
        current_user: Current authenticated user
        db: Database session

    Returns:
        List of experiments
    """
    service = ExperimentService(db)
    experiments = service.list_experiments(
        user_id=current_user.id,
        status=status,
        limit=limit,
        offset=offset,
    )

    # Get total count
    total = db.query(User).filter(User.id == current_user.id).count()

    return ExperimentListResponse(
        experiments=experiments,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get experiment by ID.

    Args:
        experiment_id: Experiment ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        Experiment details

    Raises:
        HTTPException: If experiment not found or unauthorized
    """
    service = ExperimentService(db)
    experiment = service.get_experiment(experiment_id, current_user.id)

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )

    return experiment


@router.post("/{experiment_id}/run", response_model=ExperimentResponse)
async def submit_experiment(
    experiment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Submit experiment for execution.

    Args:
        experiment_id: Experiment ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        Updated experiment

    Raises:
        HTTPException: If experiment not found or cannot be submitted
    """
    service = ExperimentService(db)
    experiment = service.get_experiment(experiment_id, current_user.id)

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )

    try:
        service.submit_experiment(experiment)
        # TODO: Trigger Celery task here
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return experiment
```

**Step 3: Update main.py**

File: `platform/api/main.py` (add to imports and routers)

```python
from platform.api.routes import auth, experiments

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
```

**Step 4: Run manual test**

Run: `uvicorn platform.api.main:app --reload`

Then test with curl or in /docs:
- Create experiment
- List experiments
- Get experiment by ID
- Submit experiment

**Step 5: Commit**

```bash
git add platform/api/routes/experiments.py platform/api/schemas/experiment.py platform/api/main.py
git commit -m "feat: add experiment API endpoints

- Create experiment creation endpoint
- Add list experiments with filtering
- Add get experiment by ID
- Add submit experiment endpoint
- Include Pydantic schemas
- Enforce user-based access control"
```

---

## Task 10: Celery Worker Setup

**Files:**
- Create: `platform/workers/celery_app.py`
- Create: `platform/workers/tasks.py`
- Create: `scripts/start_worker.sh`

**Step 1: Create Celery app**

File: `platform/workers/celery_app.py`

```python
"""Celery application configuration."""
from celery import Celery

from platform.core.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "llm_experiment_platform",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
)

# Import tasks
from platform.workers import tasks  # noqa: F401, E402
```

**Step 2: Create task stubs**

File: `platform/workers/tasks.py`

```python
"""Celery tasks for experiment execution."""
import logging
from datetime import datetime

from platform.core.database import get_db_context
from platform.models.experiment import ExperimentStatus
from platform.models.job import Job, JobStatus, JobType
from platform.services.experiment_service import ExperimentService
from platform.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="run_experiment")
def run_experiment_task(self, experiment_id: int, user_id: int):
    """
    Execute an LLM experiment.

    Args:
        experiment_id: ID of experiment to run
        user_id: ID of user who owns the experiment

    Returns:
        Dict with results summary
    """
    logger.info(f"Starting experiment {experiment_id} for user {user_id}")

    with get_db_context() as db:
        # Get experiment
        service = ExperimentService(db)
        experiment = service.get_experiment(experiment_id, user_id)

        if not experiment:
            logger.error(f"Experiment {experiment_id} not found")
            return {"status": "error", "message": "Experiment not found"}

        # Create job record
        job = Job(
            experiment_id=experiment_id,
            job_type=JobType.LLM_INFERENCE,
            status=JobStatus.RUNNING,
            celery_task_id=self.request.id,
            started_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()

        # Update experiment status
        service.update_experiment_status(
            experiment,
            ExperimentStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

    try:
        # TODO: Implement actual experiment execution logic
        logger.info(f"Executing experiment {experiment_id}")

        # Placeholder for now
        result = {
            "status": "success",
            "message": "Experiment execution not yet implemented",
        }

        with get_db_context() as db:
            service = ExperimentService(db)
            experiment = service.get_experiment(experiment_id, user_id)

            # Update experiment status
            service.update_experiment_status(
                experiment,
                ExperimentStatus.COMPLETED,
                completed_at=datetime.utcnow(),
            )

            # Update job status
            job = db.query(Job).filter(Job.celery_task_id == self.request.id).first()
            if job:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                db.commit()

        return result

    except Exception as e:
        logger.error(f"Error executing experiment {experiment_id}: {e}", exc_info=True)

        with get_db_context() as db:
            service = ExperimentService(db)
            experiment = service.get_experiment(experiment_id, user_id)

            # Update experiment status
            service.update_experiment_status(
                experiment,
                ExperimentStatus.FAILED,
                completed_at=datetime.utcnow(),
            )

            # Update job status
            job = db.query(Job).filter(Job.celery_task_id == self.request.id).first()
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                db.commit()

        return {"status": "error", "message": str(e)}
```

**Step 3: Create worker start script**

File: `scripts/start_worker.sh`

```bash
#!/bin/bash
# Start Celery worker

set -e

echo "Starting Celery worker..."

celery -A platform.workers.celery_app worker \
    --loglevel=info \
    --concurrency=4 \
    --max-tasks-per-child=50 \
    --queues=default
```

**Step 4: Make script executable**

Run: `chmod +x scripts/start_worker.sh`

**Step 5: Test worker startup**

Run: `./scripts/start_worker.sh`

Expected: Worker starts and connects to Redis, shows "ready" status

**Step 6: Commit**

```bash
git add platform/workers/ scripts/start_worker.sh
git commit -m "feat: add Celery worker setup

- Create Celery app with configuration
- Add run_experiment task stub
- Configure task timeouts and limits
- Add worker startup script
- Include job status tracking"
```

---

## Task 11: Job Execution Logic (LLM Calls)

**Files:**
- Create: `platform/workers/executor.py`
- Modify: `platform/workers/tasks.py`
- Create: `tests/unit/test_executor.py`

**Step 1: Write test for executor (TDD)**

File: `tests/unit/test_executor.py`

```python
"""Tests for job executor."""
import pytest
from unittest.mock import Mock, patch

from platform.workers.executor import ExperimentExecutor


@pytest.fixture
def mock_provider():
    """Mock LLM provider."""
    return Mock()


@pytest.fixture
def mock_tracking_service():
    """Mock tracking service."""
    return Mock()


def test_executor_runs_test_cases(mock_provider, mock_tracking_service):
    """Test executor runs all test cases."""
    # Arrange
    executor = ExperimentExecutor(
        provider=mock_provider,
        tracking_service=mock_tracking_service,
    )

    test_cases = [
        {"input": "Hello", "expected": "Hi"},
        {"input": "Goodbye", "expected": "Bye"},
    ]

    mock_provider.complete.return_value = Mock(
        content="Response",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        latency_ms=100,
    )

    # Act
    results = executor.execute_test_cases(
        prompt_template="Say: {{ input }}",
        test_cases=test_cases,
        model_config={"temperature": 0.7},
    )

    # Assert
    assert len(results) == 2
    assert mock_provider.complete.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_executor.py -v`

Expected: FAIL - ImportError

**Step 3: Create executor implementation**

File: `platform/workers/executor.py`

```python
"""Experiment execution logic."""
import logging
from typing import Any

from jinja2 import Template

from platform.providers.base import BaseLLMProvider, LLMResponse
from platform.services.tracking_service import TrackingService

logger = logging.getLogger(__name__)


class ExperimentExecutor:
    """Execute LLM experiments with test cases."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        tracking_service: TrackingService,
    ):
        """
        Initialize executor.

        Args:
            provider: LLM provider instance
            tracking_service: W&B tracking service
        """
        self.provider = provider
        self.tracking = tracking_service

    def execute_test_cases(
        self,
        prompt_template: str,
        test_cases: list[dict[str, Any]],
        model_config: dict,
    ) -> list[dict[str, Any]]:
        """
        Execute experiment on test cases.

        Args:
            prompt_template: Jinja2 prompt template
            test_cases: List of test case dicts with variables
            model_config: Model configuration (temperature, etc.)

        Returns:
            List of result dicts with outputs and metrics
        """
        template = Template(prompt_template)
        results = []

        for i, test_case in enumerate(test_cases):
            logger.info(f"Processing test case {i + 1}/{len(test_cases)}")

            try:
                # Render prompt with test case variables
                prompt = template.render(**test_case)

                # Call LLM
                response = self.provider.complete(prompt, model_config)

                # Log to W&B
                self.tracking.log_metrics(
                    {
                        f"test_case_{i}_input_tokens": response.input_tokens,
                        f"test_case_{i}_output_tokens": response.output_tokens,
                        f"test_case_{i}_cost_usd": response.cost_usd,
                        f"test_case_{i}_latency_ms": response.latency_ms,
                    },
                    step=i,
                )

                # Store result
                result = {
                    "test_case_index": i,
                    "input": test_case,
                    "prompt": prompt,
                    "output": response.content,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "latency_ms": response.latency_ms,
                    "model": response.model,
                    "success": True,
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing test case {i}: {e}", exc_info=True)

                # Log error
                result = {
                    "test_case_index": i,
                    "input": test_case,
                    "error": str(e),
                    "success": False,
                }
                results.append(result)

        return results

    def calculate_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Calculate summary metrics from results.

        Args:
            results: List of result dicts

        Returns:
            Summary dict with aggregate metrics
        """
        successful_results = [r for r in results if r.get("success")]

        if not successful_results:
            return {
                "total_cases": len(results),
                "success_count": 0,
                "failure_count": len(results),
                "total_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }

        total_cost = sum(r.get("cost_usd", 0) for r in successful_results)
        total_input_tokens = sum(r.get("input_tokens", 0) for r in successful_results)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in successful_results)
        avg_latency = sum(r.get("latency_ms", 0) for r in successful_results) / len(
            successful_results
        )

        return {
            "total_cases": len(results),
            "success_count": len(successful_results),
            "failure_count": len(results) - len(successful_results),
            "total_cost_usd": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_latency_ms": avg_latency,
        }
```

**Step 4: Update tasks.py to use executor**

File: `platform/workers/tasks.py` (modify run_experiment_task)

```python
# Add imports at top
from platform.providers.factory import create_provider
from platform.services.tracking_service import TrackingService
from platform.workers.executor import ExperimentExecutor

# Replace TODO section in run_experiment_task with:

        # Get user API keys
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        # Get provider API key
        if experiment.provider == "openai":
            api_key = user.openai_api_key
        elif experiment.provider == "anthropic":
            api_key = user.anthropic_api_key
        else:
            raise ValueError(f"Unknown provider: {experiment.provider}")

        if not api_key:
            raise ValueError(f"No API key configured for {experiment.provider}")

        # Create provider
        provider = create_provider(experiment.provider, api_key)

        # Create W&B tracking service
        tracking = TrackingService(
            api_key=user.wandb_api_key,
            project=f"llm-experiments-{user_id}",
        )

        # Create W&B run
        run_id = tracking.create_run(
            name=experiment.name,
            config={
                "provider": experiment.provider,
                "model": experiment.model_name,
                **experiment.model_params,
            },
        )

        # Update experiment with W&B run ID
        experiment.wandb_run_id = run_id
        db.commit()

    # Execute experiment
    executor = ExperimentExecutor(provider=provider, tracking_service=tracking)

    # Load dataset (inline for now, GCS support later)
    test_cases = experiment.dataset_inline or []

    # Load prompt template
    # TODO: Load from Git repo, for now use inline
    prompt_template = "{{ input }}"

    # Execute
    results = executor.execute_test_cases(
        prompt_template=prompt_template,
        test_cases=test_cases,
        model_config=experiment.model_params,
    )

    # Calculate summary
    summary = executor.calculate_summary(results)

    # Update W&B summary
    tracking.update_summary(summary)
    tracking.finish_run(exit_code=0)

    result = {
        "status": "success",
        "summary": summary,
    }

    with get_db_context() as db:
        service = ExperimentService(db)
        experiment = service.get_experiment(experiment_id, user_id)

        # Update experiment with results
        service.update_experiment_results(
            experiment,
            total_cost_usd=summary["total_cost_usd"],
            total_input_tokens=summary["total_input_tokens"],
            total_output_tokens=summary["total_output_tokens"],
        )

        # ... rest of completion logic
```

**Step 5: Run tests**

Run: `pytest tests/unit/test_executor.py -v`

Expected: Tests PASS

**Step 6: Commit**

```bash
git add platform/workers/executor.py platform/workers/tasks.py tests/unit/test_executor.py
git commit -m "feat: implement experiment execution logic

- Create ExperimentExecutor for running test cases
- Add Jinja2 template rendering for prompts
- Integrate with LLM providers and W&B tracking
- Calculate summary metrics from results
- Update Celery task to use executor
- Include unit tests"
```

---

## Task 12: Basic Streamlit UI

**Files:**
- Create: `platform/ui/app.py`
- Create: `platform/ui/pages/experiments.py`
- Create: `platform/ui/pages/results.py`
- Create: `platform/ui/components/auth.py`
- Create: `scripts/start_ui.sh`

**Step 1: Create Streamlit app**

File: `platform/ui/app.py`

```python
"""Streamlit UI entry point."""
import streamlit as st

st.set_page_config(
    page_title="LLM Experiment Platform",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 LLM Experiment Platform")
st.markdown("Run and track LLM experiments with ease")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Experiments", "Results", "Settings"],
)

if page == "Experiments":
    from platform.ui.pages import experiments
    experiments.show()
elif page == "Results":
    from platform.ui.pages import results
    results.show()
elif page == "Settings":
    st.info("Settings page coming soon!")
```

**Step 2: Create experiments page**

File: `platform/ui/pages/experiments.py`

```python
"""Experiments page for Streamlit UI."""
import streamlit as st
import requests

API_URL = "http://localhost:8000"


def show():
    """Display experiments page."""
    st.header("Experiments")

    # Check if user is logged in
    if "access_token" not in st.session_state:
        st.warning("Please login first")
        show_login()
        return

    # Show create experiment form
    with st.expander("➕ Create New Experiment", expanded=False):
        show_create_experiment()

    st.divider()

    # Show experiments list
    show_experiments_list()


def show_login():
    """Display login form."""
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            response = requests.post(
                f"{API_URL}/auth/login",
                json={"email": email, "password": password},
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.access_token = data["access_token"]
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Login failed")


def show_create_experiment():
    """Display create experiment form."""
    with st.form("create_experiment"):
        name = st.text_input("Experiment Name")
        description = st.text_area("Description (optional)")

        col1, col2 = st.columns(2)
        with col1:
            provider = st.selectbox("Provider", ["openai", "anthropic"])
        with col2:
            if provider == "openai":
                model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4", "gpt-4o"])
            else:
                model = st.selectbox("Model", ["claude-3-5-sonnet-20241022"])

        temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
        max_tokens = st.number_input("Max Tokens", 100, 4000, 1000)

        dataset = st.text_area(
            "Dataset (JSON array)",
            value='[{"input": "Hello"}, {"input": "Goodbye"}]',
        )

        submit = st.form_submit_button("Create Experiment")

        if submit:
            import json

            try:
                dataset_parsed = json.loads(dataset)
            except json.JSONDecodeError:
                st.error("Invalid JSON in dataset")
                return

            response = requests.post(
                f"{API_URL}/experiments/",
                json={
                    "name": name,
                    "description": description,
                    "provider": provider,
                    "model_name": model,
                    "model_params": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    "dataset_inline": dataset_parsed,
                },
                headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            )

            if response.status_code == 201:
                st.success("Experiment created!")
                st.rerun()
            else:
                st.error(f"Failed to create experiment: {response.text}")


def show_experiments_list():
    """Display list of experiments."""
    response = requests.get(
        f"{API_URL}/experiments/",
        headers={"Authorization": f"Bearer {st.session_state.access_token}"},
    )

    if response.status_code != 200:
        st.error("Failed to load experiments")
        return

    data = response.json()
    experiments = data.get("experiments", [])

    if not experiments:
        st.info("No experiments yet. Create one to get started!")
        return

    for exp in experiments:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

            with col1:
                st.write(f"**{exp['name']}**")
                if exp.get("description"):
                    st.caption(exp["description"])

            with col2:
                st.write(f"Provider: {exp['provider']}")
                st.write(f"Model: {exp['model_name']}")

            with col3:
                status_emoji = {
                    "draft": "📝",
                    "queued": "⏳",
                    "running": "▶️",
                    "completed": "✅",
                    "failed": "❌",
                }.get(exp["status"], "❓")
                st.write(f"Status: {status_emoji} {exp['status']}")

            with col4:
                if exp["status"] == "draft":
                    if st.button("▶️ Run", key=f"run_{exp['id']}"):
                        run_response = requests.post(
                            f"{API_URL}/experiments/{exp['id']}/run",
                            headers={
                                "Authorization": f"Bearer {st.session_state.access_token}"
                            },
                        )
                        if run_response.status_code == 200:
                            st.success("Experiment submitted!")
                            st.rerun()

            st.divider()
```

**Step 3: Create results page stub**

File: `platform/ui/pages/results.py`

```python
"""Results page for Streamlit UI."""
import streamlit as st


def show():
    """Display results page."""
    st.header("Results")
    st.info("Results visualization coming soon!")
```

**Step 4: Create UI start script**

File: `scripts/start_ui.sh`

```bash
#!/bin/bash
# Start Streamlit UI

set -e

echo "Starting Streamlit UI..."

streamlit run platform/ui/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0
```

**Step 5: Make script executable**

Run: `chmod +x scripts/start_ui.sh`

**Step 6: Test UI**

Run: `./scripts/start_ui.sh`

Expected: UI opens at http://localhost:8501

**Step 7: Commit**

```bash
git add platform/ui/ scripts/start_ui.sh
git commit -m "feat: add basic Streamlit UI

- Create main Streamlit app with navigation
- Add experiments page with create and list views
- Add login functionality
- Include experiment submission
- Add results page stub
- Create UI startup script"
```

---

## Task 13: Docker Compose for Local Development

**Files:**
- Create: `docker-compose.yml`
- Create: `.env` (gitignored)
- Create: `scripts/dev_setup.sh`

**Step 1: Create docker-compose.yml**

File: `docker-compose.yml`

```yaml
version: '3.8'

services:
  # PostgreSQL database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: llm_user
      POSTGRES_PASSWORD: llm_password
      POSTGRES_DB: llm_experiments
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U llm_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for Celery
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # FastAPI application
  api:
    build:
      context: .
      dockerfile: Dockerfile
    command: uvicorn platform.api.main:app --host 0.0.0.0 --port 8080 --reload
    ports:
      - "8000:8080"
    env_file:
      - .env
    environment:
      DATABASE_URL: postgresql://llm_user:llm_password@postgres:5432/llm_experiments
      REDIS_URL: redis://redis:6379/0
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/1
    volumes:
      - .:/app
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  # Streamlit UI
  ui:
    build:
      context: .
      dockerfile: Dockerfile
    command: streamlit run platform/ui/app.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - .:/app
    depends_on:
      - api

  # Celery worker
  worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A platform.workers.celery_app worker --loglevel=info
    env_file:
      - .env
    environment:
      DATABASE_URL: postgresql://llm_user:llm_password@postgres:5432/llm_experiments
      REDIS_URL: redis://redis:6379/0
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/1
    volumes:
      - .:/app
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

volumes:
  postgres_data:
```

**Step 2: Create dev setup script**

File: `scripts/dev_setup.sh`

```bash
#!/bin/bash
# Setup local development environment

set -e

echo "Setting up local development environment..."

# Copy .env.example to .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please edit .env and add your API keys"
fi

# Start services
echo "Starting Docker Compose services..."
docker-compose up -d postgres redis

# Wait for database
echo "Waiting for PostgreSQL..."
sleep 5

# Run migrations
echo "Running database migrations..."
docker-compose run --rm api alembic upgrade head

echo ""
echo "✅ Development environment ready!"
echo ""
echo "Start services with:"
echo "  docker-compose up"
echo ""
echo "Or start individually:"
echo "  docker-compose up api      # FastAPI on http://localhost:8000"
echo "  docker-compose up ui       # Streamlit on http://localhost:8501"
echo "  docker-compose up worker   # Celery worker"
```

**Step 3: Make script executable**

Run: `chmod +x scripts/dev_setup.sh`

**Step 4: Test docker-compose**

Run: `docker-compose up -d`

Expected: All services start successfully

**Step 5: Test database migration**

Run: `docker-compose exec api alembic upgrade head`

Expected: Migrations run successfully

**Step 6: Test API health**

Run: `curl http://localhost:8000/health`

Expected: {"status": "healthy", ...}

**Step 7: Commit**

```bash
git add docker-compose.yml scripts/dev_setup.sh
git commit -m "feat: add Docker Compose for local development

- Create docker-compose.yml with all services
- Add PostgreSQL and Redis containers
- Configure API, UI, and worker services
- Add dev setup script for initialization
- Include health checks and dependencies"
```

---

## Task 14: Integration with Experiment Submission

**Files:**
- Modify: `platform/api/routes/experiments.py`
- Modify: `platform/workers/tasks.py`

**Step 1: Update experiment submission to trigger Celery task**

File: `platform/api/routes/experiments.py` (modify submit_experiment function)

```python
# Add import at top
from platform.workers.tasks import run_experiment_task

# In submit_experiment function, replace TODO with:

    try:
        service.submit_experiment(experiment)

        # Trigger Celery task
        task = run_experiment_task.delay(experiment.id, current_user.id)

        # Update experiment with task ID (optional, for tracking)
        logger.info(f"Submitted experiment {experiment.id} as task {task.id}")

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
```

**Step 2: Add missing User import to tasks.py**

File: `platform/workers/tasks.py` (add to imports)

```python
from platform.models.user import User
```

**Step 3: Test end-to-end flow**

Run:
1. Start all services: `docker-compose up`
2. Open UI: http://localhost:8501
3. Register/login
4. Create experiment
5. Submit experiment
6. Check worker logs for execution

Expected: Experiment runs successfully, status updates to completed

**Step 4: Commit**

```bash
git add platform/api/routes/experiments.py platform/workers/tasks.py
git commit -m "feat: integrate experiment submission with Celery

- Trigger Celery task when experiment is submitted
- Pass experiment and user IDs to worker
- Add proper error handling
- Complete end-to-end execution flow"
```

---

## Task 15: Documentation and Final Setup

**Files:**
- Create: `README.md`
- Create: `docs/getting-started.md`
- Create: `docs/api-reference.md`

**Step 1: Update README.md**

File: `README.md`

```markdown
# LLM Experiment Platform

A comprehensive platform for running, tracking, and evaluating LLM experiments with LLMOps capabilities.

## Features

- 🧪 **Experiment Management** - Create and run LLM experiments with multiple providers
- 📊 **W&B Integration** - Track metrics and visualize results with Weights & Biases
- 🔄 **Git-based Prompts** - Version control prompts with Git (coming soon)
- 📈 **Cost Tracking** - Monitor token usage and costs across experiments
- 🎯 **Evaluations** - Automated evaluation with W&B Weave (coming soon)
- 🌐 **REST API** - Full REST API for programmatic access
- 💻 **Web UI** - Streamlit-based web interface

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- OpenAI and/or Anthropic API keys
- W&B account and API key

### Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd llm-experiment-platform
```

2. Setup development environment:
```bash
./scripts/dev_setup.sh
```

3. Edit `.env` and add your API keys

4. Start services:
```bash
docker-compose up
```

5. Access the application:
- **Web UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

### Running Experiments

1. Register/login in the Web UI
2. Create a new experiment:
   - Choose provider (OpenAI/Anthropic)
   - Select model
   - Configure parameters
   - Add test cases
3. Submit experiment and track progress
4. View results in W&B

## Architecture

The platform uses a modular monolith architecture:

- **FastAPI** - REST API
- **Streamlit** - Web UI
- **Celery** - Async job execution
- **PostgreSQL** - Data persistence
- **Redis** - Job queue
- **W&B SDK** - Experiment tracking

## Development

### Project Structure

```
platform/
├── api/          # FastAPI application
├── ui/           # Streamlit interface
├── services/     # Business logic
├── workers/      # Celery tasks
├── models/       # Database models
├── providers/    # LLM provider integrations
└── core/         # Shared utilities

infrastructure/
└── terraform/    # GCP infrastructure (coming soon)

tests/
├── unit/         # Unit tests
└── integration/  # Integration tests
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=platform
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Deployment

See [docs/deployment.md](docs/deployment.md) for GCP deployment instructions (coming soon).

## License

MIT

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first (coming soon).
```

**Step 2: Verify everything works**

Run complete test:
1. `docker-compose down -v` (clean slate)
2. `./scripts/dev_setup.sh`
3. `docker-compose up`
4. Create and run experiment via UI
5. Verify in W&B dashboard

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README

- Add quick start guide
- Document architecture and features
- Include development instructions
- Add project structure overview"
```

---

## Summary

This implementation plan provides step-by-step instructions for building the MVP of the LLM Experiment Platform. Each task follows TDD principles with:

1. Write failing tests
2. Implement minimal code to pass
3. Run tests
4. Commit frequently

The plan covers:
- ✅ Project structure and dependencies
- ✅ Database models and migrations
- ✅ OpenAI provider integration
- ✅ W&B tracking service
- ✅ FastAPI REST API
- ✅ Authentication
- ✅ Experiment management
- ✅ Celery workers
- ✅ Job execution logic
- ✅ Streamlit UI
- ✅ Docker Compose setup
- ✅ End-to-end integration
- ✅ Documentation

## Next Steps

After completing this MVP:

1. **Add Anthropic provider**
2. **Implement Git-based prompt management**
3. **Add evaluation orchestration**
4. **Create GCP Terraform infrastructure**
5. **Build cost analytics dashboard**
6. **Add more comprehensive tests**
7. **Performance optimization**
8. **Security hardening**
