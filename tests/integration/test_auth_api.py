"""Integration tests for authentication API."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from platform.api.main import app
from platform.core.database import Base, get_db

# Import all models to ensure relationships are registered
from platform.models.user import User
from platform.models.experiment import Experiment
from platform.models.prompt_repository import PromptRepository
from platform.models.job import Job
from platform.models.evaluation_run import EvaluationRun
from platform.models.provider_config import ProviderConfig

# Create test database with StaticPool for in-memory SQLite
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # Use StaticPool to share connection across threads
)
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
