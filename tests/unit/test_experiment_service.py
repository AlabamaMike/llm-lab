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
