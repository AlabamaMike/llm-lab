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
