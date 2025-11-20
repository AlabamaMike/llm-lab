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
