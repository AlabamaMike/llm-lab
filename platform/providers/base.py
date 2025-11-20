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
