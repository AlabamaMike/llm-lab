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
