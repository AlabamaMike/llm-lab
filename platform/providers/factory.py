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
