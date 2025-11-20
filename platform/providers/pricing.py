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
