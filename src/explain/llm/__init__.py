from ._access_token import set_env_secrets
from ._client import LLMClient, LLMConfig, LLMResponse


def create_client(model: str | None = None, provider: str = "litellm", **kwargs) -> LLMClient:
    """Create an LLM client with the specified provider.

    Args:
        model: Model name (uses provider defaults if not specified)
        provider: LLM provider - 'anthropic', 'gemini', 'openai', or 'litellm' (default)
        **kwargs: Additional configuration options

    Returns:
        An LLM client instance conforming to the BaseLLMClient interface.
    """
    set_env_secrets()
    provider = provider or "litellm"
    # Set default models for each provider
    default_models = {
        "anthropic": "claude-sonnet-4@20250514",
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4.1",
        "litellm": "anthropic/claude-sonnet-4-20250514",
    }

    if model is None:
        model = default_models.get(provider)

    config = LLMConfig(provider=provider, model=model, **kwargs)
    return LLMClient(config)


__all__ = ["LLMClient", "LLMConfig", "LLMResponse", "create_client"]
