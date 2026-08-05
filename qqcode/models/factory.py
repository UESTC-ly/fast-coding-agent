"""Adapter construction from configuration.

Turns a `Config` into a ready `BilledClient`. Two reasons this lives in its own
module rather than inside the adapters:

- The adapters take an already-built SDK client so tests can pass a fake. The
  factory is the only place that touches real SDK constructors and credentials.
- Tier-to-model mapping is overridable per run. Evaluation and smoke runs pin
  specific model ids without editing the adapters' defaults.
"""

from __future__ import annotations

from typing import Any

from qqcode.config import Config, ProviderConfig
from qqcode.models.anthropic_adapter import AnthropicAdapter
from qqcode.models.billing import DEFAULT_RETRY, BilledClient, RetryPolicy
from qqcode.models.openai_adapter import OpenAIAdapter
from qqcode.models.protocol import CostLedger, ModelClient, ModelTier

Provider = str  # "anthropic" | "openai"


def uniform_tiers(model: str) -> dict[ModelTier, str]:
    """Map every tier to one model id.

    Used when a run must be pinned to a single model — smoke tests and
    like-for-like cost comparisons, where letting FAST silently pick a cheaper
    model would make the numbers incomparable.
    """
    return dict.fromkeys(ModelTier, model)


def build_anthropic(
    cfg: ProviderConfig,
    *,
    tier_models: dict[ModelTier, str] | None = None,
) -> AnthropicAdapter:
    """Construct an Anthropic adapter from provider credentials.

    Raises:
        ImportError: The `anthropic` SDK is not installed.
    """
    from anthropic import Anthropic

    kwargs: dict[str, Any] = {"api_key": cfg.api_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return AnthropicAdapter(Anthropic(**kwargs), tier_models=tier_models)


def build_openai(
    cfg: ProviderConfig,
    *,
    tier_models: dict[ModelTier, str] | None = None,
    reasoning_effort: str | None = None,
) -> OpenAIAdapter:
    """Construct an OpenAI adapter from provider credentials.

    Raises:
        ImportError: The `openai` SDK is not installed.
    """
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": cfg.api_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAIAdapter(
        OpenAI(**kwargs),
        tier_models=tier_models,
        reasoning_effort=reasoning_effort,
    )


def build_adapter(
    config: Config,
    *,
    provider: Provider | None = None,
    tier_models: dict[ModelTier, str] | None = None,
    reasoning_effort: str | None = None,
) -> ModelClient:
    """Build the adapter for `provider`, defaulting to `config.default_provider`.

    `reasoning_effort` applies to the OpenAI path only; the Anthropic adapter has
    no equivalent knob wired, so it is not forwarded there rather than being
    accepted and silently dropped at request time.

    Raises:
        ValueError: The provider is unknown, or its credentials are absent.
    """
    name = provider or config.default_provider

    if name == "anthropic":
        if config.anthropic is None:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return build_anthropic(config.anthropic, tier_models=tier_models)

    if name == "openai":
        if config.openai is None:
            raise ValueError("OPENAI_API_KEY is not set")
        return build_openai(
            config.openai,
            tier_models=tier_models,
            reasoning_effort=reasoning_effort,
        )

    raise ValueError(f"Unknown provider: {name}")


def build_client(
    config: Config,
    *,
    ledger: CostLedger | None = None,
    provider: Provider | None = None,
    tier_models: dict[ModelTier, str] | None = None,
    reasoning_effort: str | None = None,
    retry_policy: RetryPolicy = DEFAULT_RETRY,
) -> tuple[BilledClient, CostLedger]:
    """Build the billed client and the ledger it accumulates into.

    Returns:
        `(client, ledger)`. The ledger is returned so callers can read the run's
        cost without reaching into the client.
    """
    ledger = ledger if ledger is not None else CostLedger()
    adapter = build_adapter(
        config,
        provider=provider,
        tier_models=tier_models,
        reasoning_effort=reasoning_effort,
    )
    return BilledClient(adapter, ledger=ledger, retry_policy=retry_policy), ledger
