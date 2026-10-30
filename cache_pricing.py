"""
cache_pricing.py - provider-aware cost calculation (Prompt Caching support).

Providers bill cached prompt tokens at a different rate than fresh ones:

  Anthropic : usage.input_tokens                -> fresh tokens only (cache NOT included)
              usage.cache_read_input_tokens     -> billed at ~0.1x input price
              usage.cache_creation_input_tokens -> billed at 1.25x (5 min TTL) / 2x (1 h TTL)
  OpenAI    : usage.prompt_tokens (or input_tokens) INCLUDES the cached part,
              prompt_tokens_details.cached_tokens -> billed at ~0.5x (model dependent)

`normalize_usage()` converts any of those shapes into one `Usage` object where
`input_tokens` always means "fresh tokens billed at the full input price".
The multipliers differ per model/provider and change over time, so everything
is configurable through `PricingTable`.
"""
from dataclasses import dataclass, replace


def _g(obj, key, default=None):
    """Read a key from a dict OR an attribute from an SDK object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _i(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class Usage:
    input_tokens: int = 0                    # fresh (uncached) input tokens, full price
    output_tokens: int = 0
    cache_read_input_tokens: int = 0         # served from cache (discounted)
    cache_creation_input_tokens: int = 0     # written to cache (total, 5m + 1h)
    cache_creation_1h_input_tokens: int = 0  # subset of the above written with 1 h TTL
    provider: str = "anthropic"

    def merge_max(self, other: "Usage") -> "Usage":
        """Streaming sends usage in several events (cumulative counters) - keep the max."""
        return Usage(
            input_tokens=max(self.input_tokens, other.input_tokens),
            output_tokens=max(self.output_tokens, other.output_tokens),
            cache_read_input_tokens=max(self.cache_read_input_tokens, other.cache_read_input_tokens),
            cache_creation_input_tokens=max(self.cache_creation_input_tokens, other.cache_creation_input_tokens),
            cache_creation_1h_input_tokens=max(self.cache_creation_1h_input_tokens, other.cache_creation_1h_input_tokens),
            provider=other.provider or self.provider,
        )

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


@dataclass
class PricingTable:
    """USD per 1M tokens plus cache multipliers (relative to the input price)."""
    input_per_mtok: float = 3.0     # same defaults as the original TokenGuard (Claude 3.5 Sonnet)
    output_per_mtok: float = 15.0
    cache_read_multiplier: float = 0.1
    cache_write_5m_multiplier: float = 1.25
    cache_write_1h_multiplier: float = 2.0

    @classmethod
    def for_provider(cls, provider: str = "anthropic", **overrides) -> "PricingTable":
        provider = (provider or "anthropic").lower()
        if provider == "openai":
            # Cached reads are 0.5x on gpt-4o class models (0.25x / 0.1x on newer ones) and
            # writes are free on most models. Override via kwargs for your exact model.
            base = cls(cache_read_multiplier=0.5, cache_write_5m_multiplier=1.0, cache_write_1h_multiplier=1.0)
        else:
            base = cls()
        return replace(base, **overrides) if overrides else base

    def cost(self, usage: Usage) -> float:
        m = 1_000_000
        in_p = self.input_per_mtok
        write_1h = min(usage.cache_creation_1h_input_tokens, usage.cache_creation_input_tokens)
        write_5m = usage.cache_creation_input_tokens - write_1h
        return (
            (usage.input_tokens / m * in_p)
            + (usage.output_tokens / m * self.output_per_mtok)
            + (usage.cache_read_input_tokens / m * in_p * self.cache_read_multiplier)
            + (write_5m / m * in_p * self.cache_write_5m_multiplier)
            + (write_1h / m * in_p * self.cache_write_1h_multiplier)
        )

    def full_price_cost(self, usage: Usage) -> float:
        """What the same call would cost with NO caching discounts (the 'naive' figure)."""
        m = 1_000_000
        return (usage.total_input_tokens / m * self.input_per_mtok) + (usage.output_tokens / m * self.output_per_mtok)


def normalize_usage(raw, provider: str = None) -> Usage:
    """
    Convert a provider `usage` object/dict (SDK object or plain dict) into `Usage`.
    Accepts Anthropic Messages usage, OpenAI Chat Completions usage and OpenAI Responses usage.
    """
    if raw is None:
        return Usage(provider=provider or "anthropic")
    if isinstance(raw, Usage):
        return raw

    out = _i(_g(raw, "output_tokens", None) or _g(raw, "completion_tokens", None))

    # --- OpenAI Chat Completions: prompt_tokens INCLUDES cached tokens ---
    if _g(raw, "prompt_tokens") is not None:
        details = _g(raw, "prompt_tokens_details")
        total_in = _i(_g(raw, "prompt_tokens"))
        cached = _i(_g(details, "cached_tokens"))
        written = _i(_g(details, "cache_write_tokens"))
        return Usage(input_tokens=max(total_in - cached - written, 0), output_tokens=out,
                     cache_read_input_tokens=cached, cache_creation_input_tokens=written,
                     provider=provider or "openai")

    # --- OpenAI Responses API: input_tokens INCLUDES cached (input_tokens_details) ---
    details = _g(raw, "input_tokens_details")
    if details is not None:
        total_in = _i(_g(raw, "input_tokens"))
        cached = _i(_g(details, "cached_tokens"))
        written = _i(_g(details, "cache_write_tokens"))
        return Usage(input_tokens=max(total_in - cached - written, 0), output_tokens=out,
                     cache_read_input_tokens=cached, cache_creation_input_tokens=written,
                     provider=provider or "openai")

    # --- Anthropic: input_tokens is FRESH tokens only; cache counters are separate ---
    creation = _g(raw, "cache_creation")
    return Usage(
        input_tokens=_i(_g(raw, "input_tokens")),
        output_tokens=out,
        cache_read_input_tokens=_i(_g(raw, "cache_read_input_tokens")),
        cache_creation_input_tokens=_i(_g(raw, "cache_creation_input_tokens")),
        cache_creation_1h_input_tokens=_i(_g(creation, "ephemeral_1h_input_tokens")),
        provider=provider or "anthropic",
    )
