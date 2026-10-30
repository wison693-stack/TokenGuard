# Changelog

## 2.0.0

### Added
- **Streaming / SSE support** (`guard.guard_stream`, `guard.precheck_request`): live token counting, pass-through of
  chunks without buffering, emergency abort on budget overrun, generation loops and output-token caps. Sync and async.
- **Prompt-caching aware pricing** (`cache_pricing.py`): `cache_read_input_tokens`, `cache_creation_input_tokens`
  (5 min / 1 h), OpenAI `cached_tokens`; configurable `PricingTable`; `track_usage()` for provider usage objects.
- **Persistent / shared limits** (`redis_store.py`): Redis store (atomic Lua + `INCRBY`) and in-memory store,
  shared loop history, shared budget extensions, graceful degradation when the store is down.
- **Improved loop detection** (`loop_detector.py`): sha256 history fingerprints, near-duplicate normalisation,
  A-B-A-B cycle detection, optional embedding similarity, hard cut with `resume()`.
- `AgentAlert` hooks (`add_hook`) and non-interactive alerts for loop cuts and stream aborts.
- `run_all_tests.py`, `example_usage.py`, `requirements.txt`, full README.

### Changed
- `TokenGuard.track_action()` is now cache-aware, store-aware and uses sha256 history loop detection.
  Existing calls `track_action(input_tokens, output_tokens, command_signature)` keep working unchanged.
  `command_signature` is optional; new keyword arguments: `cache_read_input_tokens`,
  `cache_creation_input_tokens`, `cache_creation_1h_input_tokens`, `provider`, `messages`.
- Loop detection uses sha256 instead of md5 and, by default, **cuts the chain hard** (no override prompt).
  Pass `hard_loop_cut=False` to `TokenGuard(...)` to keep the previous "ask the developer" behaviour.
- `AgentAlert` prompt no longer crashes without an interactive stdin (EOF is treated as "no").

## 1.0.0
- Initial release: budget limit, md5 command-hash loop detection, `AgentAlert` terminal alerts.
