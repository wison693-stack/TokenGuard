import sys

from agent_alert import AgentAlert
from cache_pricing import PricingTable, Usage, normalize_usage
from loop_detector import LoopDetector
from stream_guard import StreamGuard, estimate_tokens


class TokenGuard:
    """
    TokenGuard: Safety Middleware for Claude Code & Autonomous AI CLI Agents.
    Prevents API budget exhaustion and infinite execution loops.

    Features:
      * cache-aware cost tracking (Prompt Caching discounts are honoured)
      * streaming / SSE protection (live token counting, emergency abort)
      * persistent, shared limits (Redis or in-memory store)
      * loop detection by hashing the agent's message history (sha256)
    """

    def __init__(self, max_budget_usd: float = 2.0, max_repeats: int = 3, hard_loop_cut: bool = True,
                 store=None, session_id: str = "default", pricing=None):
        """
        max_budget_usd : spending limit for the session in USD.
        max_repeats    : how many identical consecutive requests count as a loop.
        hard_loop_cut  : True  -> a loop is cut immediately (alert, no override prompt, chain stays halted
                                  until resume()).
                         False -> classic behaviour: the developer is asked whether to force-continue.
        store          : optional BudgetStore (RedisBudgetStore / InMemoryBudgetStore) for persistent,
                         shared limits. Equivalent to calling use_store() afterwards.
        session_id     : key of the budget session inside the store.
        pricing        : optional PricingTable or provider name ("anthropic" / "openai").
        """
        self.max_budget = max_budget_usd
        self.max_repeats = max_repeats
        self.hard_loop_cut = hard_loop_cut
        self.current_cost = 0.0
        self.command_history = []          # fingerprints (sha256) of the recent requests
        self.halted = False                # set when a loop was cut; cleared by resume()
        self.halt_reason = None
        self.pricing_explicit = False
        self._pricing = PricingTable()
        self._detector = LoopDetector(max_repeats=max_repeats)
        self._store = None
        self._session_id = session_id
        self._pending = 0.0                # spend not yet flushed to the store (store outage)
        self._store_warned = False
        if pricing is not None:
            self.set_pricing(pricing)
        if store is not None:
            self.use_store(store, session_id)

    # ------------------------------------------------------------------ core tracking
    def track_action(self, input_tokens: int, output_tokens: int, command_signature: str = None, *,
                     cache_read_input_tokens: int = 0, cache_creation_input_tokens: int = 0,
                     cache_creation_1h_input_tokens: int = 0, provider: str = None, messages=None) -> bool:
        """
        Calculates token costs in real-time, tracks request fingerprints, and invokes AgentAlert
        if monetary limits or loop thresholds are breached.
        Returns True if safe to proceed, False if execution must be halted.

        input_tokens  : FRESH (uncached) input tokens - i.e. Anthropic's usage.input_tokens.
                        For OpenAI-style usage objects call track_usage(), which subtracts the cached part.
        cache_*       : Prompt Caching counters from the response metadata (billed at discounted /
                        surcharged rates instead of the full input price).
        command_signature / messages : what the agent is doing. Either (or both) enables loop detection.
                        `messages` is the chat history; only its last turns are fingerprinted.
        With no cache counters the cost is the classic $3/M input + $15/M output formula.
        """
        if self.halted:
            return False

        # 1. Monetary Budget Limit Check (cache-aware)
        self._budget_add(self.compute_cost(input_tokens, output_tokens, cache_read_input_tokens,
                                           cache_creation_input_tokens, cache_creation_1h_input_tokens, provider))
        if not self._budget_gate():
            return False

        # 2. Infinite Loop Detection via sha256 fingerprints of the latest requests
        if messages is not None or command_signature is not None:
            verdict = self._detector.check(messages, command_signature)
            self.command_history.append(verdict.digest)
            del self.command_history[:-self._detector.history_size]
            if verdict.is_loop:
                return self._handle_loop(verdict, command_signature)
        return True

    def track_action_ext(self, *args, **kwargs) -> bool:
        """Alias of track_action() kept for backwards compatibility."""
        return self.track_action(*args, **kwargs)

    def track_usage(self, usage, command_signature: str = None, messages=None, provider: str = None) -> bool:
        """track_action() fed straight from a provider `usage` object / dict (Anthropic, OpenAI, Responses)."""
        u = normalize_usage(usage, provider)
        return self.track_action(u.input_tokens, u.output_tokens, command_signature,
                                 cache_read_input_tokens=u.cache_read_input_tokens,
                                 cache_creation_input_tokens=u.cache_creation_input_tokens,
                                 cache_creation_1h_input_tokens=u.cache_creation_1h_input_tokens,
                                 provider=u.provider, messages=messages)

    # ------------------------------------------------------------------ pricing (prompt caching)
    def set_pricing(self, pricing):
        """Pass a PricingTable (custom $/M and cache multipliers) or a provider name ('anthropic'/'openai')."""
        self._pricing = PricingTable.for_provider(pricing) if isinstance(pricing, str) else pricing
        self.pricing_explicit = True

    def _pricing_for(self, provider=None) -> PricingTable:
        return self._pricing if self.pricing_explicit else PricingTable.for_provider(provider or "anthropic")

    def compute_cost(self, input_tokens: int, output_tokens: int, cache_read_input_tokens: int = 0,
                     cache_creation_input_tokens: int = 0, cache_creation_1h_input_tokens: int = 0,
                     provider: str = None) -> float:
        """Cache-aware cost in USD. `input_tokens` = fresh (uncached) input tokens."""
        usage = Usage(input_tokens, output_tokens, cache_read_input_tokens,
                      cache_creation_input_tokens, cache_creation_1h_input_tokens, provider or "anthropic")
        return self._pricing_for(provider).cost(usage)

    # ------------------------------------------------------------------ persistence (Redis / memory)
    def use_store(self, store, session_id: str = "default"):
        """
        Keep this session's spend, limit and loop history in a shared store. All instances that use the
        same session_id share ONE budget, and it survives restarts. The first instance to register a
        session defines its limit.
        """
        self._store, self._session_id = store, session_id
        try:
            store.init_session(session_id, self.max_budget)
        except Exception as exc:
            self._warn_store(exc)
        self._refresh_from_store()
        self._detector.attach_store(store, session_id)

    def _warn_store(self, exc):
        if not self._store_warned:
            self._store_warned = True
            sys.stderr.write("[TokenGuard] budget store unavailable (%s) - using local counters, "
                             "spend will be re-synced when it recovers.\n" % (exc,))

    def _refresh_from_store(self):
        if self._store is None:
            return
        try:
            total, limit = self._store.get(self._session_id)
            self.current_cost = total + self._pending
            if limit is not None:
                self.max_budget = limit
            self._store_warned = False
        except Exception as exc:
            self._warn_store(exc)

    def _budget_add(self, cost: float):
        if self._store is None:
            self.current_cost += cost
            return
        try:
            total, limit = self._store.add_cost(self._session_id, cost + self._pending, self.max_budget)
            self._pending = 0.0
            self._store_warned = False
            self.current_cost, self.max_budget = total, limit
        except Exception as exc:                       # never crash the agent because Redis blinked
            self._pending += cost
            self.current_cost += cost
            self._warn_store(exc)

    def _extend_limit(self, amount: float = 1.0):
        if self._store is not None:
            try:
                self.max_budget = self._store.extend_limit(self._session_id, amount, self.max_budget)
                return
            except Exception as exc:
                self._warn_store(exc)
        self.max_budget += amount

    def _budget_prompt(self, spent: float) -> bool:
        """Human-in-the-loop flow: alert, grant an additional $1.00 allowance on consent."""
        if AgentAlert.trigger_budget_alert(spent, self.max_budget):
            self._extend_limit(1.0)                    # Grant additional $1.00 budget allowance upon user consent
            return True
        return False

    def _budget_gate(self) -> bool:
        if self.current_cost >= self.max_budget:
            return self._budget_prompt(self.current_cost)
        return True

    # ------------------------------------------------------------------ loop detection
    def configure_loop_detection(self, **kwargs):
        """Re-create the detector, e.g. window=2, normalize=True, embed_fn=my_fn, max_cycle_period=3."""
        kwargs.setdefault("max_repeats", self.max_repeats)
        self._detector = LoopDetector(**kwargs)
        if self._store is not None:
            self._detector.attach_store(self._store, self._session_id)

    def check_loop(self, messages=None, command_signature: str = None):
        """Record one agent request and return a LoopVerdict (no alert, no halt)."""
        return self._detector.check(messages, command_signature)

    def _handle_loop(self, verdict, command_signature=None) -> bool:
        if self.hard_loop_cut:
            return self._cut_loop(verdict, command_signature)
        user_granted = AgentAlert.trigger_loop_alert(command_signature or verdict.preview, verdict.repeats)
        if user_granted:
            self.command_history.clear()               # Clear history to prevent instant re-trigger
            self._detector.reset()
            return True
        return False

    def _cut_loop(self, verdict, command_signature=None) -> bool:
        self.halted = True
        self.halt_reason = "%s loop (%d repeats): %s" % (verdict.kind, verdict.repeats, verdict.preview)
        AgentAlert.trigger_loop_hard_alert(command_signature or verdict.preview, verdict.repeats,
                                           verdict.kind, verdict.period)
        return False

    def _request_gate(self, messages=None, command_signature=None):
        """None if the request may go ahead, else (reason, detail). Used by guard_stream()."""
        if self.halted:
            return ("agent_loop", self.halt_reason or "chain was halted earlier, call resume()")
        if messages is not None or command_signature is not None:
            verdict = self._detector.check(messages, command_signature)
            if verdict.is_loop:
                self._cut_loop(verdict, command_signature)
                return ("agent_loop", self.halt_reason)
        return None

    def resume(self):
        """Re-arm the guard after a hard loop cut (clears the halt flag and the loop history)."""
        self.halted, self.halt_reason = False, None
        self.command_history.clear()
        self._detector.reset()

    # ------------------------------------------------------------------ streaming (SSE)
    def precheck_request(self, messages=None, command_signature: str = None, input_tokens: int = None,
                         cache_read_input_tokens: int = 0, cache_creation_input_tokens: int = 0,
                         provider: str = None) -> bool:
        """
        Call BEFORE sending a (streaming) request: measures the prompt, projects its cost against the
        remaining budget and runs loop detection on the request. False = do not send it.
        (Pass `messages` here OR to guard_stream(), not both - otherwise the request is counted twice.)
        """
        if self._request_gate(messages, command_signature) is not None:
            return False
        self._refresh_from_store()
        fresh = input_tokens if input_tokens is not None else estimate_tokens(messages if messages is not None else "")
        projected = self.current_cost + self._pricing_for(provider).cost(
            Usage(fresh, 0, cache_read_input_tokens, cache_creation_input_tokens, provider=provider or "anthropic"))
        if projected >= self.max_budget:
            return self._budget_prompt(projected)
        return True

    def guard_stream(self, stream, **kwargs) -> StreamGuard:
        """
        Wrap any streaming response (sync/async iterator of SDK events, SSE lines or strings).
        Chunks are passed through untouched and instantly; tokens/cost are tracked live and the stream
        is aborted (upstream closed, alert fired, StreamAbortedError raised) on budget overrun,
        max_output_tokens or a model stuck in a generation loop. Options: input_tokens,
        cache_read_input_tokens, max_output_tokens, interactive, parse_sse, ... (see stream_guard.StreamGuard).

            for event in guard.guard_stream(client.messages.create(..., stream=True), input_tokens=900):
                render(event)
        """
        return StreamGuard(self, stream, **kwargs)
