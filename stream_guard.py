"""
stream_guard.py - Server-Sent-Events / streaming support for TokenGuard.

`StreamGuard` wraps ANY streaming response (OpenAI / Anthropic SDK streams, raw SSE
lines, plain generators of strings, sync or async) and passes every chunk through
to the caller UNCHANGED and IMMEDIATELY - no buffering, so the typing effect of
the chat UI is untouched. On the side it:

  * counts output tokens on the fly (provider usage events when present, otherwise a
    chars/4 estimate, tiktoken if installed) and projects the cost of the call;
  * checks the budget on every chunk (cache-aware, see cache_pricing.py);
  * detects a model stuck in a generation loop (same unit repeated over and over);
  * enforces an optional hard cap on output tokens;
  * on a violation ABORTS: closes the upstream stream (stops the provider from
    generating/billing more), fires an AgentAlert and raises StreamAbortedError;
  * commits the real cost to the guard exactly once (also if the consumer `break`s).
"""
import inspect
import json
import math

from agent_alert import AgentAlert
from cache_pricing import PricingTable, Usage, normalize_usage
from loop_detector import request_to_text

_ENCODER = None
_ENCODER_FAILED = False


def estimate_tokens(obj) -> int:
    """Cheap token estimate for a prompt/message list/str BEFORE the request is sent."""
    global _ENCODER, _ENCODER_FAILED
    text = obj if isinstance(obj, str) else request_to_text(obj)
    if not text:
        return 0
    if _ENCODER is None and not _ENCODER_FAILED:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER_FAILED = True
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    return max(1, math.ceil(len(text) / 4))


class StreamAbortedError(RuntimeError):
    """Raised inside the consumer's loop when TokenGuard cuts a stream."""

    def __init__(self, reason, detail="", partial_text="", output_tokens=0, cost=0.0):
        super().__init__("TokenGuard aborted stream (%s): %s" % (reason, detail))
        self.reason, self.detail = reason, detail
        self.partial_text, self.output_tokens, self.cost = partial_text, output_tokens, cost


# ------------------------------------------------------------------ chunk parsing
def _g(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class ChunkInfo:
    __slots__ = ("text", "usage", "done")

    def __init__(self, text="", usage=None, done=False):
        self.text, self.usage, self.done = text, usage, done


def _parse_sse(blob: str) -> ChunkInfo:
    info = ChunkInfo()
    for line in blob.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue                                    # event:/id:/retry:/comments
        payload = line[5:].strip()
        if payload == "[DONE]":
            info.done = True
            continue
        try:
            sub = extract_chunk(json.loads(payload))
        except ValueError:
            continue
        info.text += sub.text
        info.done = info.done or sub.done
        if sub.usage is not None:
            info.usage = sub.usage if info.usage is None else info.usage.merge_max(sub.usage)
    return info


def extract_chunk(chunk, parse_sse: bool = False) -> ChunkInfo:
    """Pull (text delta, usage, done) out of one streamed item, whatever its shape."""
    if chunk is None:
        return ChunkInfo()
    if isinstance(chunk, (bytes, bytearray)):
        chunk = bytes(chunk).decode("utf-8", "replace")
    if isinstance(chunk, str):
        return _parse_sse(chunk) if parse_sse else ChunkInfo(text=chunk)

    typ = _g(chunk, "type")
    # ---- Anthropic Messages stream events
    if typ == "message_start":
        u = _g(_g(chunk, "message"), "usage")
        return ChunkInfo(usage=normalize_usage(u) if u is not None else None)
    if typ == "content_block_delta":
        d = _g(chunk, "delta")
        return ChunkInfo(text=_g(d, "text") or _g(d, "partial_json") or _g(d, "thinking") or "")
    if typ == "message_delta":
        u = _g(chunk, "usage")
        return ChunkInfo(usage=normalize_usage(u) if u is not None else None)
    if typ == "message_stop":
        return ChunkInfo(done=True)
    # ---- OpenAI Responses API events
    if typ == "response.output_text.delta":
        return ChunkInfo(text=_g(chunk, "delta") or "")
    if typ == "response.completed":
        u = _g(_g(chunk, "response"), "usage")
        return ChunkInfo(usage=normalize_usage(u) if u is not None else None, done=True)

    # ---- OpenAI Chat Completions / legacy completions chunk
    text, done = "", False
    choices = _g(chunk, "choices")
    if choices:
        c0 = choices[0]
        delta = _g(c0, "delta")
        text = _g(delta, "content") or _g(c0, "text") or ""
        for call in (_g(delta, "tool_calls") or []):    # streamed tool-call arguments are output too
            text += _g(_g(call, "function"), "arguments") or ""
        done = bool(_g(c0, "finish_reason"))
    u = _g(chunk, "usage")
    return ChunkInfo(text=text, usage=normalize_usage(u) if u else None, done=done)


# --------------------------------------------------------------- repetition check
def detect_repetition(tail: str, min_span: int = 240, min_repeats: int = 6, max_unit: int = 400):
    """
    Return a description if the END of `tail` is one unit repeated back-to-back
    (e.g. "I will retry. I will retry. I will retry. ..."), else None.
    """
    n = len(tail)
    if n < min_span:
        return None
    # a single character (spaces / newlines / dots...) repeated for ages is degenerate output too
    last = tail[-1]
    run = n - len(tail.rstrip(last)) if last else 0
    if run >= max(min_span, 400):
        return "single character %r repeated %d times" % (last, run)
    for p in range(2, min(max_unit, n // min_repeats) + 1):
        unit = tail[-p:]
        if unit.strip() == "":
            continue
        reps, pos = 1, n - p
        while pos - p >= 0 and tail[pos - p:pos] == unit:
            reps += 1
            pos -= p
        if reps >= min_repeats and reps * p >= min_span:
            return "unit of %d chars repeated %d times in a row: %r" % (p, reps, unit[:60])
    return None


# ------------------------------------------------------------------- the wrapper
class StreamGuard:
    def __init__(self, guard, stream, *, input_tokens=0, cache_read_input_tokens=0,
                 cache_creation_input_tokens=0, provider=None, pricing=None,
                 command_signature=None, messages=None, max_output_tokens=None,
                 detect_repeats=True, repeat_min_span=240, repeat_min_repeats=6,
                 check_every_chars=64, interactive=True, raise_on_abort=True, parse_sse=False):
        self.guard, self.stream = guard, stream
        self._auto_pricing = pricing is None and not guard.pricing_explicit
        self.pricing = pricing or guard._pricing_for(provider)
        self.provider = provider
        self.command_signature, self.messages = command_signature, messages
        self.max_output_tokens = max_output_tokens
        self.detect_repeats = detect_repeats
        self.repeat_min_span, self.repeat_min_repeats = repeat_min_span, repeat_min_repeats
        self.check_every_chars = max(1, int(check_every_chars))
        self.interactive, self.raise_on_abort, self.parse_sse = interactive, raise_on_abort, parse_sse
        # request-side usage known up-front; response events can only raise it (merge_max)
        self.usage = Usage(input_tokens=int(input_tokens), cache_read_input_tokens=int(cache_read_input_tokens),
                           cache_creation_input_tokens=int(cache_creation_input_tokens),
                           provider=provider or "anthropic")
        self._pieces, self._chars, self._since_check = [], 0, 0
        self._tail = ""
        self.finished = self.aborted = self._committed = False
        self.abort_reason = self.abort_detail = None
        self.cost = 0.0
        self.safe_to_continue = True

    # ---- observable state
    @property
    def text(self):
        return "".join(self._pieces)

    @property
    def output_tokens(self):
        return max(self.usage.output_tokens, math.ceil(self._chars / 4))

    def projected_cost(self):
        u = Usage(self.usage.input_tokens, self.output_tokens, self.usage.cache_read_input_tokens,
                  self.usage.cache_creation_input_tokens, self.usage.cache_creation_1h_input_tokens, self.usage.provider)
        return self.pricing.cost(u)

    # ---- core logic shared by sync and async iteration
    def _start(self):
        """Request-level gate before the first chunk is delivered."""
        verdict = self.guard._request_gate(self.messages, self.command_signature)
        if verdict is not None:
            return verdict
        self.guard._refresh_from_store()
        return None

    def _on_chunk(self, chunk):
        """Returns (reason, detail) if the stream must be cut, else None."""
        info = extract_chunk(chunk, self.parse_sse)
        if info.usage is not None:
            self.usage = self.usage.merge_max(info.usage)
            if self._auto_pricing and self.provider is None:        # provider detected from the usage shape
                self.pricing = self.guard._pricing_for(info.usage.provider)
        if info.text:
            self._pieces.append(info.text)
            self._chars += len(info.text)
            self._since_check += len(info.text)
            self._tail = (self._tail + info.text)[-4000:]

        # 1) hard cap on generated tokens
        if self.max_output_tokens and self.output_tokens >= self.max_output_tokens:
            return "max_output_tokens", "generated ~%d tokens (cap %d)" % (self.output_tokens, self.max_output_tokens)

        # 2) budget, projected live (already committed spend + this call so far)
        projected_total = self.guard.current_cost + self.projected_cost()
        if projected_total >= self.guard.max_budget:
            if self.interactive and self.guard._budget_prompt(projected_total):
                pass                                            # user granted more budget, keep streaming
            else:
                return "budget", "projected spend $%.4f reached limit $%.2f" % (projected_total, self.guard.max_budget)

        # 3) model stuck repeating itself
        if self.detect_repeats and self._since_check >= self.check_every_chars:
            self._since_check = 0
            found = detect_repetition(self._tail, self.repeat_min_span, self.repeat_min_repeats)
            if found:
                return "generation_loop", found
        return None

    def _finalize(self):
        """Commit the cost exactly once, on normal end, abort, or early `break`."""
        if self._committed:
            return
        self._committed = True
        self.cost = self.projected_cost()
        self.guard._budget_add(self.cost)
        if not self.aborted:
            self.safe_to_continue = self.guard._budget_gate()   # same alert flow as track_action()

    def _abort(self, reason, detail):
        self.aborted, self.abort_reason, self.abort_detail = True, reason, detail
        self.safe_to_continue = False
        AgentAlert.trigger_stream_abort_alert(reason, detail, self.output_tokens, self.projected_cost())

    def _raise(self):
        return StreamAbortedError(self.abort_reason, self.abort_detail or "", self.text, self.output_tokens, self.cost)

    @staticmethod
    def _close_quietly(stream):
        for name in ("close", "aclose"):
            fn = getattr(stream, name, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    return None
        return None

    # ---- sync iteration
    def __iter__(self):
        try:
            blocked = self._start()
            if blocked:
                self._abort(*blocked)
                self._close_quietly(self.stream)
                self._finalize()
                if self.raise_on_abort:
                    raise self._raise()
                return
            for chunk in self.stream:
                verdict = self._on_chunk(chunk)
                if verdict:
                    self._abort(*verdict)
                    self._close_quietly(self.stream)            # stop the provider generating (and billing)
                    self._finalize()
                    if self.raise_on_abort:
                        raise self._raise()
                    return
                yield chunk
            self.finished = True
        finally:
            self._finalize()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._finalize()
        return False

    def close(self):
        self._close_quietly(self.stream)
        self._finalize()

    # ---- async iteration
    async def __aiter__(self):
        try:
            blocked = self._start()
            if blocked:
                self._abort(*blocked)
                res = self._close_quietly(self.stream)
                if inspect.isawaitable(res):
                    await res
                self._finalize()
                if self.raise_on_abort:
                    raise self._raise()
                return
            source = self.stream.__aiter__() if hasattr(self.stream, "__aiter__") else self.stream
            async for chunk in _aiter(source):
                verdict = self._on_chunk(chunk)
                if verdict:
                    self._abort(*verdict)
                    res = self._close_quietly(self.stream)
                    if inspect.isawaitable(res):
                        await res
                    self._finalize()
                    if self.raise_on_abort:
                        raise self._raise()
                    return
                yield chunk
            self.finished = True
        finally:
            self._finalize()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self._finalize()
        return False


async def _aiter(source):
    if hasattr(source, "__anext__"):
        while True:
            try:
                yield await source.__anext__()
            except StopAsyncIteration:
                return
    else:
        for item in source:
            yield item
