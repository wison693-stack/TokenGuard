"""
loop_detector.py - agent loop detection based on hashing of the message history.

Autonomous agents re-send the model nearly the same turn over and over
(same tool call -> same error -> same tool call ...). A raw hash of the whole
conversation never repeats because the history keeps growing, so we fingerprint
only the LAST `window` messages (default: the assistant's action + the tool
result it got back) and look at the sequence of fingerprints:

  exact      - sha256 of the canonical turn repeated `max_repeats` times in a row
  near-dup   - same, after normalising volatile noise (numbers, ids, timestamps, ANSI)
  cycle      - short patterns A,B,A,B,A,B (period 2..max_cycle_period)
  semantic   - optional: cosine similarity of embeddings from your own `embed_fn`

History can live in-process (default) or in a shared store (see redis_store.py),
so a loop is also caught when the agent's requests are spread across several pods.
"""
import hashlib
import json
import math
import re
from collections import deque
from dataclasses import dataclass

_ID_KEYS = {"id", "tool_use_id", "tool_call_id", "call_id"}   # random per request - never part of the hash
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_HEX = re.compile(r"\b(?:0x)?[0-9a-f]{8,}\b")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_WS = re.compile(r"\s+")


def _plain(obj):
    """SDK objects (pydantic etc.) -> plain python structures."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _plain(fn())
            except Exception:
                pass
    return str(obj)


def _canon(obj) -> str:
    """Deterministic text for any structure, minus per-request random ids."""
    obj = _plain(obj)
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k not in _ID_KEYS}
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str) if not isinstance(obj, str) else obj


def _content_text(content) -> str:
    content = _plain(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_text(b) for b in content)
    if isinstance(content, dict):
        t = content.get("type")
        if t == "text":
            return str(content.get("text", ""))
        if t == "tool_use":
            return "tool_use:%s:%s" % (content.get("name"), _canon(content.get("input")))
        if t == "tool_result":
            return "tool_result:" + _content_text(content.get("content"))
        return _canon(content)
    return str(content)


def message_to_text(message) -> str:
    message = _plain(message)
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return str(message)
    parts = ["%s:%s" % (message.get("role", ""), _content_text(message.get("content")))]
    for call in message.get("tool_calls") or []:          # OpenAI style tool calls
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        parts.append("tool_call:%s:%s" % (fn.get("name"), fn.get("arguments")))
    return "\n".join(parts)


def request_to_text(request, window: int = None) -> str:
    """Flatten a prompt / message list / {'messages': [...]} to text (last `window` messages)."""
    request = _plain(request)
    if request is None:
        return ""
    if isinstance(request, dict) and "messages" in request:
        request = request["messages"]
    if isinstance(request, str):
        return request
    if isinstance(request, list):
        msgs = request[-window:] if window else request
        return "\n".join(message_to_text(m) for m in msgs)
    return message_to_text(request)


def normalize_text(text: str) -> str:
    text = _ANSI.sub("", text.lower())
    text = _UUID.sub("<id>", text)
    text = _HEX.sub("<hex>", text)
    text = _NUM.sub("#", text)
    return _WS.sub(" ", text).strip()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class LoopVerdict:
    is_loop: bool = False
    kind: str = ""          # exact | near-duplicate | cycle | near-cycle | semantic
    repeats: int = 0
    period: int = 1
    digest: str = ""
    preview: str = ""


class LoopDetector:
    def __init__(self, max_repeats: int = 3, history_size: int = 50, window: int = 2,
                 normalize: bool = True, embed_fn=None, similarity_threshold: float = 0.95,
                 max_cycle_period: int = 3, store=None, session_id: str = "default"):
        self.max_repeats = max(2, int(max_repeats))
        self.window = window
        self.normalize = normalize
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_cycle_period = max(1, int(max_cycle_period))
        self.history_size = max(int(history_size), self.max_repeats * self.max_cycle_period + 1)
        self.store = store
        self.session_id = session_id
        self._local = {"exact": deque(maxlen=self.history_size), "norm": deque(maxlen=self.history_size)}
        self._vectors = deque(maxlen=self.max_repeats)

    # ------------------------------------------------------------------ public
    def attach_store(self, store, session_id: str = "default"):
        self.store, self.session_id = store, session_id

    def reset(self):
        for d in self._local.values():
            d.clear()
        self._vectors.clear()
        if self.store is not None:
            try:
                self.store.clear_history(self.session_id)
            except Exception:
                pass

    def fingerprint(self, request, signature: str = None):
        text = request_to_text(request, self.window)
        if signature:
            text = signature.strip() + "\n" + text
        return text, sha256_hex(text), sha256_hex(normalize_text(text))

    def check(self, request=None, signature: str = None) -> LoopVerdict:
        """Record one agent request and report whether the chain is looping."""
        text, exact, norm = self.fingerprint(request, signature)
        preview = (signature or text).strip().replace("\n", " ")[:120]

        hist = self._push("exact", exact)
        period = self._scan(hist)
        if period:
            return LoopVerdict(True, "exact" if period == 1 else "cycle", self._tail_repeats(hist, period),
                               period, exact, preview)

        if self.normalize:
            hist_n = self._push("norm", norm)
            period = self._scan(hist_n)
            if period:
                return LoopVerdict(True, "near-duplicate" if period == 1 else "near-cycle",
                                   self._tail_repeats(hist_n, period), period, norm, preview)

        if self.embed_fn is not None:
            try:
                self._vectors.append(list(self.embed_fn(text)))
            except Exception:
                self._vectors.clear()
            vecs = list(self._vectors)
            if len(vecs) >= self.max_repeats and all(
                    _cosine(vecs[i], vecs[i + 1]) >= self.similarity_threshold for i in range(len(vecs) - 1)):
                return LoopVerdict(True, "semantic", self.max_repeats, 1, exact, preview)

        return LoopVerdict(False, digest=exact, preview=preview)

    # ---------------------------------------------------------------- internals
    def _push(self, kind: str, digest: str):
        if self.store is not None:
            try:
                return list(self.store.push_history(self.session_id, kind, digest, self.history_size))
            except Exception:
                pass                                    # store down -> degrade to local history
        self._local[kind].append(digest)
        return list(self._local[kind])

    @staticmethod
    def _tail_repeats(hist, p: int) -> int:
        if len(hist) < p:
            return 0
        k = 1
        while len(hist) >= p * (k + 1) and hist[-p * (k + 1):-p * k] == hist[-p:]:
            k += 1
        return k

    def _scan(self, hist) -> int:
        """Return the period (1..max_cycle_period) of a repeating tail, or 0."""
        for p in range(1, self.max_cycle_period + 1):
            if len(hist) >= p * self.max_repeats and self._tail_repeats(hist, p) >= self.max_repeats:
                return p
        return 0
