"""
redis_store.py - persistent / shared budget counters for TokenGuard.

Why: when the budget lives in a Python variable it is lost on restart and every
pod/instance counts on its own. Here the per-session counters live in Redis and
are updated ATOMICALLY (single Lua script = INCRBY + limit read in one step), so
N instances share one budget and a restart does not reset it.

Money is stored as integer NANO-dollars (1e-9 USD) because INCRBY works on
integers and float drift would otherwise accumulate.

Keys use a Redis-Cluster hash tag {session_id} so all keys of one session live
in the same slot (required for multi-key Lua scripts on a cluster):
    tokenguard:{sid}:cost        INCRBY counter, spent nano-USD
    tokenguard:{sid}:limit       current limit in nano-USD (raised on user consent)
    tokenguard:{sid}:hist:exact  last N request fingerprints (loop detection)
    tokenguard:{sid}:hist:norm   last N normalised fingerprints

Two interchangeable backends are provided:
    RedisBudgetStore     - production (needs `pip install redis`; imported lazily)
    InMemoryBudgetStore  - same interface, zero dependencies, for tests / single process
"""
import os
import threading

NANO = 1_000_000_000


def to_nano(usd: float) -> int:
    return int(round(float(usd) * NANO))


def from_nano(nano) -> float:
    return int(nano) / NANO


# ----------------------------------------------------------------------------- Lua
# KEYS[1]=cost  KEYS[2]=limit   ARGV[1]=delta_nano  ARGV[2]=default_limit_nano  ARGV[3]=ttl_seconds
# Creates the limit on first use (so every instance agrees), atomically adds the cost,
# returns {total_nano, limit_nano}.
ADD_COST_LUA = """
local limit = redis.call('GET', KEYS[2])
if not limit then
  redis.call('SET', KEYS[2], ARGV[2])
  limit = ARGV[2]
end
local total = redis.call('INCRBY', KEYS[1], ARGV[1])
local ttl = tonumber(ARGV[3])
if ttl > 0 then
  redis.call('EXPIRE', KEYS[1], ttl)
  redis.call('EXPIRE', KEYS[2], ttl)
end
return {total, tonumber(limit)}
"""

# KEYS[1]=limit   ARGV[1]=extra_nano  ARGV[2]=default_limit_nano  ARGV[3]=ttl
EXTEND_LIMIT_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then
  redis.call('SET', KEYS[1], ARGV[2])
end
local limit = redis.call('INCRBY', KEYS[1], ARGV[1])
local ttl = tonumber(ARGV[3])
if ttl > 0 then redis.call('EXPIRE', KEYS[1], ttl) end
return limit
"""

# KEYS[1]=hist list   ARGV[1]=digest  ARGV[2]=max_len  ARGV[3]=ttl
# Appends the fingerprint, trims to the last N and returns the list (oldest -> newest).
PUSH_HISTORY_LUA = """
redis.call('RPUSH', KEYS[1], ARGV[1])
redis.call('LTRIM', KEYS[1], -tonumber(ARGV[2]), -1)
local ttl = tonumber(ARGV[3])
if ttl > 0 then redis.call('EXPIRE', KEYS[1], ttl) end
return redis.call('LRANGE', KEYS[1], 0, -1)
"""


def _s(value):
    return value.decode() if isinstance(value, (bytes, bytearray)) else value


class BudgetStore:
    """Interface every store implements (all money values are USD floats)."""

    def init_session(self, session_id, limit_usd, ttl=None):
        raise NotImplementedError

    def add_cost(self, session_id, cost_usd, default_limit_usd=0.0):
        """Atomically add cost. Returns (total_usd, limit_usd)."""
        raise NotImplementedError

    def get(self, session_id):
        """Returns (total_usd, limit_usd_or_None)."""
        raise NotImplementedError

    def extend_limit(self, session_id, extra_usd, default_limit_usd=0.0):
        """Atomically raise the limit. Returns the new limit in USD."""
        raise NotImplementedError

    def push_history(self, session_id, kind, digest, size):
        """Atomically append a fingerprint, return the last `size` (oldest -> newest)."""
        raise NotImplementedError

    def clear_history(self, session_id):
        raise NotImplementedError

    def reset(self, session_id):
        raise NotImplementedError


class InMemoryBudgetStore(BudgetStore):
    """Thread-safe in-process store with the same semantics as the Redis one."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cost, self._limit, self._hist = {}, {}, {}

    def init_session(self, session_id, limit_usd, ttl=None):
        with self._lock:
            self._limit.setdefault(session_id, to_nano(limit_usd))     # like SET NX
            self._cost.setdefault(session_id, 0)

    def add_cost(self, session_id, cost_usd, default_limit_usd=0.0):
        with self._lock:
            self._limit.setdefault(session_id, to_nano(default_limit_usd))
            self._cost[session_id] = self._cost.get(session_id, 0) + to_nano(cost_usd)
            return from_nano(self._cost[session_id]), from_nano(self._limit[session_id])

    def get(self, session_id):
        with self._lock:
            limit = self._limit.get(session_id)
            return from_nano(self._cost.get(session_id, 0)), (from_nano(limit) if limit is not None else None)

    def extend_limit(self, session_id, extra_usd, default_limit_usd=0.0):
        with self._lock:
            self._limit[session_id] = self._limit.get(session_id, to_nano(default_limit_usd)) + to_nano(extra_usd)
            return from_nano(self._limit[session_id])

    def push_history(self, session_id, kind, digest, size):
        with self._lock:
            hist = self._hist.setdefault((session_id, kind), [])
            hist.append(digest)
            del hist[:-int(size)]
            return list(hist)

    def clear_history(self, session_id):
        with self._lock:
            for key in [k for k in self._hist if k[0] == session_id]:
                del self._hist[key]

    def reset(self, session_id):
        with self._lock:
            self._cost.pop(session_id, None)
            self._limit.pop(session_id, None)
        self.clear_history(session_id)


class RedisBudgetStore(BudgetStore):
    """
    Redis-backed store.

        store = RedisBudgetStore(url="redis://localhost:6379/0")      # or REDIS_URL env var
        store = RedisBudgetStore(client=my_existing_redis_client)

    `ttl` (seconds, default 24 h) is refreshed on every write; use 0 to keep keys forever.
    """

    def __init__(self, client=None, url=None, prefix="tokenguard", ttl=86400):
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise ImportError("RedisBudgetStore needs the 'redis' package: pip install redis") from exc
            client = redis.Redis.from_url(url or os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
                                          decode_responses=True)
        self.client = client
        self.prefix = prefix
        self.ttl = int(ttl or 0)

    # keys ------------------------------------------------------------------
    def _k(self, session_id, suffix):
        return "%s:{%s}:%s" % (self.prefix, session_id, suffix)

    def _eval(self, script, keys, args):
        return self.client.eval(script, len(keys), *keys, *[str(a) for a in args])

    # interface ----------------------------------------------------------------
    def init_session(self, session_id, limit_usd, ttl=None):
        key = self._k(session_id, "limit")
        self.client.set(key, to_nano(limit_usd), nx=True)             # SET NX: first instance wins
        self.client.set(self._k(session_id, "cost"), 0, nx=True)

    def add_cost(self, session_id, cost_usd, default_limit_usd=0.0):
        total, limit = self._eval(ADD_COST_LUA,
                                  [self._k(session_id, "cost"), self._k(session_id, "limit")],
                                  [to_nano(cost_usd), to_nano(default_limit_usd), self.ttl])
        return from_nano(_s(total)), from_nano(_s(limit))

    def get(self, session_id):
        cost = _s(self.client.get(self._k(session_id, "cost")))
        limit = _s(self.client.get(self._k(session_id, "limit")))
        return from_nano(cost or 0), (from_nano(limit) if limit is not None else None)

    def extend_limit(self, session_id, extra_usd, default_limit_usd=0.0):
        limit = self._eval(EXTEND_LIMIT_LUA, [self._k(session_id, "limit")],
                           [to_nano(extra_usd), to_nano(default_limit_usd), self.ttl])
        return from_nano(_s(limit))

    def push_history(self, session_id, kind, digest, size):
        items = self._eval(PUSH_HISTORY_LUA, [self._k(session_id, "hist:" + kind)], [digest, int(size), self.ttl])
        return [_s(i) for i in items]

    def clear_history(self, session_id):
        self.client.delete(self._k(session_id, "hist:exact"), self._k(session_id, "hist:norm"))

    def reset(self, session_id):
        self.client.delete(self._k(session_id, "cost"), self._k(session_id, "limit"),
                           self._k(session_id, "hist:exact"), self._k(session_id, "hist:norm"))
