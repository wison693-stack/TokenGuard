import os

from agent_alert import AgentAlert
from guard import TokenGuard
from redis_store import (ADD_COST_LUA, EXTEND_LIMIT_LUA, PUSH_HISTORY_LUA, InMemoryBudgetStore,
                         RedisBudgetStore, to_nano)


class FakeRedis:
    """
    Minimal Redis double for the Python-side glue (key names, argument order, reply decoding).
    The Lua bodies are emulated in Python - run the REDIS_URL test below against a real server
    to exercise the real scripts.
    """

    def __init__(self):
        self.kv, self.lists = {}, {}

    def set(self, key, value, nx=False):
        if nx and key in self.kv:
            return None
        self.kv[key] = str(value)
        return True

    def get(self, key):
        return self.kv.get(key)

    def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)
            self.lists.pop(k, None)

    def eval(self, script, numkeys, *args):
        keys, argv = list(args[:numkeys]), list(args[numkeys:])
        if script == ADD_COST_LUA:
            self.kv.setdefault(keys[1], argv[1])
            self.kv[keys[0]] = str(int(self.kv.get(keys[0], 0)) + int(argv[0]))
            return [int(self.kv[keys[0]]), int(self.kv[keys[1]])]
        if script == EXTEND_LIMIT_LUA:
            self.kv.setdefault(keys[0], argv[1])
            self.kv[keys[0]] = str(int(self.kv[keys[0]]) + int(argv[0]))
            return int(self.kv[keys[0]])
        if script == PUSH_HISTORY_LUA:
            lst = self.lists.setdefault(keys[0], [])
            lst.append(argv[0])
            del lst[:-int(argv[1])]
            return list(lst)
        raise AssertionError("unknown script")


def _two_pods(store, budget=0.10):
    """Two TokenGuard instances (think: two Kubernetes pods) bound to the same session."""
    pods = []
    for _ in range(2):
        g = TokenGuard(max_budget_usd=budget, max_repeats=3)
        g.use_store(store, session_id="user-42")
        pods.append(g)
    return pods


def _check_backend(store, label):
    original_prompt = AgentAlert._prompt_user
    AgentAlert._prompt_user = staticmethod(lambda: False)
    try:
        store.reset("user-42")
        pod_a, pod_b = _two_pods(store)

        # budget is shared: $0.06 on pod A + $0.06 on pod B = $0.12 >= $0.10
        assert pod_a.track_action(15_000, 1_000, "a1") is True            # $0.06
        assert pod_b.track_action(15_000, 1_000, "b1") is False           # shared total $0.12
        assert abs(pod_b.current_cost - 0.12) < 1e-9, pod_b.current_cost
        print(f"✅ [{label}] two instances share one budget (total ${pod_b.current_cost:.2f})")

        # a "restart": brand-new process, same session -> spend is remembered
        pod_c = TokenGuard(max_budget_usd=0.10)
        pod_c.use_store(store, session_id="user-42")
        assert abs(pod_c.current_cost - 0.12) < 1e-9
        print(f"✅ [{label}] counters survive a restart (resumed at ${pod_c.current_cost:.2f})")

        # user consent (+$1) is persisted for everyone
        AgentAlert._prompt_user = staticmethod(lambda: True)
        assert pod_c.track_action(10, 10, "c1") is True
        pod_d = TokenGuard(max_budget_usd=0.10)
        pod_d.use_store(store, session_id="user-42")
        assert abs(pod_d.max_budget - 1.10) < 1e-9, pod_d.max_budget
        print(f"✅ [{label}] granted allowance is shared (limit now ${pod_d.max_budget:.2f})")
        AgentAlert._prompt_user = staticmethod(lambda: False)

        # loop detection history is shared too: 3 identical requests split across 3 pods
        store.reset("user-42")
        pods = []
        for _ in range(3):
            g = TokenGuard(max_budget_usd=10.0, max_repeats=3)
            g.use_store(store, session_id="user-42")
            pods.append(g)
        results = [p.track_action(1, 1, "pytest -x") for p in pods]
        assert results == [True, True, False], results
        print(f"✅ [{label}] loop history shared across instances")
        store.reset("user-42")
    finally:
        AgentAlert._prompt_user = original_prompt


def run_persistence_test():
    print("=" * 65)
    print("RUNNING TEST 7: Persistent / Shared Limits (TokenGuard)")
    print("=" * 65)
    _check_backend(InMemoryBudgetStore(), "in-memory")
    _check_backend(RedisBudgetStore(client=FakeRedis()), "redis-glue(fake client)")

    # store outage must never crash the agent: spend is kept locally and flushed on recovery
    class Flaky(InMemoryBudgetStore):
        down = False

        def add_cost(self, *a, **k):
            if self.down:
                raise ConnectionError("redis down")
            return super().add_cost(*a, **k)

    flaky = Flaky()
    g = TokenGuard(max_budget_usd=10.0)
    g.use_store(flaky, "s1")
    flaky.down = True
    assert g.track_action(1000, 100, "x") is True
    flaky.down = False
    assert g.track_action(1000, 100, "y") is True
    total, _ = flaky.get("s1")
    assert abs(total - 2 * (0.003 + 0.0015)) < 1e-9, total
    print("✅ store outage tolerated, offline spend flushed after recovery")

    url = os.environ.get("REDIS_URL")
    if url:
        _check_backend(RedisBudgetStore(url=url, prefix="tokenguard_test"), "REAL redis")
    else:
        print("ℹ️  set REDIS_URL=redis://localhost:6379/0 to run the same checks against a real Redis (real Lua).")
    print("✅ [TEST RESULT] Persistence checks passed.")


if __name__ == "__main__":
    run_persistence_test()
