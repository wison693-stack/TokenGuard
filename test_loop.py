import time
from guard import TokenGuard

def run_loop_test():
    """
    Simulation Test: Detects and stops an agent stuck in an infinite retry loop.
    """
    print("=" * 65)
    print("RUNNING TEST 1: Infinite Loop Protection (TokenGuard)")
    print("=" * 65)

    # Initialize guard with a $10 budget limit and a maximum repeat limit of 3
    guard = TokenGuard(max_budget_usd=10.0, max_repeats=3)
    failing_command = "pytest tests/test_core.py --maxfail=1"

    for step in range(1, 6):
        print(f"\n[Agent Action #{step}] Executing command: '{failing_command}'...")

        # Simulate small token consumption per step
        is_safe = guard.track_action(
            input_tokens=1200, 
            output_tokens=250, 
            command_signature=failing_command
        )

        if not is_safe:
            print(f"❌ [TEST RESULT] Agent execution halted at step {step} due to loop protection.")
            return

        print(f"✅ [Step {step}] Executed successfully.")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Added: history-hash loop detection for autonomous agents (sha256 of the last turns)
# ---------------------------------------------------------------------------
def _agent_history(step, command="pytest tests/test_core.py --maxfail=1", result="FAILED test_core.py::test_x - AssertionError"):
    """A growing agent conversation. Tool-call ids are random per request; only the LAST turn repeats."""
    history = [{"role": "user", "content": "Fix the failing test."}]
    for i in range(1, step + 1):
        history.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"toolu_{step}_{i}", "name": "bash", "input": {"cmd": command}}]})
        history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"toolu_{step}_{i}", "content": result}]})
    return history


def run_history_hash_loop_test():
    """
    Simulation Test: 3 identical agent turns in a row (same tool call -> same result) are cut hard.
    The request history keeps growing and tool-call ids change, so only a hash of the last turns catches it.
    """
    from agent_alert import AgentAlert

    print("=" * 65)
    print("RUNNING TEST 4: sha256 History Loop Detection (TokenGuard)")
    print("=" * 65)

    alerts = []
    AgentAlert.add_hook(lambda event, payload: alerts.append(event))
    try:
        guard = TokenGuard(max_budget_usd=10.0, max_repeats=3)
        halted_at = None
        for step in range(1, 6):
            print(f"\n[Agent Action #{step}] re-sending {2 * step + 1} messages to the model...")
            if not guard.track_action(1200, 250, messages=_agent_history(step)):
                halted_at = step
                print(f"❌ [Step {step}] chain cut by TokenGuard.")
                break
            print(f"✅ [Step {step}] ok")
            time.sleep(0.2)
        assert halted_at == 3, halted_at
        assert alerts == ["loop_cut"], alerts
        assert guard.track_action(1200, 250, messages=_agent_history(9)) is False, "chain must stay cut"
        guard.resume()
        assert guard.track_action(1200, 250, messages=_agent_history(10, command="ls")) is True
        print(f"\n✅ [TEST RESULT] Loop cut at step {halted_at}, alert sent, resume() re-armed the guard.")
    finally:
        AgentAlert.clear_hooks()


def run_near_duplicate_and_cycle_test():
    """
    Simulation Test: 'almost identical' turns (only timings / ids differ), A-B-A-B-A-B cycles are caught,
    while an agent that really makes progress (different errors) is left alone.
    """
    from agent_alert import AgentAlert

    print("=" * 65)
    print("RUNNING TEST 5: Near-duplicate & Cycle Detection (TokenGuard)")
    print("=" * 65)

    original_print = AgentAlert.trigger_loop_hard_alert
    AgentAlert.trigger_loop_hard_alert = staticmethod(lambda *a, **k: False)   # keep the output short
    try:
        g = TokenGuard(max_budget_usd=10.0, max_repeats=3)
        outcome = [g.track_action(10, 10, messages=_agent_history(1, result=f"3 failed in {t}.{t}s (run 0x{t:08x})"))
                   for t in (11, 12, 13)]
        assert outcome == [True, True, False], outcome
        print("✅ near-duplicate turns (different timings/ids) caught on the 3rd repeat")

        g = TokenGuard(max_budget_usd=10.0, max_repeats=3)
        results = [g.track_action(10, 10, messages=_agent_history(1, command=c)) for c in ("a", "b") * 3]
        assert results == [True] * 5 + [False], results
        print("✅ A,B,A,B,A,B cycle caught on the 6th request")

        g = TokenGuard(max_budget_usd=10.0, max_repeats=3)
        progress = [g.track_action(10, 10, messages=_agent_history(1, result=f"FAILED test_{name}")) for name in ("a", "b", "c", "d", "e")]
        assert all(progress), progress
        print("✅ real progress (new errors) is never flagged")
        print("✅ [TEST RESULT] Near-duplicate, cycle and no-false-positive checks passed.")
    finally:
        AgentAlert.trigger_loop_hard_alert = original_print


if __name__ == "__main__":
    run_loop_test()
    run_history_hash_loop_test()
    run_near_duplicate_and_cycle_test()
