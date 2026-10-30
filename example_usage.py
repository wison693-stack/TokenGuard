"""
Minimal integration example. Replace `fake_model_call` with your real SDK call.

    python example_usage.py
"""
from agent_alert import AgentAlert
from guard import TokenGuard
from redis_store import InMemoryBudgetStore            # swap for RedisBudgetStore(url=...) in production
from stream_guard import StreamAbortedError


def fake_model_call(messages):
    """Pretend to be an LLM SDK: returns streamed events and a final usage object."""
    text = "Running the tests again. "
    events = [{"choices": [{"delta": {"content": word + " "}, "finish_reason": None}]} for word in text.split()]
    usage = {"input_tokens": 900, "output_tokens": len(events), "cache_read_input_tokens": 12000}
    return events, usage


def main():
    # 1. Create the guard. Optional: persistent budget shared by all instances.
    guard = TokenGuard(max_budget_usd=2.0, max_repeats=3)
    guard.use_store(InMemoryBudgetStore(), session_id="demo-user")

    # 2. Optional: get alerts in Slack / a webhook / your logs.
    AgentAlert.add_hook(lambda event, payload: print(f"[hook] {event}: {payload}"))

    messages = [{"role": "user", "content": "Fix the failing test."}]

    for step in range(1, 6):
        # 3. Streaming: check the prompt before sending, then wrap the stream.
        if not guard.precheck_request(input_tokens=900):
            print("Budget would be exceeded - request not sent.")
            break
        events, usage = fake_model_call(messages)
        try:
            for event in guard.guard_stream(events, input_tokens=900, interactive=False):
                pass                                      # render(event) in a real app
        except StreamAbortedError as exc:
            print("Stream cut:", exc.reason, exc.detail)
            break

        # 4. Non-streaming / end-of-turn accounting with cache metadata + loop detection.
        messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{step}", "name": "bash",
                                                            "input": {"cmd": "pytest -x"}}]})
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{step}",
                                                      "content": "FAILED test_core.py::test_x"}]})
        if not guard.track_usage(usage, messages=messages):
            print(f"Agent halted at step {step}: {guard.halt_reason or 'budget'}")
            break
        print(f"step {step} ok - spent ${guard.current_cost:.4f} of ${guard.max_budget:.2f}")


if __name__ == "__main__":
    main()
