import time
from guard import TokenGuard

def run_budget_test():
    """
    Simulation Test: Detects and stops an agent when API cost exceeds the budget limit.
    """
    print("=" * 65)
    print("RUNNING TEST 2: API Budget Protection (TokenGuard)")
    print("=" * 65)

    # Low budget threshold set for quick testing ($0.05 limit)
    guard = TokenGuard(max_budget_usd=0.05, max_repeats=10)

    # Simulated heavy token context per step (~$0.06 total per step)
    heavy_input_tokens = 15_000  # ~$0.045
    heavy_output_tokens = 1_000  # ~$0.015

    for step in range(1, 5):
        print(f"\n[Agent Action #{step}] Processing large context codebase...")

        is_safe = guard.track_action(
            input_tokens=heavy_input_tokens,
            output_tokens=heavy_output_tokens,
            command_signature=f"refactor_module_v{step}.py"
        )

        if not is_safe:
            print(f"❌ [TEST RESULT] Agent execution halted at step {step} due to budget limit breach.")
            return

        print(f"✅ [Step {step}] Executed successfully. Current spent: ${guard.current_cost:.4f}")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Added: Prompt Caching aware budgeting (cache_read_input_tokens & friends)
# ---------------------------------------------------------------------------
def run_cache_budget_test():
    """
    Simulation Test: the same agent workload is billed twice - 'naively' (all input tokens at the
    full price, original track_action) and cache-aware (cache reads at ~0.1x, cache write at 1.25x,
    taken from the response metadata). The naive guard kills a healthy agent; the honest one does not.
    """
    from agent_alert import AgentAlert

    print("=" * 65)
    print("RUNNING TEST 3: Prompt Caching Aware Budget (TokenGuard)")
    print("=" * 65)

    # The terminal alert would block on input(); answer 'no' automatically for this simulation.
    original_prompt = AgentAlert._prompt_user
    AgentAlert._prompt_user = staticmethod(lambda: False)
    try:
        budget = 0.15
        # Typical agent turn: 14k tokens of stable context (system prompt + repo) + 1k fresh input, 1k output.
        # Step 1 writes the cache, steps 2+ read it. Fields mirror Anthropic's usage metadata.
        usage_per_step = [
            dict(input_tokens=1_000, output_tokens=1_000, cache_creation_input_tokens=14_000, cache_read_input_tokens=0),
            dict(input_tokens=1_000, output_tokens=1_000, cache_creation_input_tokens=0, cache_read_input_tokens=14_000),
            dict(input_tokens=1_000, output_tokens=1_000, cache_creation_input_tokens=0, cache_read_input_tokens=14_000),
            dict(input_tokens=1_000, output_tokens=1_000, cache_creation_input_tokens=0, cache_read_input_tokens=14_000),
        ]

        # --- A) naive accounting: every input token at the full $3/M price
        naive = TokenGuard(max_budget_usd=budget, max_repeats=10)
        naive_halt = None
        for step, u in enumerate(usage_per_step, 1):
            total_in = u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"]
            if not naive.track_action(total_in, u["output_tokens"], f"agent_turn_{step}"):
                naive_halt = step
                break
        print(f"\n[Naive]       halted at step {naive_halt}, reported spend ${naive.current_cost:.4f}")

        # --- B) cache-aware accounting from the response metadata
        honest = TokenGuard(max_budget_usd=budget, max_repeats=10)
        honest_halt = None
        for step, u in enumerate(usage_per_step, 1):
            if not honest.track_action(command_signature=f"agent_turn_{step}", **u):
                honest_halt = step
                break
            print(f"✅ [Step {step}] cache-aware spend: ${honest.current_cost:.4f}")
        print(f"[Cache-aware] halted at step {honest_halt}, real spend ${honest.current_cost:.4f}")

        assert naive_halt is not None, "naive accounting should have tripped the budget"
        assert honest_halt is None, "cache-aware accounting should let the healthy agent finish"
        assert abs(honest.current_cost - 0.1371) < 1e-9, honest.current_cost

        # --- C) raw provider usage objects are understood too
        g = TokenGuard(max_budget_usd=10.0, max_repeats=10)
        g.track_usage({"input_tokens": 1000, "output_tokens": 1000,
                       "cache_read_input_tokens": 14000, "cache_creation_input_tokens": 0})          # Anthropic
        anthropic_cost = g.current_cost
        g2 = TokenGuard(max_budget_usd=10.0, max_repeats=10)
        g2.track_usage({"prompt_tokens": 15000, "completion_tokens": 1000,
                        "prompt_tokens_details": {"cached_tokens": 14000}})                          # OpenAI (0.5x cached)
        print(f"[Usage objects] anthropic ${anthropic_cost:.4f} | openai ${g2.current_cost:.4f}")
        assert abs(anthropic_cost - 0.0222) < 1e-9
        assert abs(g2.current_cost - (0.003 + 14000 / 1e6 * 3.0 * 0.5 + 0.015)) < 1e-9

        print(f"✅ [TEST RESULT] Naive guard would have stopped the agent at step {naive_halt}; "
              f"cache-aware guard let it finish for ${honest.current_cost:.4f}.")
    finally:
        AgentAlert._prompt_user = original_prompt


if __name__ == "__main__":
    run_budget_test()
    run_cache_budget_test()
