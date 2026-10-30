"""
Runs every TokenGuard test scenario in sequence.

    python run_all_tests.py

Tests 1 and 2 are the classic interactive demos: when the guard trips they ask
"Proceed execution? [y/N]". Answer `n` (or just press Enter). When stdin is not a
terminal (CI, pipes) the prompt is answered with "no" automatically.
Set REDIS_URL=redis://localhost:6379/0 to also run the persistence checks against a real Redis.
"""
import sys

from budget_test import run_budget_test, run_cache_budget_test
from test_loop import run_history_hash_loop_test, run_loop_test, run_near_duplicate_and_cycle_test
from test_persistence import run_persistence_test
from test_streaming import run_streaming_test

TESTS = [
    run_loop_test,                      # 1. classic loop protection (interactive alert)
    run_budget_test,                    # 2. classic budget protection (interactive alert)
    run_cache_budget_test,              # 3. prompt-caching aware budget
    run_history_hash_loop_test,         # 4. sha256 history loop detection
    run_near_duplicate_and_cycle_test,  # 5. near-duplicates, cycles, no false positives
    run_streaming_test,                 # 6. streaming / SSE
    run_persistence_test,               # 7. persistent / shared limits
]

if __name__ == "__main__":
    failed = []
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failed.append(test.__name__)
            print(f"❌ {test.__name__} FAILED: {exc!r}")
        print()
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print("🎉 All TokenGuard tests passed.")
