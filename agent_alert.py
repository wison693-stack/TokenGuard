import sys

class AgentAlert:
    """
    Terminal-based Agent Alert system for human-in-the-loop intervention.
    Outputs highlighted alerts, plays a system bell sound, and pauses 
    execution pending developer input.
    """

    # ANSI Color Codes for Rich Terminal Output
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def trigger_budget_alert(current_cost: float, limit: float) -> bool:
        """
        Triggers a visual and acoustic alert when the API budget threshold is breached.
        Returns True if the user grants budget extension, False to halt.
        """
        # System bell sound trigger
        sys.stdout.write("\a")
        sys.stdout.flush()

        banner = f"""
{AgentAlert.RED}{AgentAlert.BOLD}=============================================================
[TokenGuard] ⚠️ AGENT BUDGET LIMIT BREACHED
============================================================={AgentAlert.RESET}
Current Session Cost : {AgentAlert.YELLOW}${current_cost:.4f}{AgentAlert.RESET}
Max Budget Threshold : ${limit:.2f}

Action Required: The AI agent has reached its monetary limit.
Do you want to grant an additional $1.00 allowance and continue execution?
"""
        print(banner)
        return AgentAlert._prompt_user()

    @staticmethod
    def trigger_loop_alert(command_signature: str, repeat_count: int) -> bool:
        """
        Triggers an alert when an infinite retry loop is detected.
        Returns True if the user forces continuation, False to halt.
        """
        # System bell sound trigger
        sys.stdout.write("\a")
        sys.stdout.flush()

        banner = f"""
{AgentAlert.RED}{AgentAlert.BOLD}=============================================================
[TokenGuard] 🔄 INFINITE LOOP DETECTED IN AGENT EXECUTION
============================================================={AgentAlert.RESET}
Repeated Command : {AgentAlert.YELLOW}{command_signature}{AgentAlert.RESET}
Repetition Count : {repeat_count} consecutive attempts without state change.

Action Required: The agent appears to be stuck in an error loop.
Do you want to override and force-continue execution?
"""
        print(banner)
        return AgentAlert._prompt_user()

    @staticmethod
    def _prompt_user() -> bool:
        """
        Pauses process execution and requests interactive developer input via standard stdin.
        """
        try:
            choice = input(f"{AgentAlert.BOLD}Proceed execution? [y/N]: {AgentAlert.RESET}").strip().lower()
            if choice == 'y':
                print(f"{AgentAlert.YELLOW}[TokenGuard] Execution resumed by user override.{AgentAlert.RESET}\n")
                return True
            else:
                print(f"{AgentAlert.RED}[TokenGuard] Execution aborted by developer.{AgentAlert.RESET}\n")
                return False
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C, or no interactive stdin (CI, service, pipe): fail safe and halt.
            print(f"\n{AgentAlert.RED}[TokenGuard] Execution aborted via interrupt / no interactive input.{AgentAlert.RESET}\n")
            return False

    # =====================================================================
    # Additive extensions (existing alerts above are unchanged)
    # =====================================================================
    _hooks = []   # optional callbacks fn(event: str, payload: dict) -> e.g. Slack / webhook / log

    @staticmethod
    def add_hook(fn):
        """Register a callback that receives every extended alert (never raises into the guard)."""
        AgentAlert._hooks.append(fn)

    @staticmethod
    def clear_hooks():
        AgentAlert._hooks.clear()

    @staticmethod
    def _emit(event: str, payload: dict):
        for fn in list(AgentAlert._hooks):
            try:
                fn(event, payload)
            except Exception:
                pass

    @staticmethod
    def trigger_loop_hard_alert(command_signature: str, repeat_count: int, kind: str = "exact", period: int = 1) -> bool:
        """
        Non-interactive alert used by the hard loop cut: bell + banner + hooks, NO override prompt.
        Always returns False (execution must be halted).
        """
        sys.stdout.write("\a")
        sys.stdout.flush()
        print(f"""
{AgentAlert.RED}{AgentAlert.BOLD}=============================================================
[TokenGuard] 🔄 AGENT LOOP CUT ({kind}, period={period})
============================================================={AgentAlert.RESET}
Repeated Request : {AgentAlert.YELLOW}{command_signature}{AgentAlert.RESET}
Repetition Count : {repeat_count} consecutive identical/near-identical requests.
The chain was terminated automatically. Call guard.resume() to re-arm it.
""")
        AgentAlert._emit("loop_cut", {"signature": command_signature, "repeats": repeat_count,
                                      "kind": kind, "period": period})
        return False

    @staticmethod
    def trigger_stream_abort_alert(reason: str, detail: str = "", output_tokens: int = 0, cost: float = 0.0) -> None:
        """Non-interactive alert fired when a streaming response is cut mid-flight."""
        sys.stdout.write("\a")
        sys.stdout.flush()
        print(f"""
{AgentAlert.RED}{AgentAlert.BOLD}=============================================================
[TokenGuard] ✂️ STREAM ABORTED: {reason}
============================================================={AgentAlert.RESET}
Detail           : {AgentAlert.YELLOW}{detail}{AgentAlert.RESET}
Output so far    : ~{output_tokens} tokens, ${cost:.4f}
""")
        AgentAlert._emit("stream_abort", {"reason": reason, "detail": detail,
                                          "output_tokens": output_tokens, "cost": cost})
