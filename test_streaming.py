import asyncio
import json
import time
from types import SimpleNamespace as NS

from agent_alert import AgentAlert
from guard import TokenGuard
from stream_guard import StreamAbortedError, extract_chunk


class FakeUpstream:
    """Stand-in for an SDK stream: iterable + close(), remembers whether it was closed."""

    def __init__(self, chunks, limit=None):
        self.chunks, self.limit, self.closed, self.sent = chunks, limit, False, 0

    def __iter__(self):
        it = iter(self.chunks() if callable(self.chunks) else self.chunks)
        for chunk in it:
            if self.closed:
                return
            self.sent += 1
            yield chunk

    def close(self):
        self.closed = True


def openai_chunk(text, usage=None):
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}], "usage": usage}


def anthropic_events(words, in_tokens=1000, cache_read=14000, out_tokens=None):
    yield NS(type="message_start", message=NS(usage=NS(input_tokens=in_tokens, output_tokens=1,
                                                       cache_read_input_tokens=cache_read, cache_creation_input_tokens=0)))
    for w in words:
        yield NS(type="content_block_delta", delta=NS(type="text_delta", text=w))
    yield NS(type="message_delta", usage=NS(output_tokens=out_tokens or len(words)))
    yield NS(type="message_stop")


def run_streaming_test():
    print("=" * 65)
    print("RUNNING TEST 6: Streaming / SSE Protection (TokenGuard)")
    print("=" * 65)
    original_prompt = AgentAlert._prompt_user
    AgentAlert._prompt_user = staticmethod(lambda: False)
    try:
        # 1) healthy OpenAI-style stream: chunks pass through untouched, real usage is billed
        guard = TokenGuard(max_budget_usd=10.0)
        chunks = [openai_chunk("Hello "), openai_chunk("world"),
                  {"choices": [], "usage": {"prompt_tokens": 15000, "completion_tokens": 40,
                                             "prompt_tokens_details": {"cached_tokens": 14000}}}]
        sg = guard.guard_stream(FakeUpstream(chunks), provider="openai")
        got = list(sg)
        assert got == chunks, "chunks must be forwarded unchanged"
        assert sg.text == "Hello world" and sg.finished and not sg.aborted
        expected = 1000 / 1e6 * 3 + 14000 / 1e6 * 3 * 0.5 + 40 / 1e6 * 15
        assert abs(guard.current_cost - expected) < 1e-9, guard.current_cost
        print("✅ healthy stream forwarded unchanged, cache-aware cost committed: $%.6f" % guard.current_cost)

        # 2) Anthropic events with cache metadata
        guard = TokenGuard(max_budget_usd=10.0)
        sg = guard.guard_stream(anthropic_events(["a ", "b ", "c"], out_tokens=3))
        list(sg)
        assert abs(guard.current_cost - (1000 / 1e6 * 3 + 14000 / 1e6 * 0.3 + 3 / 1e6 * 15)) < 1e-9, guard.current_cost
        print("✅ Anthropic message_start/message_delta usage parsed (cache_read_input_tokens honoured)")

        # 3) raw SSE lines
        guard = TokenGuard(max_budget_usd=10.0)
        lines = ["event: message\n", "data: " + json.dumps(openai_chunk("Hi ")) + "\n\n",
                 "data: " + json.dumps(openai_chunk("there")) + "\n\n", "data: [DONE]\n\n"]
        sg = guard.guard_stream(FakeUpstream(lines), parse_sse=True)
        list(sg)
        assert sg.text == "Hi there", sg.text
        print("✅ raw SSE lines parsed (data:/[DONE])")

        # 4) runaway generation: model repeats itself forever -> aborted + upstream closed
        guard = TokenGuard(max_budget_usd=100.0)
        up = FakeUpstream(lambda: (openai_chunk("Let me try that again. ") for _ in iter(int, 1)))
        sg = guard.guard_stream(up, input_tokens=500)
        try:
            for _ in sg:
                pass
            raise AssertionError("runaway stream was not aborted")
        except StreamAbortedError as exc:
            assert exc.reason == "generation_loop", exc.reason
        assert up.closed and sg.aborted and up.sent < 100, (up.closed, up.sent)
        assert guard.current_cost > 0, "partial cost must still be committed"
        print("✅ infinite loop in output aborted after %d chunks, upstream closed, $%.5f committed" % (up.sent, guard.current_cost))

        # 5) budget breach mid-stream (non-interactive server mode)
        guard = TokenGuard(max_budget_usd=0.002)
        up = FakeUpstream(lambda: (openai_chunk("word%d " % i) for i in range(100000)))
        sg = guard.guard_stream(up, input_tokens=100, interactive=False, detect_repeats=False)
        try:
            list(sg)
            raise AssertionError("budget breach not detected")
        except StreamAbortedError as exc:
            assert exc.reason == "budget", exc.reason
        assert up.closed and up.sent < 1000
        print("✅ budget overrun aborted mid-stream after %d chunks" % up.sent)

        # 6) hard cap on output tokens
        guard = TokenGuard(max_budget_usd=100.0)
        up = FakeUpstream(lambda: (openai_chunk("x%d " % i) for i in range(100000)))
        sg = guard.guard_stream(up, max_output_tokens=50, detect_repeats=False)
        try:
            list(sg)
        except StreamAbortedError as exc:
            assert exc.reason == "max_output_tokens"
        assert up.closed
        print("✅ max_output_tokens cap enforced")

        # 7) consumer breaks early: cost is still committed once
        guard = TokenGuard(max_budget_usd=10.0)
        sg = guard.guard_stream(FakeUpstream([openai_chunk("The quick brown fox jumps over the lazy dog. " * 4 + str(i)) for i in range(5)]), input_tokens=1000)
        for _ in sg:
            break
        sg.close()
        assert guard.current_cost > 0
        print("✅ early `break` still bills what was generated")

        # 8) pre-request check: huge prompt vs. remaining budget
        guard = TokenGuard(max_budget_usd=0.01)
        assert guard.precheck_request(input_tokens=500_000) is False
        assert TokenGuard(max_budget_usd=5.0).precheck_request(messages=[{"role": "user", "content": "hi"}]) is True
        print("✅ precheck_request blocks a prompt that cannot fit the budget before it is sent")

        # 9) async streams
        async def agen():
            for _ in range(100000):
                yield openai_chunk("again and again. ")

        async def run_async():
            guard = TokenGuard(max_budget_usd=100.0)
            sg = guard.guard_stream(agen(), input_tokens=10)
            try:
                async for _ in sg:
                    pass
            except StreamAbortedError as exc:
                return exc.reason
            return None

        assert asyncio.run(run_async()) == "generation_loop"
        print("✅ async streams supported")

        # 10) a looping AGENT request is refused before streaming even starts
        guard = TokenGuard(max_budget_usd=10.0, max_repeats=3)
        req = [{"role": "user", "content": "same prompt"}]
        refused = None
        for i in range(4):
            try:
                list(guard.guard_stream(FakeUpstream([openai_chunk("ok")]), messages=req))
            except StreamAbortedError as exc:
                refused = (i, exc.reason)
                break
        assert refused == (2, "agent_loop"), refused
        print("✅ 3rd identical streaming request refused (agent_loop)")
        print("✅ [TEST RESULT] Streaming protection checks passed.")
    finally:
        AgentAlert._prompt_user = original_prompt


if __name__ == "__main__":
    run_streaming_test()
