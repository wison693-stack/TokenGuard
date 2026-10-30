# TokenGuard 🛡️

**Safety middleware for Claude Code and autonomous AI agents.**
TokenGuard sits between your agent and the LLM API and stops the two things that quietly burn money:
**runaway API spend** and **infinite agent loops**. Zero required dependencies, pure Python 3.8+.

```text
 agent ──► TokenGuard ──► LLM API
            │  cache-aware cost tracking     │  streaming abort
            │  sha256 loop detection         │  Redis-backed shared limits
            └─► AgentAlert: bell + banner + hooks (Slack / webhook / logs)
```

---

## Table of contents

1. [The problem](#-the-problem)
2. [Features](#-features)
3. [Installation](#-installation)
4. [Quick start](#-quick-start)
5. [Running the tests](#-running-the-tests)
6. [Usage guide](#-usage-guide)
   - [Budget & loop tracking](#1-budget--loop-tracking-track_action)
   - [Prompt caching aware pricing](#2-prompt-caching-aware-pricing)
   - [Streaming (SSE)](#3-streaming-sse)
   - [Persistent / shared limits (Redis)](#4-persistent--shared-limits-redis)
   - [Loop detection](#5-loop-detection)
   - [Alerts and hooks](#6-alerts-and-hooks)
7. [API reference](#-api-reference)
8. [Project layout](#-project-layout)
9. [How it works](#-how-it-works)
10. [Limitations & notes](#-limitations--notes)
11. [FAQ](#-faq)
12. [Contributing](#-contributing)

---

## 🎯 The problem

When AI CLI agents (Claude Code and friends) run fully autonomously, three things go wrong:

1. **Infinite execution loops.** The agent keeps re-running a failing command (`pytest`, `npm test`, a syntax fix) and
   re-sends nearly the same turn hundreds of times, burning thousands of tokens without progress.
2. **Uncontrolled API costs.** High-context loops can burn $50+ in minutes before anyone notices.
3. **Blind spots in modern setups.** Streamed (SSE) answers, prompt-cache discounts and multi-instance deployments
   make naive "tokens x price" accounting wrong or useless.

## 🚀 Features

| Feature | What it does |
|---|---|
| **Real-time cost tracking** | Computes input/output spend per call; default pricing is $3/M input, $15/M output (configurable). |
| **Prompt-caching aware** | Reads `cache_read_input_tokens`, `cache_creation_input_tokens` (Anthropic) and `cached_tokens` (OpenAI) so cached prompts are billed at their real, discounted price. |
| **Streaming / SSE support** | Wraps any streaming response, forwards chunks instantly, counts tokens live, aborts the stream on budget overrun, output loops or a token cap. |
| **Persistent shared limits** | Spend, limit and loop history live in Redis (atomic Lua + `INCRBY`), so restarts and multiple pods share one budget. |
| **Loop detection** | `sha256` fingerprints of the agent's last turns: exact repeats, near-duplicates, A-B-A-B cycles, optional embedding similarity. |
| **AgentAlert** | Terminal bell + highlighted banner, human-in-the-loop `[y/N]` for budget extensions, hard non-interactive cut for loops, hooks for Slack/webhooks. |

---

## 📦 Installation

TokenGuard is a small set of Python modules - copy the folder into your project or clone the repo.

```bash
git clone https://github.com/<your-user>/token_guard.git
cd token_guard
python --version        # 3.8 or newer
```

Optional extras:

```bash
pip install redis       # only if you use RedisBudgetStore (persistent / shared limits)
pip install tiktoken    # optional: more accurate pre-request token estimates (falls back to chars/4)
```

Everything else is standard library.

## ⚡ Quick start

```python
from guard import TokenGuard

guard = TokenGuard(max_budget_usd=2.0, max_repeats=3)

# after every model call in your agent loop:
ok = guard.track_action(
    input_tokens=900,                       # fresh (uncached) input tokens
    output_tokens=450,
    command_signature="pytest tests/ -x",   # what the agent is doing
    cache_read_input_tokens=12_000,         # from the response metadata
)
if not ok:
    raise SystemExit("TokenGuard halted the agent")
```

Or feed it the provider's `usage` object directly (Anthropic, OpenAI Chat, OpenAI Responses):

```python
ok = guard.track_usage(response.usage, messages=messages)
```

See [`example_usage.py`](example_usage.py) for a complete runnable integration (`python example_usage.py`).

---

## 🧪 Running the tests

All commands are run from the project folder. No setup is needed.

```bash
python run_all_tests.py        # everything, in order, with a final summary

# or one at a time
python test_loop.py            # tests 1, 4, 5 - loop protection
python budget_test.py          # tests 2, 3    - budget protection, prompt caching
python test_streaming.py       # test 6        - streaming / SSE
python test_persistence.py     # test 7        - persistent & shared limits
```

| # | Test | What it proves |
|---|---|---|
| 1 | Infinite loop protection | Classic repeated command is stopped on the 3rd repeat. |
| 2 | API budget protection | An expensive agent is stopped when it crosses the limit. |
| 3 | Prompt-caching aware budget | Naive pricing kills a healthy cached agent; cache-aware pricing does not. |
| 4 | sha256 history loop detection | A loop is caught even though the history grows and tool-call ids change. |
| 5 | Near-duplicates & cycles | Noise-only differences and A-B-A-B patterns are caught; real progress is not flagged. |
| 6 | Streaming / SSE | Chunks pass through untouched; runaway output, budget overrun and token cap abort the stream. |
| 7 | Persistent limits | Two "pods" share one budget, a restart keeps the spend, a store outage does not crash the agent. |

Tests 1 and 2 are the classic interactive demos: when TokenGuard trips it asks `Proceed execution? [y/N]`.
Type `n` (or press Enter). When there is no interactive stdin (CI, pipes) the prompt is answered "no" automatically.

**Testing against a real Redis** (runs the real Lua scripts):

```bash
docker run -d --name tg-redis -p 6379:6379 redis:7
pip install redis
REDIS_URL=redis://localhost:6379/0 python test_persistence.py        # Linux / macOS
set REDIS_URL=redis://localhost:6379/0 && python test_persistence.py # Windows cmd
```

---

## 📘 Usage guide

### 1. Budget & loop tracking (`track_action`)

`track_action()` is the one call you put in your agent loop. It adds the cost of the step, checks the budget and
(if you pass `command_signature` and/or `messages`) checks for loops. It returns `True` if it is safe to continue
and `False` if execution must stop.

```python
guard = TokenGuard(max_budget_usd=5.0, max_repeats=3)

while agent_has_work():
    response = call_model(messages)
    if not guard.track_action(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        command_signature=current_command,
        cache_read_input_tokens=response.usage.cache_read_input_tokens,
        cache_creation_input_tokens=response.usage.cache_creation_input_tokens,
        messages=messages,
    ):
        break
```

When the budget is reached, `AgentAlert` shows a banner and asks whether to grant an extra **$1.00**. Answering `y`
raises the limit (in the shared store too) and continues; anything else halts.

### 2. Prompt caching aware pricing

Providers bill cached prompt tokens differently, so "total tokens x price" overstates the cost of cached agents.

| Provider | Field | Meaning | Default billing |
|---|---|---|---|
| Anthropic | `input_tokens` | fresh tokens only (cache excluded) | 1x input price |
| Anthropic | `cache_read_input_tokens` | served from cache | 0.1x |
| Anthropic | `cache_creation_input_tokens` | written to cache | 1.25x (5 min TTL), 2x (1 h TTL) |
| OpenAI | `prompt_tokens` / `input_tokens` | **includes** cached tokens | 1x for the fresh part |
| OpenAI | `prompt_tokens_details.cached_tokens` | served from cache | 0.5x |

`track_usage()` understands all of these shapes and subtracts the cached part for OpenAI automatically.
**Multipliers differ per model and change over time**, so set the ones for your model:

```python
from cache_pricing import PricingTable

guard.set_pricing("openai")                                    # provider defaults
guard.set_pricing(PricingTable(input_per_mtok=3.0, output_per_mtok=15.0,
                               cache_read_multiplier=0.05,     # e.g. a model with a 95% cache discount
                               cache_write_5m_multiplier=1.25,
                               cache_write_1h_multiplier=2.0))
```

Without any cache counters the result equals the classic `$3/M input + $15/M output` formula.

### 3. Streaming (SSE)

Wrap the stream; chunks are forwarded **unchanged and instantly** (no buffering - the typing effect is untouched).

```python
from stream_guard import StreamAbortedError

guard = TokenGuard(max_budget_usd=2.0)

if guard.precheck_request(messages=messages):                 # measure the prompt BEFORE sending it
    stream = client.messages.create(model=..., messages=messages, stream=True)
    try:
        for event in guard.guard_stream(stream, input_tokens=900, max_output_tokens=4000):
            render(event)
    except StreamAbortedError as exc:
        print("cut:", exc.reason, exc.detail)                 # exc.partial_text has what was generated
```

TokenGuard aborts the stream (closes the upstream connection so the provider stops generating, fires an alert,
bills what was already generated) when:

| `exc.reason` | Trigger |
|---|---|
| `budget` | committed spend + projected cost of this call reaches the limit |
| `generation_loop` | the model repeats the same text again and again (e.g. "Let me try again." x N) |
| `max_output_tokens` | your hard cap on generated tokens is reached |
| `agent_loop` | the request itself is a repeat of previous requests (when you pass `messages=` / `command_signature=`) |

Options of `guard_stream(...)`:

| Option | Default | Meaning |
|---|---|---|
| `input_tokens` | 0 | fresh input tokens of the request (provider usage events can only raise it) |
| `cache_read_input_tokens`, `cache_creation_input_tokens` | 0 | known cache counters of the request |
| `provider` | auto | `"anthropic"` / `"openai"`, picks the pricing profile |
| `max_output_tokens` | none | hard cap on output tokens |
| `interactive` | `True` | ask `[y/N]` for more budget mid-stream. **Use `False` in servers** (no stdin). |
| `raise_on_abort` | `True` | `False` = end the iteration silently, inspect `sg.aborted` / `sg.abort_reason` |
| `parse_sse` | `False` | the stream yields raw `data: {...}` lines instead of SDK objects |
| `detect_repeats`, `repeat_min_span`, `repeat_min_repeats` | on, 240, 6 | tuning of the generation-loop detector |

Supported chunk shapes: OpenAI Chat Completions, OpenAI Responses events, Anthropic Messages events
(`message_start`, `content_block_delta`, `message_delta`), plain strings, raw SSE lines. Sync (`for`) and async
(`async for`) streams both work. Token counts come from the provider's usage events when present and from a
`chars/4` estimate (or `tiktoken` if installed) while streaming; the final provider usage replaces the estimate.

### 4. Persistent / shared limits (Redis)

By default state lives in the Python process, so it is lost on restart and is per-instance. Put it in a store:

```python
from redis_store import RedisBudgetStore

guard = TokenGuard(max_budget_usd=5.0)
guard.use_store(RedisBudgetStore(url="redis://localhost:6379/0"), session_id="user-42")
# equivalent: TokenGuard(max_budget_usd=5.0, store=RedisBudgetStore(...), session_id="user-42")
```

- Every instance that uses the same `session_id` shares **one** budget; a restart resumes where it stopped.
- Updates are **atomic**: one Lua script does `INCRBY` + limit read in a single step. Money is stored as integer
  nano-dollars, so there is no float drift.
- The first instance to register a session defines its limit; extra allowances granted by the developer (+$1.00) are
  stored and seen by everyone.
- Loop-detection history is shared too, so a loop split across several pods is still caught.
- Keys use a Redis-Cluster hash tag (`tokenguard:{session_id}:...`) so multi-key scripts work on a cluster.
  Default TTL is 24 h, refreshed on every write (`RedisBudgetStore(ttl=0)` keeps keys forever).
- If Redis is unreachable the agent **keeps running** on local counters, a warning is printed once, and the offline
  spend is flushed when Redis is back.
- No Redis? `InMemoryBudgetStore()` has the identical interface (useful for tests and single-process apps).
  You can also implement `BudgetStore` for a SQL database.

### 5. Loop detection

Agents re-send a growing history every step, so hashing the whole conversation never repeats. TokenGuard
fingerprints only the **last `window` messages** (default 2: the assistant's action and the tool result) with
`sha256`, ignoring random tool-call ids, and inspects the sequence of fingerprints:

| `verdict.kind` | Detects |
|---|---|
| `exact` | the same turn `max_repeats` times in a row (default **3**) |
| `near-duplicate` | same after normalising numbers, hex/uuid ids, timestamps and ANSI codes ("failed in 1.2s" vs "failed in 3.4s") |
| `cycle` / `near-cycle` | repeating patterns A,B,A,B,A,B (period up to `max_cycle_period`, default 3) |
| `semantic` | optional: cosine similarity of your own embeddings above a threshold |

```python
guard.configure_loop_detection(
    max_repeats=3,
    window=2,                     # how many trailing messages form one "turn"
    normalize=True,               # near-duplicate detection (turn off if your agent legitimately polls)
    max_cycle_period=3,
    embed_fn=my_embedding_fn,     # optional: text -> list[float]
    similarity_threshold=0.95,
)
```

On detection TokenGuard **cuts the chain hard**: it fires an alert, does not ask for an override, and every later
`track_action()` returns `False` until you call `guard.resume()`. Prefer the old "ask the developer" behaviour?
Create the guard with `TokenGuard(hard_loop_cut=False)`.

You can also inspect without side effects: `verdict = guard.check_loop(messages)`
(`verdict.is_loop`, `.kind`, `.repeats`, `.period`).

### 6. Alerts and hooks

`AgentAlert` rings the terminal bell and prints a highlighted banner. To get alerts elsewhere, register hooks:

```python
from agent_alert import AgentAlert

def to_slack(event, payload):          # event: "loop_cut" | "stream_abort"
    post_to_slack(f"TokenGuard {event}: {payload}")

AgentAlert.add_hook(to_slack)          # hooks never raise into the guard
```

---

## 📚 API reference

### `TokenGuard(max_budget_usd=2.0, max_repeats=3, hard_loop_cut=True, store=None, session_id="default", pricing=None)`

| Method | Description |
|---|---|
| `track_action(input_tokens, output_tokens, command_signature=None, *, cache_read_input_tokens=0, cache_creation_input_tokens=0, cache_creation_1h_input_tokens=0, provider=None, messages=None) -> bool` | Add cost, check budget, check loops. `False` = halt. |
| `track_usage(usage, command_signature=None, messages=None, provider=None) -> bool` | Same, from a provider `usage` object/dict. |
| `compute_cost(...) -> float` | Cache-aware cost in USD, no side effects. |
| `set_pricing(table_or_provider)` | Custom `PricingTable` or `"anthropic"` / `"openai"`. |
| `use_store(store, session_id)` | Persistent / shared limits. |
| `precheck_request(messages=None, command_signature=None, input_tokens=None, ...) -> bool` | Before sending: will this prompt fit the budget / is it a loop? |
| `guard_stream(stream, **options) -> StreamGuard` | Wrap a streaming response. |
| `configure_loop_detection(**options)` | Tune the loop detector. |
| `check_loop(messages=None, command_signature=None) -> LoopVerdict` | Record a request, get a verdict, no alert. |
| `resume()` | Re-arm after a hard loop cut. |

Attributes: `current_cost`, `max_budget`, `halted`, `halt_reason`, `command_history`.
`track_action_ext` is kept as an alias of `track_action`.

### `StreamGuard` (returned by `guard_stream`)

`text`, `output_tokens`, `usage`, `cost`, `finished`, `aborted`, `abort_reason`, `abort_detail`,
`safe_to_continue`, `projected_cost()`, `close()`; usable as a context manager and with `for` / `async for`.

### Modules

| Module | Purpose |
|---|---|
| `cache_pricing.py` | `PricingTable`, `Usage`, `normalize_usage()` |
| `stream_guard.py` | `StreamGuard`, `StreamAbortedError`, `extract_chunk()`, `estimate_tokens()` |
| `loop_detector.py` | `LoopDetector`, `LoopVerdict` |
| `redis_store.py` | `BudgetStore`, `RedisBudgetStore`, `InMemoryBudgetStore` |
| `agent_alert.py` | `AgentAlert` (banners, prompts, hooks) |

## 🗂 Project layout

```text
token_guard/
├── guard.py              TokenGuard - the main entry point
├── agent_alert.py        terminal alerts, human-in-the-loop prompts, hooks
├── cache_pricing.py      prompt-caching aware pricing and usage normalisation
├── stream_guard.py       streaming / SSE wrapper with emergency abort
├── loop_detector.py      sha256 history fingerprints, near-duplicates, cycles, embeddings
├── redis_store.py        Redis (Lua + INCRBY) and in-memory stores
├── budget_test.py        tests 2-3: budget + prompt caching
├── test_loop.py          tests 1, 4, 5: loop protection
├── test_streaming.py     test 6: streaming
├── test_persistence.py   test 7: persistent / shared limits
├── run_all_tests.py      runs all tests
├── example_usage.py      runnable integration example
├── requirements.txt      optional dependencies
└── CHANGELOG.md
```

## ⚙️ How it works

1. **Cost.** `cost = fresh_in/1M x P_in + out/1M x P_out + cache_read/1M x P_in x 0.1 + cache_write/1M x P_in x 1.25 (or 2)`.
2. **Budget gate.** After adding the cost, if `current_cost >= max_budget` the developer is asked for +$1.00;
   no consent = halt.
3. **Loop gate.** Fingerprint the last turns -> push to a bounded history (local or Redis list, atomically) ->
   look for a repeating tail.
4. **Streaming.** For each chunk: parse text/usage -> update token estimate -> project `committed + current call` ->
   periodic repetition scan of the output tail -> forward chunk. On a violation: close upstream, alert, commit the
   partial cost exactly once, raise `StreamAbortedError`.

## ⚠️ Limitations & notes

- Live token counts during streaming are **estimates** until the provider sends its final usage; the final numbers
  replace them for billing accuracy.
- Budget checks mid-stream use the spend committed by this instance plus the current call. Spend from other instances
  is re-read from Redis at the start of each stream and on every commit, not on every chunk.
- The interactive prompt uses `input()`. In servers pass `interactive=False` to `guard_stream()` and register hooks.
- Default prices ($3/M in, $15/M out) are the Claude 3.5 Sonnet figures. **Set your own model's prices** with
  `set_pricing()`. Cache multipliers also vary per model and change over time.
- `normalize=True` can flag an agent that legitimately polls the same command while only numbers change
  (e.g. progress percentages). Disable it with `configure_loop_detection(normalize=False)` in that case.
- The Redis Lua scripts need Redis 2.6+. Cluster mode is supported through hash tags.
- Not a security boundary and not a replacement for provider-side spend limits; use both.

## ❓ FAQ

**Do I have to use Redis?** No. Without a store everything is in memory. Redis is only for persistence and
multi-instance sharing.

**Will the guard slow my streaming UI?** No. Chunks are yielded to your code immediately; the guard only does a
few cheap checks per chunk.

**An agent was cut by mistake. What now?** Call `guard.resume()`, or raise `max_repeats`, or set
`normalize=False`. Use `hard_loop_cut=False` if you prefer to be asked.

**Does it work with providers other than Claude?** Yes - pricing is configurable and usage objects from
OpenAI (Chat + Responses) are understood.

## 🤝 Contributing

Issues and pull requests are welcome. Please run `python run_all_tests.py` before opening a PR and add a test for
new behaviour.
