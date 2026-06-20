"""Unit coverage for the background-review cost-control helpers.

These cover the deterministic logic added to bound and cheapen the post-turn
self-improvement fork:

  ① aux-model routing       — _resolve_review_runtime
  ② tail+digest replay      — _build_review_history / _digest_old_messages
  ③ token-budget guard      — _build_review_history threshold behaviour
  ④ adaptive cadence        — effective_skill_interval

All pure-function / config-driven; no live model calls.
"""

from unittest.mock import patch

from agent import background_review as br


# ---------------------------------------------------------------------------
# ② + ③  context digest / budget guard
# ---------------------------------------------------------------------------

def _msg(role, content, tool_calls=None):
    m = {"role": role, "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


def test_rough_token_count_counts_content_and_tool_args():
    msgs = [
        _msg("user", "a" * 275),  # 100 tokens
        _msg("assistant", "", tool_calls=[
            {"function": {"name": "terminal", "arguments": "b" * 275}}
        ]),  # 100 tokens
    ]
    assert br._rough_token_count(msgs) == 200


def test_build_history_under_budget_returns_full_snapshot():
    msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
    history, meta = br._build_review_history(msgs, max_context_tokens=48000, digest_tail_messages=24)
    assert history == msgs
    assert meta["digested"] is False
    assert meta["replayed_messages"] == 2


def test_build_history_guard_disabled_returns_full_even_when_huge():
    # max_context_tokens=0 disables the guard entirely.
    msgs = [_msg("user", "x" * 1_000_000)]
    history, meta = br._build_review_history(msgs, max_context_tokens=0, digest_tail_messages=4)
    assert history == msgs
    assert meta["digested"] is False


def test_build_history_over_budget_digests_old_keeps_tail_verbatim():
    # 60 user/assistant turns, each big enough to blow a small budget.
    msgs = []
    for i in range(60):
        msgs.append(_msg("user", f"u{i} " + "x" * 300))
        msgs.append(_msg("assistant", f"a{i} " + "y" * 300))
    history, meta = br._build_review_history(
        msgs, max_context_tokens=2000, digest_tail_messages=10
    )
    assert meta["digested"] is True
    # First message is the synthetic digest (user role to preserve alternation).
    assert history[0]["role"] == "user"
    assert history[0]["content"].startswith("[Earlier conversation digest")
    # Tail preserved verbatim — last real message survives untouched.
    assert history[-1] == msgs[-1]
    # Replayed tokens are strictly smaller than the full snapshot.
    assert meta["replayed_tokens"] < meta["full_tokens"]
    # 1 digest + 10 tail.
    assert meta["replayed_messages"] == 11


def test_build_history_does_not_open_tail_on_a_tool_message():
    # Construct a tail boundary that would land on a `tool` message; the guard
    # must walk back so the tool result isn't orphaned from its assistant call.
    msgs = []
    for i in range(40):
        msgs.append(_msg("user", "u" + "x" * 300))
        msgs.append(_msg("assistant", "", tool_calls=[
            {"function": {"name": "terminal", "arguments": "z" * 300}}
        ]))
        msgs.append({"role": "tool", "content": "result " + "w" * 300})
    history, meta = br._build_review_history(
        msgs, max_context_tokens=1500, digest_tail_messages=2
    )
    assert meta["digested"] is True
    # The verbatim tail (everything after the digest msg) must not begin on a
    # bare tool message.
    assert history[1]["role"] != "tool"


def test_digest_old_messages_captures_tool_names():
    old = [
        _msg("user", "do the thing"),
        _msg("assistant", "", tool_calls=[
            {"function": {"name": "skill_view", "arguments": "{}"}},
            {"function": {"name": "patch", "arguments": "{}"}},
        ]),
    ]
    digest = br._digest_old_messages(old)
    assert "USER: do the thing" in digest
    assert "tools: skill_view, patch" in digest


def test_digest_foregrounds_signal_turns():
    # A correction buried in filler must be surfaced under the SIGNALS header,
    # not just truncated into the arc — this is the mechanism that lets the
    # digest IMPROVE capture rather than merely preserve it.
    old = [_msg("user", "filler question " + str(i)) for i in range(20)]
    old.append(_msg("user", "stop being so verbose, just give me the answer from now on"))
    old.append(_msg("assistant", "the fix was setting stdin=subprocess.DEVNULL on the bridge"))
    digest = br._digest_old_messages(old)
    assert "POTENTIAL LEARNING SIGNALS" in digest
    assert "stop being so verbose" in digest
    assert "stdin=subprocess.DEVNULL" in digest
    # Arc still present for context.
    assert "Conversation arc" in digest


def test_digest_no_signals_omits_signal_header():
    old = [_msg("user", "what's the capital of france"), _msg("assistant", "Paris.")]
    digest = br._digest_old_messages(old)
    assert "POTENTIAL LEARNING SIGNALS" not in digest
    assert "Conversation arc" in digest


def test_is_signal_turn_detects_corrections_and_facts():
    assert br._is_signal_turn(_msg("user", "stop doing that, it's too verbose"))
    assert br._is_signal_turn(_msg("user", "for context, I run a 3-node proxmox cluster"))
    assert br._is_signal_turn(_msg("user", "from now on always use tabs"))
    assert br._is_signal_turn(_msg("assistant", "the fix was a missing await"))
    # Plain Q&A is not a signal.
    assert not br._is_signal_turn(_msg("user", "what's the default postgres port"))
    assert not br._is_signal_turn(_msg("assistant", "It's 5432."))


# ---------------------------------------------------------------------------
# pre-pass (cheap-model signal extraction prepended to the digest)
# ---------------------------------------------------------------------------

def _fake_resp(text):
    from types import SimpleNamespace
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_prepass_brief_returned_and_labeled():
    old = [_msg("user", "for context I run proxmox"), _msg("assistant", "noted")]
    with patch("agent.auxiliary_client.call_llm",
               return_value=_fake_resp("FACT | user runs a 3-node proxmox cluster")):
        brief = br._extract_signal_brief(old)
    assert brief is not None
    assert "PRE-SCREENED LEARNING CANDIDATES" in brief
    assert "proxmox" in brief


def test_prepass_none_response_falls_back():
    old = [_msg("user", "what's 2+2"), _msg("assistant", "4")]
    with patch("agent.auxiliary_client.call_llm", return_value=_fake_resp("NONE")):
        assert br._extract_signal_brief(old) is None


def test_prepass_exception_falls_back_silently():
    old = [_msg("user", "for context I run proxmox"), _msg("assistant", "noted")]
    with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("boom")):
        assert br._extract_signal_brief(old) is None  # never raises


def test_build_history_prepass_prepends_brief_to_digest():
    msgs = []
    for i in range(60):
        msgs.append(_msg("user", f"u{i} " + "x" * 300))
        msgs.append(_msg("assistant", f"a{i} " + "y" * 300))
    with patch("agent.auxiliary_client.call_llm",
               return_value=_fake_resp("PREF | user wants terse answers")):
        history, meta = br._build_review_history(
            msgs, max_context_tokens=2000, digest_tail_messages=10,
            use_prepass=True,
        )
    assert meta["digested"] is True
    assert meta["prepass_used"] is True
    assert "PRE-SCREENED LEARNING CANDIDATES" in history[0]["content"]
    assert "user wants terse answers" in history[0]["content"]
    # Tail still verbatim.
    assert history[-1] == msgs[-1]


def test_build_history_prepass_off_by_default():
    msgs = []
    for i in range(60):
        msgs.append(_msg("user", f"u{i} " + "x" * 300))
        msgs.append(_msg("assistant", f"a{i} " + "y" * 300))
    # use_prepass defaults False → no call_llm, no brief.
    with patch("agent.auxiliary_client.call_llm", side_effect=AssertionError("should not be called")):
        history, meta = br._build_review_history(
            msgs, max_context_tokens=2000, digest_tail_messages=10,
        )
    assert meta["prepass_used"] is False
    assert "PRE-SCREENED" not in history[0]["content"]


# ---------------------------------------------------------------------------
# ④  adaptive cadence
# ---------------------------------------------------------------------------

def _cad(**over):
    base = {"skip_tool_free_turns": True, "adaptive_backoff": True, "adaptive_backoff_after": 3}
    base.update(over)
    return base


def test_effective_interval_unchanged_below_threshold():
    with patch.object(br, "_read_skill_cadence", return_value=_cad()):
        assert br.effective_skill_interval(10, noop_streak=0) == 10
        assert br.effective_skill_interval(10, noop_streak=2) == 10


def test_effective_interval_backs_off_after_threshold():
    with patch.object(br, "_read_skill_cadence", return_value=_cad(adaptive_backoff_after=3)):
        # 3 no-ops → 2x, 6 → 3x, 9 → 4x (capped), 100 → 4x.
        assert br.effective_skill_interval(10, noop_streak=3) == 20
        assert br.effective_skill_interval(10, noop_streak=6) == 30
        assert br.effective_skill_interval(10, noop_streak=9) == 40
        assert br.effective_skill_interval(10, noop_streak=100) == 40  # capped at 4x


def test_effective_interval_respects_disabled_backoff():
    with patch.object(br, "_read_skill_cadence", return_value=_cad(adaptive_backoff=False)):
        assert br.effective_skill_interval(10, noop_streak=99) == 10


def test_effective_interval_zero_base_disables():
    with patch.object(br, "_read_skill_cadence", return_value=_cad()):
        assert br.effective_skill_interval(0, noop_streak=99) == 0


# ---------------------------------------------------------------------------
# ①  aux-model routing
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, provider="openai-codex", model="gpt-5.5"):
        self.provider = provider
        self.model = model

    def _current_main_runtime(self):
        return {
            "api_key": "parent-key",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_app_server",
        }


def test_routing_auto_inherits_parent_and_downgrades_codex_app_server():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {"provider": "auto", "model": ""}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"
    assert rt["model"] == "gpt-5.5"
    # codex_app_server downgraded so agent-loop tools dispatch.
    assert rt["api_mode"] == "codex_responses"


def test_routing_to_aux_model_marks_routed_and_resolves_credentials():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter",
        "model": "google/gemini-3-flash-preview",
    }}}
    fake_rp = {
        "provider": "openrouter",
        "api_key": "or-key",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }
    with patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=fake_rp):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is True
    assert rt["provider"] == "openrouter"
    assert rt["model"] == "google/gemini-3-flash-preview"
    assert rt["api_key"] == "or-key"
    assert rt["base_url"] == "https://openrouter.ai/api/v1"


def test_routing_same_as_parent_is_not_treated_as_routed():
    agent = _FakeAgent(provider="openrouter", model="anthropic/claude-opus-4.8")
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter",
        "model": "anthropic/claude-opus-4.8",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    # Same provider+model as parent → keep the cache-share path.
    assert rt["routed"] is False


def test_routing_resolution_failure_falls_back_to_parent():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter",
        "model": "google/gemini-3-flash-preview",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               side_effect=RuntimeError("boom")):
        rt = br._resolve_review_runtime(agent)
    # Resolution failure must not break the fork — fall back to the main model.
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"
