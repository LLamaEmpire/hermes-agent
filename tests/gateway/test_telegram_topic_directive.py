"""Telegram natural-language topic directive tests.

Covers Critic requirements:
3. NL directive parser: "set topic to scout", "topic scout", "clear topic",
   "topic status" → expected (command, topic_id) tuples.
4. Unknown topic fails closed: returns typed error message, pointer untouched.
5. No registry checker: fails closed with "not configured" message, pointer
   untouched.
6. Gateway preempt contract: _try_handle_telegram_topic_directive returns
   non-None for a recognized directive, which causes the call site to
   short-circuit before the agent runner (documented at run.py:8620-8621).
7. Error isolation: a recognized directive that raises inside the handler
   still returns a non-None error reply — never None, which would leak the
   directive text to the agent runner.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.active_topic as active_topic_module
from gateway.active_topic import (
    PlatformPrincipal,
    TopicNotRegisteredError,
    handle_telegram_topic_directive,
    parse_telegram_topic_directive,
    set_registered_check,
)
from gateway.config import Platform
from gateway.session import SessionSource
from hermes_state import SessionDB


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_module_state():
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)
    yield
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)


def _source(**overrides):
    defaults = dict(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        user_id="208214988",
        chat_type="dm",
        thread_id="1234",
    )
    defaults.update(overrides)
    return SessionSource(**defaults)


def _ok_checker():
    async def _check(app_id, topic_id):
        return True
    return _check


# ── parse_telegram_topic_directive ────────────────────────────────────


class TestParseDirective:
    def test_set_topic_to_form(self):
        assert parse_telegram_topic_directive("set topic to scout") == ("set", "scout")

    def test_switch_topic_to_form(self):
        assert parse_telegram_topic_directive("switch topic to research") == ("set", "research")

    def test_change_topic_to_form(self):
        assert parse_telegram_topic_directive("change topic to planner") == ("set", "planner")

    def test_bare_topic_name(self):
        assert parse_telegram_topic_directive("topic scout") == ("set", "scout")

    def test_bare_topic_name_case_insensitive(self):
        result = parse_telegram_topic_directive("Topic SCOUT")
        assert result is not None
        assert result[0] == "set"
        assert result[1] == "scout"

    def test_topic_status_does_not_parse_as_set(self):
        # "topic status" must route to the status command, not set("status")
        result = parse_telegram_topic_directive("topic status")
        assert result == ("status", None)

    def test_clear_topic(self):
        assert parse_telegram_topic_directive("clear topic") == ("clear", None)

    def test_reset_topic(self):
        assert parse_telegram_topic_directive("reset topic") == ("clear", None)

    def test_topic_status(self):
        assert parse_telegram_topic_directive("topic status") == ("status", None)

    def test_what_is_topic(self):
        result = parse_telegram_topic_directive("what is the active topic")
        assert result is not None
        assert result[0] == "status"

    def test_whats_topic(self):
        result = parse_telegram_topic_directive("what's the topic")
        assert result is not None
        assert result[0] == "status"

    def test_set_topic_lowercases_topic_id(self):
        result = parse_telegram_topic_directive("set topic to SCOUT")
        assert result is not None
        assert result[1] == "scout"

    def test_non_directive_returns_none(self):
        assert parse_telegram_topic_directive("hello, how are you?") is None

    def test_empty_string_returns_none(self):
        assert parse_telegram_topic_directive("") is None

    def test_whitespace_only_returns_none(self):
        assert parse_telegram_topic_directive("   ") is None

    def test_partial_set_phrase_returns_none(self):
        assert parse_telegram_topic_directive("set topic") is None

    def test_leading_whitespace_is_tolerated(self):
        result = parse_telegram_topic_directive("  set topic to scout  ")
        assert result == ("set", "scout")


# ── handle_telegram_topic_directive (full integration) ───────────────


class TestHandleDirective:
    def _run(self, coro):
        return asyncio.run(coro)

    # ── set → success (registered) ──────────────────────────────────

    def test_set_topic_success(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        set_registered_check(_ok_checker())
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="set topic to scout",
                updated_by="test"
            )
        )
        assert reply is not None
        assert "scout" in reply
        # Pointer must be written.
        row = db.read_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234", app_id="hermes-agent",
        )
        assert row is not None
        assert row["topic_id"] == "scout"
        db.close()

    # ── set → already set (same topic) ──────────────────────────────

    def test_set_topic_already_set_same_topic(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        set_registered_check(_ok_checker())
        self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="set topic to scout",
                updated_by="test"
            )
        )
        reply2 = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="set topic to scout",
                updated_by="test"
            )
        )
        assert reply2 is not None
        assert "already" in reply2.lower()
        db.close()

    # ── set → switch from prior topic ───────────────────────────────

    def test_set_topic_switch_from_prior(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        set_registered_check(_ok_checker())
        self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="set topic to research",
                updated_by="test"
            )
        )
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="set topic to scout",
                updated_by="test"
            )
        )
        assert reply is not None
        assert "research" in reply or "switched" in reply.lower() or "from" in reply.lower()
        db.close()

    # ── set → unknown topic fails closed, pointer untouched ─────────

    def test_set_unknown_topic_fails_closed(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")

        async def reject(app_id, topic_id):
            return False

        set_registered_check(reject)
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="set topic to ghost",
                updated_by="test"
            )
        )
        # Must return a typed error string (not None).
        assert reply is not None
        assert isinstance(reply, str)
        assert "ghost" in reply or "Unknown" in reply or "unknown" in reply

        # Pointer must NOT have been written.
        row = db.read_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234", app_id="hermes-agent",
        )
        assert row is None, "pointer must not be written for an unknown topic"
        db.close()

    # ── set → no checker fails closed with "not configured" ─────────

    def test_set_no_checker_fails_closed(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        # No checker wired (fixture reset to None).
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="set topic to scout",
                updated_by="test"
            )
        )
        # Must return a clear error message about the missing registry.
        assert reply is not None
        assert isinstance(reply, str)
        assert "not configured" in reply.lower() or "registry" in reply.lower()

        # Pointer must NOT have been written.
        row = db.read_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234", app_id="hermes-agent",
        )
        assert row is None, "pointer must not be written when no checker is wired"
        db.close()

    # ── clear → prior exists ─────────────────────────────────────────

    def test_clear_topic_with_prior(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        set_registered_check(_ok_checker())
        self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="set topic to scout",
                updated_by="test"
            )
        )
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="clear topic",
                updated_by="test"
            )
        )
        assert reply is not None
        assert "scout" in reply or "cleared" in reply.lower()

        row = db.read_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234", app_id="hermes-agent",
        )
        assert row is None, "pointer must be cleared"
        db.close()

    # ── clear → no prior ─────────────────────────────────────────────

    def test_clear_topic_no_prior(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="clear topic",
                updated_by="test"
            )
        )
        assert reply is not None
        assert "No active topic" in reply or "no active" in reply.lower()
        db.close()

    # ── status → topic set ───────────────────────────────────────────

    def test_status_with_topic_set(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        set_registered_check(_ok_checker())
        self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="set topic to scout",
                updated_by="test"
            )
        )
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="topic status",
                updated_by="test"
            )
        )
        assert reply is not None
        assert "scout" in reply
        db.close()

    # ── status → no topic ────────────────────────────────────────────

    def test_status_no_topic(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent", text="topic status",
                updated_by="test"
            )
        )
        assert reply is not None
        assert "No active topic" in reply or "no active" in reply.lower()
        db.close()

    # ── non-directive returns None ────────────────────────────────────

    def test_non_directive_returns_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="hermes-agent",
                text="can you write a poem about pandas?",
                updated_by="test",
            )
        )
        assert reply is None
        db.close()


# ── Gateway preempt contract (items 6 & 7) ───────────────────────────
#
# _try_handle_telegram_topic_directive is a GatewayRunner method. We call
# it as an unbound function with a minimal stub to avoid instantiating the
# full GatewayRunner.
#
# The preempt contract (run.py lines 8620-8621):
#   if _nl_topic_reply is not None:
#       return _nl_topic_reply
#
# This means: any non-None return from _try_handle_telegram_topic_directive
# prevents the message from reaching the agent runner. Tests here prove:
#   - Recognized directives always return non-None (items 6 & 7).
#   - Non-directives return None (so they fall through to the runner).


class TestPreemptContract:
    def _run(self, coro):
        return asyncio.run(coro)

    def _make_self(self, *, db, app_id="hermes-agent"):
        return SimpleNamespace(
            config=SimpleNamespace(topic_default_app_id=app_id),
            _session_db=db,
        )

    def _make_event(self, text: str):
        return SimpleNamespace(text=text)

    def _call(self, self_stub, event, source):
        from gateway.run import GatewayRunner
        return self._run(
            GatewayRunner._try_handle_telegram_topic_directive(
                self_stub, event, source
            )
        )

    # ── item 6: recognized directive short-circuits (returns non-None) ──

    def test_set_directive_returns_non_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        set_registered_check(_ok_checker())
        stub = self._make_self(db=db)
        reply = self._call(stub, self._make_event("set topic to scout"), _source())
        assert reply is not None, "recognized set directive must return non-None"
        db.close()

    def test_clear_directive_returns_non_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        stub = self._make_self(db=db)
        reply = self._call(stub, self._make_event("clear topic"), _source())
        assert reply is not None, "recognized clear directive must return non-None"
        db.close()

    def test_status_directive_returns_non_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        stub = self._make_self(db=db)
        reply = self._call(stub, self._make_event("topic status"), _source())
        assert reply is not None, "recognized status directive must return non-None"
        db.close()

    def test_bare_topic_name_directive_returns_non_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        set_registered_check(_ok_checker())
        stub = self._make_self(db=db)
        reply = self._call(stub, self._make_event("topic scout"), _source())
        assert reply is not None, "bare topic-name directive must return non-None"
        db.close()

    # ── item 7: recognized directive that errors still returns non-None ──

    def test_recognized_directive_error_never_returns_none(self, tmp_path):
        """A recognized directive that raises inside the handler must return an
        error reply, NOT None. Returning None would leak the directive text to
        the agent runner, violating the preempt contract.
        """
        db = SessionDB(db_path=tmp_path / "state.db")
        stub = self._make_self(db=db)

        # Wire a checker that raises an unexpected error (not TopicNotRegisteredError).
        async def exploding_checker(app_id, topic_id):
            raise RuntimeError("simulated internal registry failure")

        set_registered_check(exploding_checker)
        reply = self._call(stub, self._make_event("set topic to scout"), _source())
        assert reply is not None, (
            "recognized directive that errors internally must return a non-None "
            "error reply — returning None would leak to the agent runner"
        )
        db.close()

    # ── non-directive falls through (returns None) ───────────────────────

    def test_non_directive_returns_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        stub = self._make_self(db=db)
        reply = self._call(
            stub,
            self._make_event("can you write a haiku about sushi?"),
            _source(),
        )
        assert reply is None, "non-directive must return None so it reaches the agent"
        db.close()

    def test_no_app_id_returns_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        stub = self._make_self(db=db, app_id="")
        reply = self._call(stub, self._make_event("set topic to scout"), _source())
        assert reply is None
        db.close()

    def test_no_session_db_returns_none(self):
        stub = SimpleNamespace(
            config=SimpleNamespace(topic_default_app_id="hermes-agent"),
            _session_db=None,
        )
        reply = self._call(stub, self._make_event("set topic to scout"), _source())
        assert reply is None


# ── Live surface regression: Telegram group/thread sources ───────────────
#
# The live miss: owner sent "topic status" in a Telegram group/thread and it
# reached the agent runner instead of being intercepted.  These tests model
# the exact live source shapes so the preempt is verified to hold for:
#   1. Normal group message (chat_type="group", thread_id set, user_id set).
#   2. Group observe-attribution message (user_id=None after the rewrite).
#   3. Non-directive text from a group source (must still fall through).
#
# The platform is the Platform.TELEGRAM enum value throughout — that is what
# the adapter stamps on every message.


def _group_source(**overrides):
    """Build a Telegram group/forum-thread SessionSource."""
    defaults = dict(
        platform=Platform.TELEGRAM,
        chat_id="-100123456789",
        user_id="208214988",
        chat_type="group",
        thread_id="567",
    )
    defaults.update(overrides)
    return SessionSource(**defaults)


class TestLiveGroupThreadSurface:
    """Regression: preempt must hold for group/thread sources, not just DMs."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_self(self, *, db, app_id="hermes-agent"):
        return SimpleNamespace(
            config=SimpleNamespace(topic_default_app_id=app_id),
            _session_db=db,
        )

    def _make_event(self, text: str):
        return SimpleNamespace(text=text)

    def _call(self, self_stub, event, source):
        from gateway.run import GatewayRunner
        return self._run(
            GatewayRunner._try_handle_telegram_topic_directive(
                self_stub, event, source
            )
        )

    # ── item 1: normal group message, "topic status" ──────────────────────

    def test_topic_status_group_source_returns_non_none(self, tmp_path):
        """'topic status' from a group/thread source must be preempted.

        This is the exact live miss scenario: owner sent 'topic status' in a
        Telegram group/thread and it reached the agent runner.
        """
        db = SessionDB(db_path=tmp_path / "state.db")
        stub = self._make_self(db=db)
        reply = self._call(
            stub,
            self._make_event("topic status"),
            _group_source(),
        )
        assert reply is not None, (
            "recognized 'topic status' from group/thread must be preempted "
            "(live miss regression)"
        )
        db.close()

    def test_set_directive_group_source_returns_non_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")

        async def _ok(app_id, topic_id):
            return True

        set_registered_check(_ok)
        stub = self._make_self(db=db)
        reply = self._call(
            stub,
            self._make_event("set topic to scout"),
            _group_source(),
        )
        assert reply is not None, "set directive from group/thread must be preempted"
        db.close()

    def test_clear_directive_group_source_returns_non_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        stub = self._make_self(db=db)
        reply = self._call(
            stub,
            self._make_event("clear topic"),
            _group_source(),
        )
        assert reply is not None, "clear directive from group/thread must be preempted"
        db.close()

    # ── item 2: user_id=None (observe-attribution rewrite) ───────────────

    def test_topic_status_null_user_id_returns_non_none(self, tmp_path):
        """Preempt must hold even when source.user_id is None.

        Telegram group observe-attribution rewrites the trigger source with
        user_id=None before dispatch.  The previous bug: PlatformPrincipal
        .from_source raised ValueError, handle_telegram_topic_directive
        returned None, and the directive reached the agent runner.
        """
        db = SessionDB(db_path=tmp_path / "state.db")
        stub = self._make_self(db=db)
        reply = self._call(
            stub,
            self._make_event("topic status"),
            _group_source(user_id=None),
        )
        assert reply is not None, (
            "recognized directive with user_id=None must still be preempted — "
            "returning None leaks to the agent runner"
        )
        db.close()

    def test_set_directive_null_user_id_returns_non_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        stub = self._make_self(db=db)
        reply = self._call(
            stub,
            self._make_event("set topic to scout"),
            _group_source(user_id=None),
        )
        assert reply is not None, (
            "set directive with user_id=None must be preempted with an error reply"
        )
        db.close()

    # ── item 3: non-directive from group source falls through ─────────────

    def test_non_directive_group_source_returns_none(self, tmp_path):
        """Non-directives from group sources must still fall through to the runner."""
        db = SessionDB(db_path=tmp_path / "state.db")
        stub = self._make_self(db=db)
        reply = self._call(
            stub,
            self._make_event("can you summarize the last meeting?"),
            _group_source(),
        )
        assert reply is None, "non-directive from group source must return None"
        db.close()

    def test_non_directive_null_user_id_returns_none(self, tmp_path):
        """Non-directives with user_id=None must also fall through."""
        db = SessionDB(db_path=tmp_path / "state.db")
        stub = self._make_self(db=db)
        reply = self._call(
            stub,
            self._make_event("what time is it?"),
            _group_source(user_id=None),
        )
        assert reply is None, "non-directive with user_id=None must return None"
        db.close()
