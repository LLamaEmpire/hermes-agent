"""Tests for per-thread app binding and extended topic directives.

Covers:
- hermes_state app_binding CRUD (persistence, schema, backcompat)
- resolve_effective_app_id: per-thread override vs gateway default fallback
- Session routing: per-thread app binding flows to topic session key
- New NL directive parsing: set_app, app_status, set_qualified, list_topics,
  create_topic, suggest_topics
- Handler integration for each new command
- gateway/topic_registry.py list_topics_in_registry + add_topic_to_registry
- Fail-closed / unknown-app / unknown-topic cases
- Non-directive not swallowed
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.active_topic as active_topic_module
from gateway.active_topic import (
    PlatformPrincipal,
    TopicNotRegisteredError,
    handle_telegram_topic_directive,
    parse_telegram_topic_directive,
    read_app_binding_for_source,
    resolve_effective_app_id,
    resolve_effective_app_id_sync,
    set_app_binding_for_source,
    set_registered_check,
)
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.topic_registry import (
    TOPIC_SLUG_RE,
    add_topic_to_registry,
    list_topics_in_registry,
)
from hermes_state import SessionDB


# ── Fixtures ──────────────────────────────────────────────────────────────────


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


def _group_source(**overrides):
    defaults = dict(
        platform=Platform.TELEGRAM,
        chat_id="-100123456789",
        user_id="208214988",
        chat_type="group",
        thread_id="567",
    )
    defaults.update(overrides)
    return SessionSource(**defaults)


def _ok_checker():
    async def _check(app_id, topic_id):
        return True
    return _check


def _make_registry(tmp_path: Path, app_id: str, topics: list) -> Path:
    """Write a minimal registry JSON file and return registry_root."""
    registry_root = tmp_path / "registry"
    registry_root.mkdir(exist_ok=True)
    (registry_root / f"{app_id}.json").write_text(
        json.dumps({"app_id": app_id, "version": 1, "topics": {t: {} for t in topics}}),
        encoding="utf-8",
    )
    return registry_root


# ── App binding DB CRUD ───────────────────────────────────────────────────────


class TestAppBindingDB:
    def test_read_before_migration_returns_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        result = db.read_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1"
        )
        assert result is None
        db.close()

    def test_set_and_read_roundtrip(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.set_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1",
            app_id="prowork", updated_by="test",
        )
        result = db.read_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1"
        )
        assert result == "prowork"
        db.close()

    def test_set_returns_prior_none_on_first(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        result = db.set_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1",
            app_id="prowork", updated_by="test",
        )
        assert result["prior"] is None
        assert result["app_id"] == "prowork"
        db.close()

    def test_set_returns_prior_on_update(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.set_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1",
            app_id="prowork", updated_by="test",
        )
        result = db.set_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1",
            app_id="development-os", updated_by="test",
        )
        assert result["prior"] == "prowork"
        assert result["app_id"] == "development-os"
        db.close()

    def test_clear_returns_prior(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.set_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1",
            app_id="prowork", updated_by="test",
        )
        prior = db.clear_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1",
            updated_by="test",
        )
        assert prior == "prowork"
        assert db.read_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1"
        ) is None
        db.close()

    def test_clear_nonexistent_returns_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.apply_app_binding_migration()
        prior = db.clear_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1",
            updated_by="test",
        )
        assert prior is None
        db.close()

    def test_thread_isolation(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.set_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1",
            app_id="prowork", updated_by="test",
        )
        db.set_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t2",
            app_id="development-os", updated_by="test",
        )
        assert db.read_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t1"
        ) == "prowork"
        assert db.read_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="t2"
        ) == "development-os"
        db.close()

    def test_default_thread_id_is_empty_string(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.set_app_binding(
            platform="telegram", user_id="u1", chat_id="c1",
            app_id="prowork", updated_by="test",
        )
        # Reading with explicit thread_id="" should find the row
        assert db.read_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id=""
        ) == "prowork"
        # thread_id="other" is isolated
        assert db.read_app_binding(
            platform="telegram", user_id="u1", chat_id="c1", thread_id="other"
        ) is None
        db.close()

    def test_migration_idempotent(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.apply_app_binding_migration()
        db.apply_app_binding_migration()  # second call must not raise
        db.close()

    def test_requires_app_id(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.apply_app_binding_migration()
        with pytest.raises(ValueError, match="app_id"):
            db.set_app_binding(
                platform="telegram", user_id="u1", chat_id="c1", thread_id="t1",
                app_id="", updated_by="test",
            )
        db.close()

    def test_requires_updated_by(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.apply_app_binding_migration()
        with pytest.raises(ValueError, match="updated_by"):
            db.set_app_binding(
                platform="telegram", user_id="u1", chat_id="c1", thread_id="t1",
                app_id="prowork", updated_by="",
            )
        db.close()


# ── resolve_effective_app_id ──────────────────────────────────────────────────


class TestResolveEffectiveAppId:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_per_thread_binding_overrides_default(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        src = _source(thread_id="999")
        db.set_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="999", app_id="prowork", updated_by="test",
        )
        result = self._run(
            resolve_effective_app_id(db, src, default_app_id="development-os")
        )
        assert result == "prowork"
        db.close()

    def test_no_binding_falls_back_to_default(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        src = _source(thread_id="999")
        result = self._run(
            resolve_effective_app_id(db, src, default_app_id="development-os")
        )
        assert result == "development-os"
        db.close()

    def test_no_binding_no_default_returns_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        src = _source(thread_id="999")
        result = self._run(
            resolve_effective_app_id(db, src, default_app_id=None)
        )
        assert result is None
        db.close()

    def test_binding_on_different_thread_doesnt_affect_other(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.set_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="thread-a", app_id="prowork", updated_by="test",
        )
        src_b = _source(thread_id="thread-b")
        result = self._run(
            resolve_effective_app_id(db, src_b, default_app_id="development-os")
        )
        assert result == "development-os"
        db.close()

    def test_sync_variant_per_thread_binding(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        src = _source(thread_id="42")
        db.set_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="42", app_id="prowork", updated_by="test",
        )
        result = resolve_effective_app_id_sync(db, src, default_app_id="development-os")
        assert result == "prowork"
        db.close()

    def test_sync_variant_fallback(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        src = _source(thread_id="42")
        result = resolve_effective_app_id_sync(db, src, default_app_id="development-os")
        assert result == "development-os"
        db.close()


# ── Routing: per-thread binding flows through to session key ──────────────────


class TestAppBindingRouting:
    """Verify that per-thread app binding is used in session key resolution."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_session_key_uses_per_thread_app(self, tmp_path):
        from gateway.active_topic import (
            resolve_topic_session_key_async,
            set_registered_check as _set_check,
        )
        from gateway.topic_registry import make_multi_app_registry_checker

        registry_root = _make_registry(tmp_path, "prowork", ["pbi-review"])
        _make_registry(tmp_path, "development-os", ["dashboards"])

        checker = make_multi_app_registry_checker(registry_root)
        _set_check(checker)

        db = SessionDB(db_path=tmp_path / "state.db")
        src = _source(thread_id="thread-pw")
        # Bind thread to "prowork"
        db.set_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="thread-pw", app_id="prowork", updated_by="test",
        )
        # Set active topic under "prowork"
        db.set_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="thread-pw", app_id="prowork", topic_id="pbi-review",
            updated_by="test",
        )

        # resolve_effective_app_id should give "prowork"
        app_id = self._run(
            resolve_effective_app_id(db, src, default_app_id="development-os")
        )
        assert app_id == "prowork"

        # The topic-routed session key should use "prowork" and "pbi-review"
        key = self._run(
            resolve_topic_session_key_async(
                src, db, app_id=app_id,
                pointer_mode_enabled=True,
                require_registered_check=True,
            )
        )
        assert key is not None
        assert "prowork" in key
        assert "pbi-review" in key
        db.close()

    def test_default_app_used_when_no_binding(self, tmp_path):
        from gateway.active_topic import (
            resolve_topic_session_key_async,
            set_registered_check as _set_check,
        )
        from gateway.topic_registry import make_multi_app_registry_checker

        registry_root = _make_registry(tmp_path, "development-os", ["dashboards"])
        checker = make_multi_app_registry_checker(registry_root)
        _set_check(checker)

        db = SessionDB(db_path=tmp_path / "state.db")
        src = _source(thread_id="thread-x")
        # No app binding set; default is "development-os"
        db.set_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="thread-x", app_id="development-os", topic_id="dashboards",
            updated_by="test",
        )

        app_id = self._run(
            resolve_effective_app_id(db, src, default_app_id="development-os")
        )
        assert app_id == "development-os"

        key = self._run(
            resolve_topic_session_key_async(
                src, db, app_id=app_id,
                pointer_mode_enabled=True,
                require_registered_check=True,
            )
        )
        assert key is not None
        assert "development-os" in key
        assert "dashboards" in key
        db.close()


# ── Directive parsing: new commands ───────────────────────────────────────────


class TestNewDirectiveParsing:
    def test_set_app_directive(self):
        assert parse_telegram_topic_directive("set app to development-os") == ("set_app", "development-os")

    def test_switch_app_directive(self):
        assert parse_telegram_topic_directive("switch app to prowork") == ("set_app", "prowork")

    def test_bind_app_directive(self):
        assert parse_telegram_topic_directive("bind app to prowork") == ("set_app", "prowork")

    def test_app_status_directive(self):
        assert parse_telegram_topic_directive("app status") == ("app_status", None)

    def test_set_qualified_topic_directive(self):
        assert parse_telegram_topic_directive("set topic to development-os/dashboards") == (
            "set_qualified", "development-os/dashboards"
        )

    def test_switch_qualified_topic_directive(self):
        assert parse_telegram_topic_directive("switch topic to prowork/pbi-review") == (
            "set_qualified", "prowork/pbi-review"
        )

    def test_bare_qualified_topic_directive(self):
        assert parse_telegram_topic_directive("topic prowork/pbi-review") == (
            "set_qualified", "prowork/pbi-review"
        )

    def test_list_topics_directive(self):
        assert parse_telegram_topic_directive("list topics") == ("list_topics", None)

    def test_show_topics_directive(self):
        assert parse_telegram_topic_directive("show topics") == ("list_topics", None)

    def test_list_topics_in_app_directive(self):
        assert parse_telegram_topic_directive("list topics in prowork") == ("list_topics", "prowork")

    def test_create_topic_directive(self):
        assert parse_telegram_topic_directive("create topic my-pbi") == ("create_topic", "my-pbi")

    def test_add_topic_directive(self):
        assert parse_telegram_topic_directive("add topic my-pbi") == ("create_topic", "my-pbi")

    def test_create_qualified_topic_directive(self):
        assert parse_telegram_topic_directive("create topic prowork/my-pbi") == (
            "create_topic", "prowork/my-pbi"
        )

    def test_suggest_topics_directive(self):
        assert parse_telegram_topic_directive("suggest topics") == ("suggest_topics", None)

    def test_propose_topic_directive(self):
        assert parse_telegram_topic_directive("propose topic") == ("suggest_topics", None)

    def test_existing_set_topic_unchanged(self):
        assert parse_telegram_topic_directive("set topic to scout") == ("set", "scout")

    def test_existing_clear_unchanged(self):
        assert parse_telegram_topic_directive("clear topic") == ("clear", None)

    def test_existing_status_unchanged(self):
        assert parse_telegram_topic_directive("topic status") == ("status", None)

    def test_non_directive_returns_none(self):
        assert parse_telegram_topic_directive("hello there") is None

    def test_case_insensitive(self):
        assert parse_telegram_topic_directive("SET APP TO Development-OS") == (
            "set_app", "development-os"
        )


# ── Handler: set_app command ─────────────────────────────────────────────────


class TestHandleSetApp:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_set_app_success(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set app to prowork", updated_by="test",
            )
        )
        assert reply is not None
        assert "prowork" in reply
        # Binding must be persisted
        assert db.read_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234",
        ) == "prowork"
        db.close()

    def test_set_app_already_bound_same_app(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set app to prowork", updated_by="test",
            )
        )
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set app to prowork", updated_by="test",
            )
        )
        assert reply is not None
        assert "already" in reply.lower()
        db.close()

    def test_set_app_switch_from_prior(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set app to prowork", updated_by="test",
            )
        )
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set app to development-os", updated_by="test",
            )
        )
        assert reply is not None
        assert "prowork" in reply or "switched" in reply.lower()
        db.close()

    def test_set_app_validates_registry_if_root_provided(self, tmp_path):
        registry_root = _make_registry(tmp_path, "prowork", ["pbi-x"])
        db = SessionDB(db_path=tmp_path / "state.db")
        # Unknown app with registry_root provided → fails closed
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set app to ghost-app", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "ghost-app" in reply or "not found" in reply.lower()
        # No binding written
        assert db.read_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234",
        ) is None
        db.close()

    def test_set_app_without_registry_root_skips_validation(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        # No registry_root → skip validation, allow any app_id string
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set app to ghost-app", updated_by="test",
                registry_root=None,
            )
        )
        assert reply is not None
        assert "ghost-app" in reply
        db.close()

    def test_null_user_id_returns_non_none_preempt(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _group_source(user_id=None), db, app_id="development-os",
                text="set app to prowork", updated_by="test",
            )
        )
        assert reply is not None, "set_app with user_id=None must not return None"
        db.close()


# ── Handler: app_status command ───────────────────────────────────────────────


class TestHandleAppStatus:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_app_status_no_binding_shows_default(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="app status", updated_by="test",
            )
        )
        assert reply is not None
        assert "development-os" in reply
        assert "gateway default" in reply.lower() or "default" in reply.lower()
        db.close()

    def test_app_status_with_binding_shows_binding(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.set_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234", app_id="prowork", updated_by="test",
        )
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="app status", updated_by="test",
            )
        )
        assert reply is not None
        assert "prowork" in reply
        assert "per-thread" in reply.lower() or "binding" in reply.lower()
        db.close()

    def test_app_status_includes_active_topic(self, tmp_path):
        set_registered_check(_ok_checker())
        db = SessionDB(db_path=tmp_path / "state.db")
        # Set topic under development-os
        db.set_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234", app_id="development-os", topic_id="dashboards",
            updated_by="test",
        )
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="app status", updated_by="test",
            )
        )
        assert reply is not None
        assert "dashboards" in reply
        db.close()


# ── Handler: set_qualified command ────────────────────────────────────────────


class TestHandleSetQualified:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_set_qualified_success_with_registry_root(self, tmp_path):
        registry_root = _make_registry(tmp_path, "prowork", ["pbi-review"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set topic to prowork/pbi-review", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "prowork" in reply
        assert "pbi-review" in reply
        # App binding must be set
        assert db.read_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234",
        ) == "prowork"
        # Topic pointer must be set
        row = db.read_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234", app_id="prowork",
        )
        assert row is not None
        assert row["topic_id"] == "pbi-review"
        db.close()

    def test_set_qualified_unknown_app_fails_closed(self, tmp_path):
        registry_root = _make_registry(tmp_path, "prowork", ["pbi-review"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set topic to ghost-app/pbi-review", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "ghost-app" in reply or "not found" in reply.lower()
        # No binding written
        assert db.read_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234",
        ) is None
        db.close()

    def test_set_qualified_unknown_topic_fails_closed(self, tmp_path):
        registry_root = _make_registry(tmp_path, "prowork", ["pbi-review"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set topic to prowork/ghost-topic", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "ghost-topic" in reply or "not registered" in reply.lower()
        # No pointer written under prowork
        assert db.read_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234", app_id="prowork",
        ) is None
        db.close()

    def test_set_qualified_atomically_sets_both(self, tmp_path):
        """App binding and topic pointer must both be set in one operation."""
        registry_root = _make_registry(tmp_path, "prowork", ["pbi-review"])
        db = SessionDB(db_path=tmp_path / "state.db")
        self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set topic to prowork/pbi-review", updated_by="test",
                registry_root=registry_root,
            )
        )
        app = db.read_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234",
        )
        topic = db.read_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234", app_id="prowork",
        )
        assert app == "prowork"
        assert topic is not None and topic["topic_id"] == "pbi-review"
        db.close()

    def test_set_qualified_without_registry_falls_back_to_checker(self, tmp_path):
        """Without registry_root, uses assert_registered with wired checker."""
        set_registered_check(_ok_checker())
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set topic to prowork/pbi-review", updated_by="test",
                registry_root=None,
            )
        )
        assert reply is not None
        assert "prowork" in reply
        db.close()


# ── Handler: list_topics command ─────────────────────────────────────────────


class TestHandleListTopics:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_list_topics_current_app(self, tmp_path):
        registry_root = _make_registry(tmp_path, "development-os", ["dashboards", "scout"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="list topics", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "dashboards" in reply
        assert "scout" in reply
        db.close()

    def test_list_topics_explicit_app(self, tmp_path):
        registry_root = _make_registry(tmp_path, "prowork", ["pbi-x", "pbi-y"])
        _make_registry(tmp_path, "development-os", ["dashboards"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="list topics in prowork", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "pbi-x" in reply
        assert "pbi-y" in reply
        db.close()

    def test_list_topics_no_registry_root_returns_error(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="list topics", updated_by="test",
                registry_root=None,
            )
        )
        assert reply is not None
        assert "not configured" in reply.lower() or "cannot list" in reply.lower()
        db.close()

    def test_list_topics_unknown_app_fails_closed(self, tmp_path):
        registry_root = _make_registry(tmp_path, "prowork", ["pbi-x"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="list topics in ghost-app", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "ghost-app" in reply or "cannot list" in reply.lower()
        db.close()

    def test_list_topics_no_app_configured_returns_error(self, tmp_path):
        """When no app is configured and no binding, show helpful message."""
        registry_root = _make_registry(tmp_path, "development-os", ["dashboards"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="",
                text="list topics", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "no app" in reply.lower() or "not configured" in reply.lower()
        db.close()


# ── Handler: create_topic command ─────────────────────────────────────────────


class TestHandleCreateTopic:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_create_topic_success(self, tmp_path):
        registry_root = _make_registry(tmp_path, "development-os", ["dashboards"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="create topic new-feature", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "new-feature" in reply
        assert "created" in reply.lower()
        # Verify it's in the file
        topics = list_topics_in_registry(registry_root, "development-os")
        assert "new-feature" in topics
        db.close()

    def test_create_topic_idempotent(self, tmp_path):
        registry_root = _make_registry(tmp_path, "development-os", ["existing"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="create topic existing", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "already" in reply.lower() or "exists" in reply.lower()
        db.close()

    def test_create_topic_invalid_slug_fails_closed(self, tmp_path):
        registry_root = _make_registry(tmp_path, "development-os", [])
        db = SessionDB(db_path=tmp_path / "state.db")
        # "pbi@review" contains @ which is invalid even after lowercasing
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="create topic pbi@review", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "invalid" in reply.lower() or "cannot create" in reply.lower()
        # Must not be written
        topics = list_topics_in_registry(registry_root, "development-os")
        assert "pbi@review" not in topics
        db.close()

    def test_create_topic_with_uppercase_fails(self, tmp_path):
        registry_root = _make_registry(tmp_path, "development-os", [])
        db = SessionDB(db_path=tmp_path / "state.db")
        # Directive lowercases the input, so "NEW-FEATURE" → "new-feature" → valid
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="create topic NEW-FEATURE", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        # new-feature should be created (after lowercasing)
        topics = list_topics_in_registry(registry_root, "development-os")
        assert "new-feature" in topics
        db.close()

    def test_create_qualified_topic(self, tmp_path):
        registry_root = _make_registry(tmp_path, "prowork", ["pbi-x"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="create topic prowork/pbi-new", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "pbi-new" in reply
        topics = list_topics_in_registry(registry_root, "prowork")
        assert "pbi-new" in topics
        db.close()

    def test_create_topic_no_registry_root_returns_error(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="create topic new-feature", updated_by="test",
                registry_root=None,
            )
        )
        assert reply is not None
        assert "not configured" in reply.lower() or "cannot create" in reply.lower()
        db.close()


# ── Handler: suggest_topics command ──────────────────────────────────────────


class TestHandleSuggestTopics:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_suggest_topics_returns_non_none(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="suggest topics", updated_by="test",
            )
        )
        assert reply is not None
        db.close()

    def test_suggest_topics_mentions_no_mutation(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="suggest topics", updated_by="test",
            )
        )
        assert reply is not None
        # Must communicate that no moves happen automatically
        assert any(
            word in reply.lower()
            for word in ("proposal", "follow-up", "not move", "no messages", "not mutating")
        )
        db.close()

    def test_suggest_topics_does_not_write_db(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="suggest topics", updated_by="test",
            )
        )
        # No app binding written
        assert db.read_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234",
        ) is None
        # No topic pointer written
        # (migration not triggered, table may not exist — read returns None gracefully)
        row = db.read_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234", app_id="development-os",
        )
        assert row is None
        db.close()


# ── topic_registry.py: list and add ──────────────────────────────────────────


class TestRegistryFunctions:
    def test_list_topics_returns_sorted(self, tmp_path):
        registry_root = _make_registry(tmp_path, "development-os", ["z-topic", "a-topic", "m-topic"])
        topics = list_topics_in_registry(registry_root, "development-os")
        assert topics == sorted(topics)
        assert set(topics) == {"z-topic", "a-topic", "m-topic"}

    def test_list_topics_missing_file_raises(self, tmp_path):
        registry_root = tmp_path / "registry"
        registry_root.mkdir()
        with pytest.raises(ValueError, match="no registry file"):
            list_topics_in_registry(registry_root, "nonexistent")

    def test_add_topic_creates_new(self, tmp_path):
        registry_root = _make_registry(tmp_path, "development-os", ["existing"])
        result = add_topic_to_registry(registry_root, "development-os", "new-topic")
        assert result == {"created": True, "app_id": "development-os", "topic_id": "new-topic"}
        topics = list_topics_in_registry(registry_root, "development-os")
        assert "new-topic" in topics
        assert "existing" in topics

    def test_add_topic_idempotent(self, tmp_path):
        registry_root = _make_registry(tmp_path, "development-os", ["existing"])
        result = add_topic_to_registry(registry_root, "development-os", "existing")
        assert result == {"created": False, "app_id": "development-os", "topic_id": "existing"}
        # File unchanged (no extra entries)
        topics = list_topics_in_registry(registry_root, "development-os")
        assert topics == ["existing"]

    def test_add_topic_invalid_slug_raises(self, tmp_path):
        registry_root = _make_registry(tmp_path, "development-os", [])
        with pytest.raises(ValueError, match="invalid topic slug"):
            add_topic_to_registry(registry_root, "development-os", "Bad Slug!")

    def test_add_topic_uppercase_slug_raises(self, tmp_path):
        registry_root = _make_registry(tmp_path, "development-os", [])
        with pytest.raises(ValueError, match="invalid topic slug"):
            add_topic_to_registry(registry_root, "development-os", "BadSlug")

    def test_add_topic_missing_registry_raises(self, tmp_path):
        registry_root = tmp_path / "registry"
        registry_root.mkdir()
        with pytest.raises(ValueError, match="no registry file"):
            add_topic_to_registry(registry_root, "nonexistent", "new-topic")

    def test_add_topic_atomic_write(self, tmp_path):
        """Verify no .tmp file left after successful write."""
        registry_root = _make_registry(tmp_path, "development-os", [])
        add_topic_to_registry(registry_root, "development-os", "new-topic")
        tmp_files = list(registry_root.glob(".*.tmp"))
        assert tmp_files == [], f"leftover tmp files: {tmp_files}"

    def test_slug_re_accepts_valid(self):
        for slug in ("scout", "pbi-review", "a", "a0", "pbi-copilot-jr", "x_y"):
            assert TOPIC_SLUG_RE.match(slug), f"{slug!r} should be valid"

    def test_slug_re_rejects_invalid(self):
        for slug in ("", "-scout", "Scout", "pbi review", "a/b", "pbi.x"):
            assert not TOPIC_SLUG_RE.match(slug), f"{slug!r} should be invalid"


# ── Fail-closed and non-directive regression ──────────────────────────────────


class TestFailClosed:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_non_directive_not_swallowed(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="what's the weather like?", updated_by="test",
            )
        )
        assert reply is None, "non-directive must return None so it reaches the agent"
        db.close()

    def test_unknown_app_fails_closed_set_qualified(self, tmp_path):
        registry_root = _make_registry(tmp_path, "prowork", ["pbi-x"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set topic to ghost/pbi-x", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "ghost" in reply or "not found" in reply.lower()
        assert db.read_app_binding(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234",
        ) is None
        db.close()

    def test_unknown_topic_fails_closed_set_qualified(self, tmp_path):
        registry_root = _make_registry(tmp_path, "prowork", ["pbi-x"])
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set topic to prowork/ghost-topic", updated_by="test",
                registry_root=registry_root,
            )
        )
        assert reply is not None
        assert "ghost-topic" in reply or "not registered" in reply.lower()
        assert db.read_active_topic(
            platform="telegram", user_id="208214988", chat_id="208214988",
            thread_id="1234", app_id="prowork",
        ) is None
        db.close()

    def test_existing_set_topic_still_works(self, tmp_path):
        """Existing 'set topic to X' (plain, current app) still works."""
        set_registered_check(_ok_checker())
        db = SessionDB(db_path=tmp_path / "state.db")
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set topic to scout", updated_by="test",
            )
        )
        assert reply is not None
        assert "scout" in reply
        db.close()

    def test_existing_status_still_works(self, tmp_path):
        set_registered_check(_ok_checker())
        db = SessionDB(db_path=tmp_path / "state.db")
        self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="set topic to scout", updated_by="test",
            )
        )
        reply = self._run(
            handle_telegram_topic_directive(
                _source(), db, app_id="development-os",
                text="topic status", updated_by="test",
            )
        )
        assert reply is not None
        assert "scout" in reply
        db.close()
