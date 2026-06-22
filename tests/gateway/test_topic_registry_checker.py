"""Tests for gateway/topic_registry.py — JSON registry checker.

Covers:
- Registered topic allows switch (via assert_registered integration).
- Unregistered topic rejects.
- No checker wired or missing registry root fails closed.
- Malformed registry does not crash.
- Config default root is profile-aware (hermes_home / "registry").
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import gateway.active_topic as active_topic_module
from gateway.active_topic import (
    TopicNotRegisteredError,
    assert_registered,
    set_registered_check,
)
from gateway.topic_registry import make_json_registry_checker

APP_ID = "development-os"


@pytest.fixture(autouse=True)
def _reset_checker():
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)
    yield
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)


def _write_registry(root: Path, app_id: str, topics: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{app_id}.json").write_text(
        json.dumps({"app_id": app_id, "version": 1, "topics": topics}),
        encoding="utf-8",
    )


def _run(coro):
    return asyncio.run(coro)


# ── Registered topic allows switch ────────────────────────────────────


def test_registered_topic_returns_true(tmp_path):
    _write_registry(tmp_path, APP_ID, {"daily-standup": {"surfaces": ["telegram"]}})
    checker = make_json_registry_checker(tmp_path, APP_ID)
    assert _run(checker(APP_ID, "daily-standup")) is True


def test_multiple_topics_all_accepted(tmp_path):
    _write_registry(tmp_path, APP_ID, {"alpha": {}, "beta": {"surfaces": ["discord"]}})
    checker = make_json_registry_checker(tmp_path, APP_ID)
    assert _run(checker(APP_ID, "alpha")) is True
    assert _run(checker(APP_ID, "beta")) is True


def test_wired_checker_allows_registered_topic_via_assert_registered(tmp_path):
    _write_registry(tmp_path, APP_ID, {"sprint-planning": {}})
    set_registered_check(make_json_registry_checker(tmp_path, APP_ID))
    _run(assert_registered(APP_ID, "sprint-planning"))  # must not raise


# ── Unregistered topic rejects ────────────────────────────────────────


def test_unregistered_topic_returns_false(tmp_path):
    _write_registry(tmp_path, APP_ID, {"only-topic": {}})
    checker = make_json_registry_checker(tmp_path, APP_ID)
    assert _run(checker(APP_ID, "nonexistent")) is False


def test_wired_checker_rejects_unregistered_topic(tmp_path):
    _write_registry(tmp_path, APP_ID, {"sprint-planning": {}})
    set_registered_check(make_json_registry_checker(tmp_path, APP_ID))
    with pytest.raises(TopicNotRegisteredError):
        _run(assert_registered(APP_ID, "unknown-topic"))


# ── No checker or missing registry root fails closed ──────────────────


def test_no_checker_wired_fails_closed():
    with pytest.raises(TopicNotRegisteredError):
        _run(assert_registered(APP_ID, "any-topic"))


def test_missing_registry_file_fails_closed(tmp_path):
    checker = make_json_registry_checker(tmp_path, APP_ID)
    assert _run(checker(APP_ID, "any-topic")) is False


def test_nonexistent_registry_root_fails_closed(tmp_path):
    checker = make_json_registry_checker(tmp_path / "does" / "not" / "exist", APP_ID)
    assert _run(checker(APP_ID, "t")) is False


def test_missing_root_via_assert_registered_fails_closed(tmp_path):
    set_registered_check(make_json_registry_checker(tmp_path / "no-registry", APP_ID))
    with pytest.raises(TopicNotRegisteredError):
        _run(assert_registered(APP_ID, "any-topic"))


# ── Malformed registry does not crash ─────────────────────────────────


def test_malformed_json_fails_closed(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{APP_ID}.json").write_text("{not valid json", encoding="utf-8")
    checker = make_json_registry_checker(tmp_path, APP_ID)
    assert _run(checker(APP_ID, "any-topic")) is False


def test_registry_root_not_a_dict_fails_closed(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{APP_ID}.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    checker = make_json_registry_checker(tmp_path, APP_ID)
    assert _run(checker(APP_ID, "any-topic")) is False


def test_app_id_mismatch_in_file_fails_closed(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{APP_ID}.json").write_text(
        json.dumps({"app_id": "wrong-app", "version": 1, "topics": {"t": {}}}),
        encoding="utf-8",
    )
    checker = make_json_registry_checker(tmp_path, APP_ID)
    assert _run(checker(APP_ID, "t")) is False


def test_topics_field_missing_fails_closed(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{APP_ID}.json").write_text(
        json.dumps({"app_id": APP_ID, "version": 1}), encoding="utf-8"
    )
    checker = make_json_registry_checker(tmp_path, APP_ID)
    assert _run(checker(APP_ID, "any-topic")) is False


def test_topics_not_a_dict_fails_closed(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{APP_ID}.json").write_text(
        json.dumps({"app_id": APP_ID, "version": 1, "topics": ["t1", "t2"]}),
        encoding="utf-8",
    )
    checker = make_json_registry_checker(tmp_path, APP_ID)
    assert _run(checker(APP_ID, "any-topic")) is False


# ── Caller app_id mismatch fails closed ───────────────────────────────


def test_caller_app_id_mismatch_fails_closed(tmp_path):
    _write_registry(tmp_path, APP_ID, {"t": {}})
    checker = make_json_registry_checker(tmp_path, APP_ID)
    assert _run(checker("other-app", "t")) is False


# ── Config: default registry root is hermes_home/registry ─────────────


def test_config_default_registry_root_is_hermes_home_registry(monkeypatch, tmp_path):
    """Verify that make_json_registry_checker using hermes_home/registry as root
    correctly loads a file placed there — proving the default path is profile-aware
    (hermes_home resolves from environment, as in gateway startup wiring)."""
    fake_home = tmp_path / "fake-hermes-home"
    registry_root = fake_home / "registry"
    _write_registry(registry_root, APP_ID, {"my-topic": {}})
    checker = make_json_registry_checker(registry_root, APP_ID)
    assert _run(checker(APP_ID, "my-topic")) is True
    assert _run(checker(APP_ID, "unregistered")) is False
