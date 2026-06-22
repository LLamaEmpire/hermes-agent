"""HRM-T0a step 7 — ``/topic move-last`` + ``/topic move-range`` slash UX.

Covers:

- Successful **dry-run** (no mutation, plan returned).
- Successful **commit** (turns moved + banner emitted).
- Invalid ``count`` (zero, negative, non-int).
- Invalid range (inverted, non-int, missing ``..``, missing positional).
- ``--to`` slug: missing, malformed, unknown (assert_registered refusal),
  same as current topic.
- ``--idempotency-key`` required on commit; ``--dry-run`` bypasses that.
- Replay: second commit with the same idempotency_key reports ``(replay)``
  and does not double-mutate.
- Already-moved / tombstoned rows are user-visibly skipped on a second
  distinct commit (the primitive filters them — slash UX surfaces the
  reduced count).
- Active-topic pointer is preserved across a successful move.
- **Default-deny / legacy**: when ``topic_slash_ux_enabled`` is off,
  ``/topic move-last`` falls through to the legacy Telegram-DM forum-thread
  handler (which refuses on a non-Telegram source).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional

import pytest

import gateway.active_topic as active_topic_module
from gateway.active_topic import set_registered_check
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_state import SessionDB


# ── Fixtures / helpers ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_module_state():
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)
    yield
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)


def _source(**overrides) -> SessionSource:
    defaults = dict(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        user_id="208214988",
        chat_type="dm",
        thread_id=None,
    )
    defaults.update(overrides)
    return SessionSource(**defaults)


def _ok_checker():
    async def _check(app_id, topic_id):
        return True
    return _check


def _registry_with(allowed: set):
    async def _check(app_id, topic_id):
        return topic_id in allowed
    return _check


@dataclass
class FakeAdapter:
    sent: list = None
    raise_on_send: Exception | None = None

    def __post_init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        if self.raise_on_send is not None:
            raise self.raise_on_send
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata})


@dataclass
class FakeSessionEntry:
    session_id: str


class FakeSessionStore:
    """Just-enough store: ``_entries[session_key] -> entry`` with
    ``get_or_create_session(source, _session_key=)`` create-on-demand.

    The slash move handler reads ``_entries.get(key)`` for src and uses
    ``get_or_create_session`` for dst when not yet materialised.
    """

    def __init__(self, *, db: SessionDB):
        self._entries: Dict[str, FakeSessionEntry] = {}
        self._db = db
        self._counter = 0

    def seed(self, session_key: str, session_id: str, *, user_id: str = "u"):
        self._entries[session_key] = FakeSessionEntry(session_id=session_id)
        # Ensure SessionDB has a matching session row.
        try:
            self._db.create_session(
                session_id=session_id, source="telegram", user_id=user_id
            )
        except Exception:
            pass

    def get_or_create_session(self, source, *, _session_key=None, **kwargs):
        if _session_key is None:
            raise RuntimeError("test store requires explicit _session_key")
        existing = self._entries.get(_session_key)
        if existing is not None:
            return existing
        self._counter += 1
        sid = f"on-demand-{self._counter}"
        self._db.create_session(
            session_id=sid,
            source=str(getattr(source.platform, "value", source.platform)),
            user_id=str(source.user_id),
        )
        entry = FakeSessionEntry(session_id=sid)
        self._entries[_session_key] = entry
        return entry


class FakeRunner(GatewaySlashCommandsMixin):
    def __init__(self, *, config, session_db, adapter, store=None):
        self.config = config
        self._session_db = session_db
        self.adapters = {Platform.TELEGRAM: adapter}
        self._is_user_authorized = lambda src: True
        self.session_store = store if store is not None else FakeSessionStore(
            db=session_db
        )


def _config(*, slash_ux_enabled=True, app_id="hermes-agent") -> GatewayConfig:
    return GatewayConfig(
        topic_pointer_mode_enabled=True,
        topic_default_app_id=app_id,
        topic_slash_ux_enabled=slash_ux_enabled,
    )


def _make_event(text: str, *, source=None):
    from gateway.platforms.base import MessageEvent, MessageType

    src = source or _source()
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=src)


def _build_keys(principal_topic: str, dst_topic: str) -> tuple[str, str]:
    """Return (src_session_key, dst_session_key) for the canonical
    Telegram principal used by these tests."""
    from gateway.active_topic import (
        PlatformPrincipal,
        build_topic_session_key,
    )
    principal = PlatformPrincipal.from_source(_source(), app_id="hermes-agent")
    return (
        build_topic_session_key(principal, topic_id=principal_topic),
        build_topic_session_key(principal, topic_id=dst_topic),
    )


def _seed_active_topic(db: SessionDB, topic_id: str):
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id=topic_id,
        updated_by="seed",
    )


# ── move-last: dry-run plans without mutation ─────────────────────────


def test_move_last_dry_run_plans_without_mutation(tmp_path):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "src")
    set_registered_check(_registry_with({"src", "dst"}))

    src_key, dst_key = _build_keys("src", "dst")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "src-sid")
    store.seed(dst_key, "dst-sid")
    for i in range(3):
        db.append_message("src-sid", "user", f"src-{i}")

    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter(), store=store)
    ev = _make_event("/topic move-last 2 --to dst --dry-run")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "dry-run" in resp and "2 turn" in resp
        assert "src" in resp and "dst" in resp
        # No mutation.
        src_active = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'src-sid' AND active = 1"
        ).fetchone()[0]
        assert src_active == 3
        dst_total = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'dst-sid'"
        ).fetchone()[0]
        assert dst_total == 0
    finally:
        db.close()


# ── move-last: commit happy path + banner ─────────────────────────────


def test_move_last_commit_moves_turns_and_emits_banner(tmp_path):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "src")
    set_registered_check(_registry_with({"src", "dst"}))

    src_key, dst_key = _build_keys("src", "dst")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "src-sid")
    store.seed(dst_key, "dst-sid")
    for i in range(4):
        db.append_message("src-sid", "user", f"src-{i}")
    db.append_message("dst-sid", "user", "dst-0")

    adapter = FakeAdapter()
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter, store=store)
    ev = _make_event("/topic move-last 2 --to dst --idempotency-key k1")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert resp == ""  # banner already sent
        assert adapter.sent and adapter.sent[0]["text"] == (
            "[topic move → dst] 2 turn(s) moved"
        )
        src_active = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'src-sid' AND active = 1"
        ).fetchone()[0]
        assert src_active == 2
        dst_total = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'dst-sid'"
        ).fetchone()[0]
        assert dst_total == 3  # 1 + 2 moved
        # Active-topic pointer preserved.
        row = db.read_active_topic(
            platform="telegram", user_id="208214988",
            chat_id="208214988", app_id="hermes-agent",
        )
        assert row and row["topic_id"] == "src"
    finally:
        db.close()


# ── move-last: replay returns (replay) banner ─────────────────────────


def test_move_last_replay_idempotent_does_not_double_move(tmp_path):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "src")
    set_registered_check(_registry_with({"src", "dst"}))

    src_key, dst_key = _build_keys("src", "dst")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "src-sid")
    store.seed(dst_key, "dst-sid")
    for i in range(3):
        db.append_message("src-sid", "user", f"src-{i}")

    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter(), store=store)
    ev1 = _make_event("/topic move-last 2 --to dst --idempotency-key replay-key")
    ev2 = _make_event("/topic move-last 2 --to dst --idempotency-key replay-key")
    try:
        r1 = asyncio.run(runner._handle_topic_command(ev1))
        r2 = asyncio.run(runner._handle_topic_command(ev2))
        assert r1 == ""
        # Second call's banner carries "(replay)".
        assert r2 == ""
        # Banners on adapter:
        last_two = runner.adapters[Platform.TELEGRAM].sent
        assert last_two[-1]["text"].endswith("(replay)")
        # Only 2 src rows tombstoned, not 4.
        dst_total = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'dst-sid'"
        ).fetchone()[0]
        assert dst_total == 2
    finally:
        db.close()


# ── move-last: already-moved rows skipped on a fresh commit ───────────


def test_move_last_skips_already_moved_rows_on_distinct_commit(tmp_path):
    """After commit-1 tombstones 2 rows, a second commit with a NEW key
    asking for last-5 only sees the 1 remaining active row."""
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "src")
    set_registered_check(_registry_with({"src", "dst"}))

    src_key, dst_key = _build_keys("src", "dst")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "src-sid")
    store.seed(dst_key, "dst-sid")
    for i in range(3):
        db.append_message("src-sid", "user", f"src-{i}")

    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter(), store=store)
    try:
        asyncio.run(
            runner._handle_topic_command(
                _make_event("/topic move-last 2 --to dst --idempotency-key k1")
            )
        )
        # Second pass — different key, asking for "last 5" but only 1 left.
        resp = asyncio.run(
            runner._handle_topic_command(
                _make_event("/topic move-last 5 --to dst --idempotency-key k2")
            )
        )
        assert resp == ""
        last_banner = runner.adapters[Platform.TELEGRAM].sent[-1]["text"]
        assert "1 turn(s) moved" in last_banner
        src_active = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'src-sid' AND active = 1"
        ).fetchone()[0]
        assert src_active == 0
    finally:
        db.close()


# ── move-last: validation errors ──────────────────────────────────────


@pytest.mark.parametrize("text,needle", [
    ("/topic move-last --to dst --dry-run", "missing N"),
    ("/topic move-last abc --to dst --dry-run", "invalid count"),
    ("/topic move-last 0 --to dst --dry-run", ">= 1"),
    ("/topic move-last -3 --to dst --dry-run", ">= 1"),
    ("/topic move-last 2 --dry-run", "--to <slug> is required"),
    ("/topic move-last 2 --to dst", "--idempotency-key required"),
    ("/topic move-last 2 --to BAD!SLUG --dry-run", "not a valid topic slug"),
])
def test_move_last_validation_errors(tmp_path, text, needle):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "src")
    set_registered_check(_registry_with({"src", "dst"}))

    src_key, dst_key = _build_keys("src", "dst")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "src-sid")
    store.seed(dst_key, "dst-sid")

    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter(), store=store)
    try:
        resp = asyncio.run(runner._handle_topic_command(_make_event(text)))
        assert needle in resp, resp
    finally:
        db.close()


def test_move_last_refuses_when_no_active_topic(tmp_path):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_registry_with({"dst"}))

    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())
    ev = _make_event("/topic move-last 2 --to dst --dry-run")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "no active topic" in resp
    finally:
        db.close()


def test_move_last_refuses_same_src_and_dst(tmp_path):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "lane")
    set_registered_check(_registry_with({"lane"}))

    src_key, _ = _build_keys("lane", "lane")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "lane-sid")

    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter(), store=store)
    ev = _make_event("/topic move-last 1 --to lane --dry-run")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "must differ" in resp
    finally:
        db.close()


def test_move_last_refuses_unregistered_dst(tmp_path):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "src")
    # registry only knows "src", not "ghost"
    set_registered_check(_registry_with({"src"}))

    src_key, _ = _build_keys("src", "ghost")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "src-sid")

    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter(), store=store)
    ev = _make_event("/topic move-last 1 --to ghost --dry-run")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "not registered" in resp and "ghost" in resp
    finally:
        db.close()


def test_move_last_refuses_when_src_session_absent(tmp_path):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "src")
    set_registered_check(_registry_with({"src", "dst"}))

    # store has NO src entry — user has switched but never sent a message.
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())
    ev = _make_event("/topic move-last 1 --to dst --dry-run")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "no session for active topic" in resp
    finally:
        db.close()


# ── move-range: commit + dry-run + validation ─────────────────────────


def test_move_range_commit_happy_path(tmp_path):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "src")
    set_registered_check(_registry_with({"src", "dst"}))

    src_key, dst_key = _build_keys("src", "dst")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "src-sid")
    store.seed(dst_key, "dst-sid")
    for i in range(4):
        db.append_message("src-sid", "user", f"src-{i}")
    src_ids = [
        r[0] for r in db._conn.execute(
            "SELECT id FROM messages WHERE session_id = 'src-sid' ORDER BY id ASC"
        ).fetchall()
    ]

    adapter = FakeAdapter()
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter, store=store)
    cmd = f"/topic move-range {src_ids[1]}..{src_ids[2]} --to dst --idempotency-key r1"
    ev = _make_event(cmd)
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert resp == ""
        assert adapter.sent[0]["text"] == "[topic move → dst] 2 turn(s) moved"
        src_active = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'src-sid' AND active = 1"
        ).fetchone()[0]
        assert src_active == 2
    finally:
        db.close()


def test_move_range_dry_run_does_not_mutate(tmp_path):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "src")
    set_registered_check(_registry_with({"src", "dst"}))

    src_key, dst_key = _build_keys("src", "dst")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "src-sid")
    store.seed(dst_key, "dst-sid")
    for i in range(2):
        db.append_message("src-sid", "user", f"src-{i}")
    src_ids = [
        r[0] for r in db._conn.execute(
            "SELECT id FROM messages WHERE session_id = 'src-sid' ORDER BY id ASC"
        ).fetchall()
    ]
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter(), store=store)
    ev = _make_event(
        f"/topic move-range {src_ids[0]}..{src_ids[1]} --to dst --dry-run"
    )
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "dry-run" in resp and "2 turn" in resp
        src_active = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'src-sid' AND active = 1"
        ).fetchone()[0]
        assert src_active == 2
    finally:
        db.close()


@pytest.mark.parametrize("text,needle", [
    ("/topic move-range --to dst --dry-run", "missing range"),
    ("/topic move-range no_dots --to dst --dry-run", "invalid range"),
    ("/topic move-range a..b --to dst --dry-run", "A and B must be integers"),
    ("/topic move-range 5..2 --to dst --dry-run", "from_id must be <= to_id"),
    ("/topic move-range 0..5 --to dst --dry-run", "must be positive"),
    ("/topic move-range 1..3 --dry-run", "--to <slug> is required"),
    ("/topic move-range 1..3 --to dst", "--idempotency-key required"),
])
def test_move_range_validation_errors(tmp_path, text, needle):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "src")
    set_registered_check(_registry_with({"src", "dst"}))

    src_key, dst_key = _build_keys("src", "dst")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "src-sid")
    store.seed(dst_key, "dst-sid")

    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter(), store=store)
    try:
        resp = asyncio.run(runner._handle_topic_command(_make_event(text)))
        assert needle in resp, resp
    finally:
        db.close()


# ── slash UX flag default-deny preserves legacy fall-through ──────────


def test_move_subcommands_default_deny_routes_to_legacy(tmp_path):
    """When ``topic_slash_ux_enabled`` is False, ``/topic move-last``
    must hit the legacy Telegram-DM forum-thread handler — not the new
    move executor — and therefore refuse on a non-Telegram source."""
    cfg = _config(slash_ux_enabled=False)
    db = SessionDB(db_path=tmp_path / "state.db")

    discord_src = SessionSource(
        platform=Platform.DISCORD, chat_id="c", user_id="u", chat_type="dm"
    )
    adapter = FakeAdapter()
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter)
    runner.adapters = {Platform.DISCORD: adapter}

    ev = _make_event(
        "/topic move-last 1 --to dst --idempotency-key k", source=discord_src
    )
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "Telegram" in resp or "telegram" in resp
    finally:
        db.close()


# ── dst materialised on demand when slug is registered ────────────────


def test_move_creates_dst_session_on_demand_when_registered(tmp_path):
    """The dst slug passed assert_registered but has no SessionEntry
    yet — the slash handler must materialise one via session_store
    rather than refusing (the user's recovery intent is unambiguous)."""
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_active_topic(db, "src")
    set_registered_check(_registry_with({"src", "dst"}))

    src_key, dst_key = _build_keys("src", "dst")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "src-sid")
    # NO dst entry seeded.
    db.append_message("src-sid", "user", "x")

    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter(), store=store)
    ev = _make_event("/topic move-last 1 --to dst --idempotency-key k")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert resp == ""
        assert dst_key in store._entries
    finally:
        db.close()


# ── /topic help text now mentions the move subcommands ────────────────


def test_topic_help_mentions_move_subcommands(tmp_path):
    cfg = _config()
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())
    ev = _make_event("/topic help")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "move-last" in resp and "move-range" in resp
    finally:
        db.close()
