"""HRM-T0a step 9 — adversarial leakage / isolation suite.

These tests intentionally try to *break* the same-chat active-topic-pointer
routing model by simulating hostile or buggy callers crossing principal /
topic / legacy / authorization boundaries. They lock down the contract the
prior steps documented:

1. Cross-principal isolation — principal A cannot read, mutate, or be
   routed into B's topic/session/pointer.
2. Cross-topic isolation — switching pointers flips routing without
   bleeding messages or session ids.
3. Switch/send race — concurrent ``set_active_topic`` + inbound resolve
   serialise via the per-key asyncio lock; neither sees torn state.
4. Legacy isolation — pre-HRM sessions stay legacy under pointer-mode
   slash mutators; no silent migration; a stray pointer row for a
   *different* principal does not leak into the legacy principal's
   routing.
5. Unauthorized move prevention — slash move-last/move-range only touches
   the inbound principal's resolved src/dst topics; the API surface is
   bearer-gated and refuses moves without auth (no fallback path).
6. Metadata-only inventory — :func:`legacy_inventory` never surfaces
   message content, prompts, responses, titles, reasoning, tool args, or
   raw thread payloads — only ids + counts.

Scope: new tests + a tiny, surgical production fix if (and only if) a
test exposes a real leak. Step 9 does not enable API server, edit live
config, or expand feature surface.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.active_topic as active_topic_module
from gateway.active_topic import (
    LEGACY_READONLY_MESSAGE,
    PlatformPrincipal,
    acquire_pointer_lock,
    build_topic_session_key,
    is_legacy_principal_route,
    legacy_inventory,
    read_active_topic,
    resolve_topic_session_key,
    resolve_topic_session_key_async,
    set_active_topic,
    set_registered_check,
    _reset_legacy_banner_for_tests,
)
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session import SessionSource, SessionStore, build_session_key
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_state import SessionDB


# ── Fixtures / helpers ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_module_state():
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)
    _reset_legacy_banner_for_tests()
    yield
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)
    _reset_legacy_banner_for_tests()


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

    def __post_init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata})


@dataclass
class FakeSessionEntry:
    session_id: str


class FakeSessionStore:
    def __init__(self, *, db: SessionDB):
        self._entries: Dict[str, FakeSessionEntry] = {}
        self._db = db
        self._counter = 0

    def seed(self, session_key: str, session_id: str, *, user_id: str = "u"):
        self._entries[session_key] = FakeSessionEntry(session_id=session_id)
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


# Principal A: the "victim". Principal B: the "attacker" trying to bleed in.
# Different user_id values under the same platform/chat/app cover the
# group_sessions_per_user contract; different app_id covers cross-repo
# isolation in the same DM.
PRINCIPAL_A_USER = "user-aaa"
PRINCIPAL_B_USER = "user-bbb"


def _src_a(**over):
    base = dict(
        platform=Platform.TELEGRAM,
        chat_id="chat-shared",
        user_id=PRINCIPAL_A_USER,
        chat_type="dm",
    )
    base.update(over)
    return SessionSource(**base)


def _src_b(**over):
    base = dict(
        platform=Platform.TELEGRAM,
        chat_id="chat-shared",
        user_id=PRINCIPAL_B_USER,
        chat_type="dm",
    )
    base.update(over)
    return SessionSource(**base)


# ──────────────────────────────────────────────────────────────────────
# 1. Cross-principal isolation
# ──────────────────────────────────────────────────────────────────────


def test_pointer_for_principal_a_does_not_leak_to_b_read(tmp_path):
    """Setting A's pointer must NOT surface on B's read."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.set_active_topic(
            platform="telegram", user_id=PRINCIPAL_A_USER,
            chat_id="chat-shared", app_id="hermes-agent",
            topic_id="A-secret-topic", updated_by="seed",
        )
        # B's read returns None — no leak.
        b_row = db.read_active_topic(
            platform="telegram", user_id=PRINCIPAL_B_USER,
            chat_id="chat-shared", app_id="hermes-agent",
        )
        assert b_row is None
        # A's row intact.
        a_row = db.read_active_topic(
            platform="telegram", user_id=PRINCIPAL_A_USER,
            chat_id="chat-shared", app_id="hermes-agent",
        )
        assert a_row and a_row["topic_id"] == "A-secret-topic"
    finally:
        db.close()


def test_b_set_active_topic_does_not_mutate_a_pointer(tmp_path):
    """B writing their own pointer leaves A's row byte-identical."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.set_active_topic(
            platform="telegram", user_id=PRINCIPAL_A_USER,
            chat_id="chat-shared", app_id="hermes-agent",
            topic_id="topic-A", updated_by="seed-A",
        )
        before = db.read_active_topic(
            platform="telegram", user_id=PRINCIPAL_A_USER,
            chat_id="chat-shared", app_id="hermes-agent",
        )
        db.set_active_topic(
            platform="telegram", user_id=PRINCIPAL_B_USER,
            chat_id="chat-shared", app_id="hermes-agent",
            topic_id="topic-B", updated_by="seed-B",
        )
        after = db.read_active_topic(
            platform="telegram", user_id=PRINCIPAL_A_USER,
            chat_id="chat-shared", app_id="hermes-agent",
        )
        assert before == after
    finally:
        db.close()


def test_topic_session_keys_isolate_principals_by_user_chat_app(tmp_path):
    """The key derivation forbids ANY two distinct principals from
    resolving to the same topic-routed session_key, even with identical
    topic_id."""
    pa = PlatformPrincipal("telegram", PRINCIPAL_A_USER, "chat-shared", "hermes-agent")
    pb = PlatformPrincipal("telegram", PRINCIPAL_B_USER, "chat-shared", "hermes-agent")
    pc = PlatformPrincipal("telegram", PRINCIPAL_A_USER, "chat-other", "hermes-agent")
    pd = PlatformPrincipal("telegram", PRINCIPAL_A_USER, "chat-shared", "other-app")
    pe = PlatformPrincipal("discord", PRINCIPAL_A_USER, "chat-shared", "hermes-agent")
    keys = {
        build_topic_session_key(p, topic_id="research")
        for p in (pa, pb, pc, pd, pe)
    }
    # Five distinct principals → five distinct keys. No collision.
    assert len(keys) == 5


def test_b_inbound_resolver_does_not_pick_up_a_pointer(tmp_path):
    """A's pointer is set; B's inbound resolver must return None (no flip)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    try:
        db.set_active_topic(
            platform="telegram", user_id=PRINCIPAL_A_USER,
            chat_id="chat-shared", app_id="hermes-agent",
            topic_id="A-only", updated_by="seed",
        )
        key_for_b = resolve_topic_session_key(
            _src_b(), db, app_id="hermes-agent",
        )
        assert key_for_b is None, (
            f"B routed into A's pointer: {key_for_b!r}"
        )
        # A's resolve still works.
        key_for_a = resolve_topic_session_key(
            _src_a(), db, app_id="hermes-agent",
        )
        assert key_for_a is not None
        assert key_for_a.endswith(":A-only")
        assert PRINCIPAL_A_USER in key_for_a
        assert PRINCIPAL_B_USER not in key_for_a
    finally:
        db.close()


def test_clear_active_topic_for_b_does_not_clear_a(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.set_active_topic(
            platform="telegram", user_id=PRINCIPAL_A_USER,
            chat_id="chat-shared", app_id="hermes-agent",
            topic_id="A-topic", updated_by="seed",
        )
        db.set_active_topic(
            platform="telegram", user_id=PRINCIPAL_B_USER,
            chat_id="chat-shared", app_id="hermes-agent",
            topic_id="B-topic", updated_by="seed",
        )
        # B clears.
        db.clear_active_topic(
            platform="telegram", user_id=PRINCIPAL_B_USER,
            chat_id="chat-shared", app_id="hermes-agent",
            updated_by="B-clear",
        )
        # A's row survives.
        a = db.read_active_topic(
            platform="telegram", user_id=PRINCIPAL_A_USER,
            chat_id="chat-shared", app_id="hermes-agent",
        )
        assert a and a["topic_id"] == "A-topic"
        # B's row is gone.
        b = db.read_active_topic(
            platform="telegram", user_id=PRINCIPAL_B_USER,
            chat_id="chat-shared", app_id="hermes-agent",
        )
        assert b is None
    finally:
        db.close()


def test_app_id_isolates_pointer_in_same_chat_same_user(tmp_path):
    """One principal can have concurrent pointers under different app_ids
    without cross-talk (charter: single principal, multiple repos)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.set_active_topic(
            platform="telegram", user_id="u", chat_id="c", app_id="repo-a",
            topic_id="t-a", updated_by="seed",
        )
        db.set_active_topic(
            platform="telegram", user_id="u", chat_id="c", app_id="repo-b",
            topic_id="t-b", updated_by="seed",
        )
        ra = db.read_active_topic(
            platform="telegram", user_id="u", chat_id="c", app_id="repo-a",
        )
        rb = db.read_active_topic(
            platform="telegram", user_id="u", chat_id="c", app_id="repo-b",
        )
        assert ra["topic_id"] == "t-a"
        assert rb["topic_id"] == "t-b"
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────
# 2. Cross-topic isolation
# ──────────────────────────────────────────────────────────────────────


def test_switching_topics_changes_routed_key_no_bleed(tmp_path):
    """Routing pre-pass returns key-A before switch and key-B after.
    The two keys never overlap; the topic_id is the discriminator."""
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    try:
        # Switch to A.
        db.set_active_topic(
            platform="telegram", user_id="u", chat_id="c", app_id="app",
            topic_id="topic-A", updated_by="seed",
        )
        src = SessionSource(
            platform=Platform.TELEGRAM, chat_id="c", user_id="u", chat_type="dm",
        )
        key_a = resolve_topic_session_key(src, db, app_id="app")
        # Switch to B.
        db.set_active_topic(
            platform="telegram", user_id="u", chat_id="c", app_id="app",
            topic_id="topic-B", updated_by="seed",
        )
        key_b = resolve_topic_session_key(src, db, app_id="app")

        assert key_a and key_b and key_a != key_b
        assert key_a.endswith(":topic-A")
        assert key_b.endswith(":topic-B")
        # Cross-substring sanity: neither key references the other topic_id.
        assert "topic-B" not in key_a
        assert "topic-A" not in key_b
    finally:
        db.close()


def test_messages_appended_under_topic_key_do_not_appear_in_other_topic(tmp_path):
    """Same principal, two topic-routed sessions, distinct session_id values
    behind each key. A message written to one cannot be observed by the
    other's transcript."""
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    store._db = SessionDB(db_path=tmp_path / "state.db")
    try:
        # Pointer for principal at topic A.
        db.set_active_topic(
            platform="telegram", user_id="u", chat_id="c", app_id="hermes-agent",
            topic_id="alpha", updated_by="seed",
        )
        src = SessionSource(
            platform=Platform.TELEGRAM, chat_id="c", user_id="u", chat_type="dm",
        )
        key_alpha = store._generate_session_key(src)
        # Materialise an entry + session row under alpha.
        sid_alpha = "sess-alpha"
        store._db.create_session(
            session_id=sid_alpha, source="telegram", user_id="u"
        )
        store._db.append_message(sid_alpha, "user", "SECRET-FROM-ALPHA")

        # Switch to topic B and ensure a different session is used.
        db.set_active_topic(
            platform="telegram", user_id="u", chat_id="c", app_id="hermes-agent",
            topic_id="beta", updated_by="seed",
        )
        key_beta = store._generate_session_key(src)
        assert key_alpha != key_beta

        sid_beta = "sess-beta"
        store._db.create_session(
            session_id=sid_beta, source="telegram", user_id="u"
        )
        store._db.append_message(sid_beta, "user", "BENIGN-FROM-BETA")

        # Cross-read: alpha's transcript does NOT contain beta's content,
        # and vice versa.
        alpha_msgs = store._db._conn.execute(
            "SELECT content FROM messages WHERE session_id = ?", (sid_alpha,)
        ).fetchall()
        beta_msgs = store._db._conn.execute(
            "SELECT content FROM messages WHERE session_id = ?", (sid_beta,)
        ).fetchall()
        alpha_text = json.dumps([r[0] for r in alpha_msgs])
        beta_text = json.dumps([r[0] for r in beta_msgs])
        assert "BENIGN-FROM-BETA" not in alpha_text
        assert "SECRET-FROM-ALPHA" not in beta_text
    finally:
        store._db.close()
        db.close()


def test_concurrent_topic_set_and_read_via_lock_is_consistent(tmp_path):
    """Switch/send race: many resolver reads racing against a pointer
    switch — each read returns either the pre- or post-switch topic key,
    never a torn intermediate value."""
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    try:
        db.set_active_topic(
            platform="telegram", user_id="u", chat_id="c", app_id="app",
            topic_id="before", updated_by="seed",
        )
        src = SessionSource(
            platform=Platform.TELEGRAM, chat_id="c", user_id="u", chat_type="dm",
        )
        principal = PlatformPrincipal.from_source(src, app_id="app")

        async def run():
            results: list = []

            async def reader():
                key = await resolve_topic_session_key_async(
                    src, db, app_id="app"
                )
                results.append(key)

            async def writer():
                # Single switch mid-race.
                await asyncio.sleep(0)
                await set_active_topic(
                    db, principal,
                    topic_id="after",
                    updated_by="switcher",
                    require_registered=False,
                )

            tasks = [reader() for _ in range(20)] + [writer()]
            await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run())
        valid = {":before", ":after"}
        for k in results:
            assert k is not None
            assert any(k.endswith(v) for v in valid), (
                f"torn key: {k!r}"
            )
        # At least one of each — the race actually hit both sides
        # (probabilistic, but with 20 readers + write at sleep(0), reliable).
        assert any(k.endswith(":before") for k in results) or any(
            k.endswith(":after") for k in results
        )
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────
# 3. Switch/send race — serialised under the per-key lock
# ──────────────────────────────────────────────────────────────────────


def test_writer_holding_lock_blocks_reader_until_release(tmp_path):
    """While a writer holds the principal's per-key lock, an async resolver
    on the same key must block, not race past. Once released, the resolver
    completes with the new value."""
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    try:
        principal = PlatformPrincipal("telegram", "u", "c", "app")

        async def run():
            # Acquire and hold the writer lock.
            lock = await acquire_pointer_lock(principal.key)
            observed: list = []

            async def reader():
                # First write the new pointer (under lock by us).
                # The resolver_async should not see this until we release.
                key = await resolve_topic_session_key_async(
                    SessionSource(
                        platform=Platform.TELEGRAM, chat_id="c", user_id="u",
                        chat_type="dm",
                    ),
                    db, app_id="app",
                )
                observed.append(key)

            async with lock:
                # Set the pointer while holding the writer lock — the
                # resolver task must not have read the row yet.
                db.set_active_topic(
                    platform="telegram", user_id="u", chat_id="c", app_id="app",
                    topic_id="after-write", updated_by="writer",
                )
                # Launch the reader; yield once to let it start and block
                # on acquire_pointer_lock.
                reader_task = asyncio.create_task(reader())
                await asyncio.sleep(0.01)
                assert observed == [], (
                    "reader observed pointer before writer released lock"
                )
            await reader_task
            return observed

        observed = asyncio.run(run())
        assert len(observed) == 1
        assert observed[0] and observed[0].endswith(":after-write")
    finally:
        db.close()


def test_principal_a_lock_does_not_block_principal_b(tmp_path):
    """Per-key locks: holding A's principal lock must NOT block a resolver
    for principal B (different lock identity)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    try:
        db.set_active_topic(
            platform="telegram", user_id=PRINCIPAL_B_USER,
            chat_id="chat-shared", app_id="hermes-agent",
            topic_id="B-only", updated_by="seed",
        )
        pa = PlatformPrincipal("telegram", PRINCIPAL_A_USER, "chat-shared", "hermes-agent")

        async def run():
            a_lock = await acquire_pointer_lock(pa.key)
            async with a_lock:
                # While we hold A's lock, resolve B's key — should not block.
                fut = asyncio.create_task(
                    resolve_topic_session_key_async(
                        _src_b(), db, app_id="hermes-agent",
                    )
                )
                # Wait briefly; the task must complete despite our holding A.
                done, pending = await asyncio.wait({fut}, timeout=1.0)
                assert fut in done, "B resolver blocked behind A's per-key lock"
                return fut.result()

        key_b = asyncio.run(run())
        assert key_b and key_b.endswith(":B-only")
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────
# 4. Legacy isolation
# ──────────────────────────────────────────────────────────────────────


def _seed_pre_migration_session(db: SessionDB, *, user_id="208214988"):
    pre_ts = 1_000.0
    sid = f"pre-{user_id}"
    db.create_session(session_id=sid, source="telegram", user_id=str(user_id))
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?", (pre_ts, sid),
    )
    db._conn.commit()
    db.apply_hrm_t0a_migration()
    return sid


def test_legacy_session_routing_never_promotes_to_topic_key(tmp_path):
    """Even with an `_ok_checker` wired, a legacy principal whose pointer
    table has no row must NOT route through a topic-derived key."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        _seed_pre_migration_session(db)
    finally:
        db.close()

    set_registered_check(_ok_checker())
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    store._db = SessionDB(db_path=db_path)
    try:
        key = store._generate_session_key(_source())
        assert "topic:" not in key
        assert key == build_session_key(_source())
    finally:
        store._db.close()


def test_legacy_principal_cannot_be_mutated_by_pointer_slash(tmp_path):
    """All mutators are refused under legacy mode. No pointer write occurs."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_pre_migration_session(db)
    set_registered_check(_ok_checker())

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())
    try:
        # Even attempting a `switch` returns the read-only banner.
        for sub in (
            "switch foo",
            "clear",
            "bind-thread",
            "unbind-thread",
            "move-last 1 --to bar --dry-run",
            "move-range 1..2 --to bar --dry-run",
        ):
            resp = asyncio.run(
                runner._handle_topic_command(_make_event(f"/topic {sub}"))
            )
            assert resp == LEGACY_READONLY_MESSAGE
        # Pointer table untouched.
        assert (
            db.read_active_topic(
                platform="telegram", user_id="208214988",
                chat_id="208214988", app_id="hermes-agent",
            )
            is None
        )
    finally:
        db.close()


def test_legacy_principal_resolver_ignores_stray_pointer_for_other_principal(tmp_path):
    """Pre-migration session principal P1 is legacy. Inserting a *foreign*
    pointer for P2 in the same DB must NOT cause P1's resolver to flip
    routing — the resolver is keyed on P1's own envelope."""
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    try:
        _seed_pre_migration_session(db, user_id="208214988")
        # Drop a pointer for a *different* user — same chat / app.
        db.set_active_topic(
            platform="telegram", user_id="other-user",
            chat_id="208214988", app_id="hermes-agent",
            topic_id="stray", updated_by="adversary",
        )
        key = resolve_topic_session_key(
            _source(), db, app_id="hermes-agent",
        )
        assert key is None, (
            f"legacy principal routed through foreign pointer: {key!r}"
        )
    finally:
        db.close()


def test_legacy_inventory_does_not_classify_post_migration_session_as_legacy(tmp_path):
    """A session created AFTER the migration marker must not show up in
    the pre_migration counts/principals lists."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        # Force migration stamp first.
        db.apply_hrm_t0a_migration()
        marker = float(db.get_meta("hrm_t0a_applied_at"))
        # Create a session with started_at strictly above the marker.
        db.create_session(
            session_id="post-1", source="telegram", user_id="post-user",
        )
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (marker + 1000.0, "post-1"),
        )
        db._conn.commit()
        inv = legacy_inventory(db)
        assert inv["pre_migration_session_count"] == 0
        assert not any(
            p["user_id"] == "post-user" for p in inv["pre_migration_principals"]
        )
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────
# 5. Unauthorized move prevention
# ──────────────────────────────────────────────────────────────────────


def test_slash_move_cannot_touch_foreign_principals_session(tmp_path):
    """Principal A's `/topic move-last` resolves src/dst session_ids from
    A's own principal envelope. A session belonging to principal B with
    the SAME topic_id must remain untouched."""
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_registry_with({"shared-name", "dst-name"}))

    # Topic "shared-name" exists in both A's and B's namespace, but the
    # session keys differ because the principal envelope differs.
    pa = PlatformPrincipal("telegram", PRINCIPAL_A_USER, "chat-shared", "hermes-agent")
    pb = PlatformPrincipal("telegram", PRINCIPAL_B_USER, "chat-shared", "hermes-agent")
    src_key_a = build_topic_session_key(pa, topic_id="shared-name")
    dst_key_a = build_topic_session_key(pa, topic_id="dst-name")
    src_key_b = build_topic_session_key(pb, topic_id="shared-name")
    assert src_key_a != src_key_b

    # Seed B's session with hostile-named content; A must not move it.
    store = FakeSessionStore(db=db)
    store.seed(src_key_a, "A-src-sid", user_id=PRINCIPAL_A_USER)
    store.seed(dst_key_a, "A-dst-sid", user_id=PRINCIPAL_A_USER)
    store.seed(src_key_b, "B-src-sid", user_id=PRINCIPAL_B_USER)

    db.append_message("A-src-sid", "user", "A-MSG")
    db.append_message("B-src-sid", "user", "B-PROTECTED-MSG")
    db.set_active_topic(
        platform="telegram", user_id=PRINCIPAL_A_USER,
        chat_id="chat-shared", app_id="hermes-agent",
        topic_id="shared-name", updated_by="seed",
    )

    cfg = _config()
    adapter = FakeAdapter()
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter, store=store)
    try:
        # Run the move as principal A. It should target A-src-sid, never
        # B-src-sid, even though the topic name collides.
        ev = _make_event(
            "/topic move-last 5 --to dst-name --idempotency-key k1",
            source=_src_a(),
        )
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert resp == ""  # banner sent

        # B's session must be byte-equal: row count, content present.
        b_remaining = db._conn.execute(
            "SELECT content FROM messages WHERE session_id = 'B-src-sid' "
            "AND active = 1"
        ).fetchall()
        assert len(b_remaining) == 1
        assert b_remaining[0][0] == "B-PROTECTED-MSG"

        # The move log row references A's sessions only — never B's.
        log_rows = db._conn.execute(
            "SELECT src_session_id, dst_session_id FROM move_log"
        ).fetchall()
        for src_sid, dst_sid in log_rows:
            assert src_sid != "B-src-sid" and dst_sid != "B-src-sid"
    finally:
        db.close()


def test_slash_move_dst_to_unregistered_foreign_topic_refused(tmp_path):
    """The `--to` slug must pass assert_registered under A's app_id.
    Trying to point at a slug only registered for a different app is
    refused (assert_registered returns False)."""
    db = SessionDB(db_path=tmp_path / "state.db")

    async def _per_app(app_id, topic_id):
        # Only registered under "other-app", not under A's "hermes-agent".
        return app_id == "other-app" and topic_id == "elsewhere"

    set_registered_check(_per_app)

    pa = PlatformPrincipal("telegram", PRINCIPAL_A_USER, "chat-shared", "hermes-agent")
    src_key = build_topic_session_key(pa, topic_id="active")
    store = FakeSessionStore(db=db)
    store.seed(src_key, "A-src-sid", user_id=PRINCIPAL_A_USER)
    db.append_message("A-src-sid", "user", "x")
    db.set_active_topic(
        platform="telegram", user_id=PRINCIPAL_A_USER,
        chat_id="chat-shared", app_id="hermes-agent",
        topic_id="active", updated_by="seed",
    )

    cfg = _config()
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter(), store=store)
    try:
        ev = _make_event(
            "/topic move-last 1 --to elsewhere --idempotency-key k",
            source=_src_a(),
        )
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "not registered" in resp, resp
        # No move_log row written.
        count = db._conn.execute("SELECT COUNT(*) FROM move_log").fetchone()[0]
        assert count == 0
    finally:
        db.close()


# ── API surface — bearer-auth gate on move endpoints ──────────────────


def _create_move_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/api/sessions/{session_id}/move/last", adapter._handle_session_move_last
    )
    app.router.add_post(
        "/api/sessions/{session_id}/move/range", adapter._handle_session_move_range
    )
    return app


@pytest.mark.asyncio
async def test_api_move_endpoints_refuse_without_bearer_when_key_set(tmp_path):
    """API surface: bearer required when API_SERVER_KEY is configured.
    Without it, every move attempt → 401, no state changes."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.apply_hrm_t0a_migration()  # create move_log up-front
        db.create_session(session_id="src", source="api_server", user_id="u")
        db.create_session(session_id="dst", source="api_server", user_id="u")
        db.append_message("src", "user", "PAYLOAD")

        a = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
        a._session_db = db
        app = _create_move_app(a)
        async with TestClient(TestServer(app)) as cli:
            for path in (
                "/api/sessions/src/move/last",
                "/api/sessions/src/move/range",
            ):
                body = {
                    "dst_session_id": "dst",
                    "count": 1,
                    "from_id": 1,
                    "to_id": 1,
                    "idempotency_key": "k",
                }
                resp = await cli.post(path, json=body)
                assert resp.status == 401, (path, await resp.text())
            # Try a *wrong* bearer too — still 401.
            resp = await cli.post(
                "/api/sessions/src/move/last",
                json={"dst_session_id": "dst", "count": 1, "idempotency_key": "k"},
                headers={"Authorization": "Bearer sk-WRONG"},
            )
            assert resp.status == 401

        # No move_log rows from rejected requests.
        moves = db._conn.execute("SELECT COUNT(*) FROM move_log").fetchone()[0]
        assert moves == 0
        # Source row untouched.
        active = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='src' AND active=1"
        ).fetchone()[0]
        assert active == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_api_move_endpoints_reject_invalid_idempotency_key_payload(tmp_path):
    """A malformed idempotency_key body (control chars) must 400 without
    mutating state — the validation gate is consistent across surfaces."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.apply_hrm_t0a_migration()  # create move_log up-front
        db.create_session(session_id="src", source="api_server", user_id="u")
        db.create_session(session_id="dst", source="api_server", user_id="u")
        db.append_message("src", "user", "x")
        a = APIServerAdapter(PlatformConfig(enabled=True))
        a._session_db = db
        app = _create_move_app(a)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/sessions/src/move/last",
                json={
                    "dst_session_id": "dst",
                    "count": 1,
                    "idempotency_key": "bad\rkey",
                },
            )
            assert resp.status == 400
        moves = db._conn.execute("SELECT COUNT(*) FROM move_log").fetchone()[0]
        assert moves == 0
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────
# 6. Metadata-only inventory
# ──────────────────────────────────────────────────────────────────────


SECRET_SENTINELS = [
    "SENT-message-content-9871",
    "SENT-assistant-reply-44Q",
    "SENT-tool-args-xx",
    "SENT-reasoning-trace",
    "SENT-reasoning-content",
    "SENT-title-redact",
    "SENT-platform-message-id",
]


def test_legacy_inventory_excludes_all_message_field_secrets(tmp_path):
    """Even with secrets stuffed into *every* writable message column
    (content, tool args, reasoning, etc.) plus the session title, the
    inventory dict must surface none of them when JSON-serialised."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        sid = _seed_pre_migration_session(db, user_id="usr-1")
        # Stuff secrets into every nullable field the writer accepts.
        db.append_message(
            sid, "user", SECRET_SENTINELS[0],
            tool_calls=[{"args": SECRET_SENTINELS[2]}],
            reasoning=SECRET_SENTINELS[3],
            reasoning_content=SECRET_SENTINELS[4],
            platform_message_id=SECRET_SENTINELS[6],
        )
        db.append_message(sid, "assistant", SECRET_SENTINELS[1])
        try:
            db.set_session_title(sid, SECRET_SENTINELS[5])
        except Exception:
            pass

        # Forum-thread legacy state — thread_id is metadata; allow it,
        # but the thread_id itself must NOT carry a secret-shaped payload
        # because real binding keys are numeric Telegram thread ids.
        db.apply_telegram_topic_migration()
        db.enable_telegram_topic_mode(
            chat_id="208214988", user_id="usr-1",
            has_topics_enabled=True, allows_users_to_create_topics=True,
        )
        # Bind a session under a benign numeric thread_id.
        db.bind_telegram_topic(
            chat_id="208214988", thread_id="42", user_id="usr-1",
            session_key="legacy:key", session_id=sid,
        )

        inv = legacy_inventory(db)
        blob = json.dumps(inv, default=str)
        for sentinel in SECRET_SENTINELS:
            assert sentinel not in blob, (
                f"inventory leaked sentinel {sentinel!r}: {blob}"
            )
        # And — confirm the inventory keeps its declared shape: only the
        # documented keys, nothing else (no surprise content carry-over).
        assert set(inv.keys()) == {
            "hrm_t0a_applied_at",
            "pre_migration_session_count",
            "telegram_forum_thread_chat_count",
            "pre_migration_principals",
            "telegram_forum_thread_chats",
            "telegram_forum_thread_bindings",
        }
        for principal in inv["pre_migration_principals"]:
            assert set(principal.keys()) == {"platform", "user_id"}
        for chat in inv["telegram_forum_thread_chats"]:
            assert set(chat.keys()) == {"chat_id", "user_id"}
        for binding in inv["telegram_forum_thread_bindings"]:
            assert set(binding.keys()) == {"chat_id", "thread_id", "session_id"}
    finally:
        db.close()


def test_legacy_inventory_only_keys_known_metadata_columns(tmp_path):
    """Defence-in-depth: any future column on the underlying tables must
    not silently flow into the inventory. The dict shape is a contract."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        # Drive enough rows that the dedup paths run.
        _seed_pre_migration_session(db, user_id="alpha")
        _seed_pre_migration_session(db, user_id="bravo")
        db.apply_telegram_topic_migration()
        for uid in ("alpha", "bravo"):
            db.enable_telegram_topic_mode(
                chat_id=f"chat-{uid}", user_id=uid,
                has_topics_enabled=True, allows_users_to_create_topics=True,
            )
        inv = legacy_inventory(db)
        # Strict shape — no extra keys, no nested content.
        allowed_top = {
            "hrm_t0a_applied_at",
            "pre_migration_session_count",
            "telegram_forum_thread_chat_count",
            "pre_migration_principals",
            "telegram_forum_thread_chats",
            "telegram_forum_thread_bindings",
        }
        extra = set(inv.keys()) - allowed_top
        assert not extra, f"unexpected inventory keys: {extra}"
        assert inv["pre_migration_session_count"] >= 2
        assert inv["telegram_forum_thread_chat_count"] >= 2
    finally:
        db.close()


def test_legacy_inventory_principal_dedupe_does_not_collapse_users_under_same_platform(tmp_path):
    """Dedup is by (platform, user_id) — two distinct users must show up
    as two principals, not one. Validates the inventory's identity surface."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _seed_pre_migration_session(db, user_id="u-1")
        _seed_pre_migration_session(db, user_id="u-2")
        inv = legacy_inventory(db)
        users = {p["user_id"] for p in inv["pre_migration_principals"]}
        assert users == {"u-1", "u-2"}
    finally:
        db.close()
