"""
Tests for GET /v1/topics/{app_id}/active and POST /v1/topics/switch.

Covers: auth, read-unset, switch-registered, reject-unregistered,
reject-no-checker, round-trip after switch, malformed payload.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.active_topic as active_topic_module
from gateway.active_topic import set_registered_check
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra: dict = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/topics/{app_id}/active", adapter._handle_get_active_topic)
    app.router.add_post("/v1/topics/switch", adapter._handle_topic_switch)
    return app


def _make_real_db(tmp_path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


@pytest.fixture(autouse=True)
def _reset_active_topic_state():
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)
    yield
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------


class TestTopicAuth:
    @pytest.mark.asyncio
    async def test_get_active_topic_requires_auth(self):
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/topics/myapp/active")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_get_active_topic_accepts_valid_key(self, tmp_path):
        adapter = _make_adapter(api_key="sk-secret")
        adapter._session_db = _make_real_db(tmp_path)
        # No checker wired — read-only, should not fail
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get(
                    "/v1/topics/myapp/active",
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 200
        finally:
            adapter._session_db.close()

    @pytest.mark.asyncio
    async def test_switch_requires_auth(self):
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/topics/switch",
                json={"app_id": "myapp", "topic_id": "t1"},
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_switch_accepts_valid_key(self, tmp_path):
        adapter = _make_adapter(api_key="sk-secret")
        adapter._session_db = _make_real_db(tmp_path)
        set_registered_check(AsyncMock(return_value=True))
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "myapp", "topic_id": "t1"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 200
        finally:
            adapter._session_db.close()


# ---------------------------------------------------------------------------
# GET /v1/topics/{app_id}/active
# ---------------------------------------------------------------------------


class TestGetActiveTopic:
    @pytest.mark.asyncio
    async def test_returns_null_when_no_pointer_set(self, tmp_path):
        adapter = _make_adapter()
        adapter._session_db = _make_real_db(tmp_path)
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/topics/hermes-agent/active")
                assert resp.status == 200
                data = await resp.json()
                assert data["app_id"] == "hermes-agent"
                assert data["active_topic"] is None
                assert data["reason"] == "no_topic_set"
        finally:
            adapter._session_db.close()

    @pytest.mark.asyncio
    async def test_returns_pointer_after_switch(self, tmp_path):
        adapter = _make_adapter()
        db = _make_real_db(tmp_path)
        adapter._session_db = db
        set_registered_check(AsyncMock(return_value=True))
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                # Switch first via the POST handler
                sw = await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "hermes-agent", "topic_id": "research"},
                )
                assert sw.status == 200

                # Now read back
                resp = await cli.get("/v1/topics/hermes-agent/active")
                assert resp.status == 200
                data = await resp.json()
                assert data["active_topic"] is not None
                assert data["active_topic"]["topic_id"] == "research"
                assert data["reason"] is None
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_db_unavailable_returns_503(self):
        adapter = _make_adapter()
        adapter._session_db = None
        with patch.object(adapter, "_ensure_session_db", return_value=None):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/topics/myapp/active")
                assert resp.status == 503


# ---------------------------------------------------------------------------
# POST /v1/topics/switch
# ---------------------------------------------------------------------------


class TestTopicSwitch:
    @pytest.mark.asyncio
    async def test_switch_registered_topic_returns_200(self, tmp_path):
        adapter = _make_adapter()
        adapter._session_db = _make_real_db(tmp_path)
        set_registered_check(AsyncMock(return_value=True))
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "hermes-agent", "topic_id": "research"},
                )
                assert resp.status == 200
                data = await resp.json()
                assert data["ok"] is True
                assert data["app_id"] == "hermes-agent"
                assert data["topic_id"] == "research"
                assert data["prior"] is None  # first switch, no prior
        finally:
            adapter._session_db.close()

    @pytest.mark.asyncio
    async def test_switch_carries_source_field(self, tmp_path):
        """source= is passed through as updated_by on the stored pointer row."""
        adapter = _make_adapter()
        db = _make_real_db(tmp_path)
        adapter._session_db = db
        set_registered_check(AsyncMock(return_value=True))
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "hermes-agent", "topic_id": "planning", "source": "nl_directive"},
                )
                assert resp.status == 200

                # Verify the stored pointer carries the source in updated_by
                rd = await cli.get("/v1/topics/hermes-agent/active")
                data = await rd.json()
                assert data["active_topic"] is not None
                assert "nl_directive" in data["active_topic"]["updated_by"]
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_switch_unregistered_topic_returns_400(self, tmp_path):
        adapter = _make_adapter()
        adapter._session_db = _make_real_db(tmp_path)
        set_registered_check(AsyncMock(return_value=False))
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "hermes-agent", "topic_id": "ghost-topic"},
                )
                assert resp.status == 400
                data = await resp.json()
                assert data["error"]["code"] == "topic_not_registered"
        finally:
            adapter._session_db.close()

    @pytest.mark.asyncio
    async def test_switch_no_checker_wired_returns_400(self, tmp_path):
        """No registry checker wired → fails closed → 400, not 500."""
        adapter = _make_adapter()
        adapter._session_db = _make_real_db(tmp_path)
        # _reset_active_topic_state leaves checker as None
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "hermes-agent", "topic_id": "research"},
                )
                assert resp.status == 400
                data = await resp.json()
                assert data["error"]["code"] == "topic_not_registered"
        finally:
            adapter._session_db.close()

    @pytest.mark.asyncio
    async def test_missing_app_id_returns_422(self, tmp_path):
        adapter = _make_adapter()
        adapter._session_db = _make_real_db(tmp_path)
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/topics/switch",
                    json={"topic_id": "research"},
                )
                assert resp.status == 422
                data = await resp.json()
                assert data["error"]["param"] == "app_id"
        finally:
            adapter._session_db.close()

    @pytest.mark.asyncio
    async def test_missing_topic_id_returns_422(self, tmp_path):
        adapter = _make_adapter()
        adapter._session_db = _make_real_db(tmp_path)
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "hermes-agent"},
                )
                assert resp.status == 422
                data = await resp.json()
                assert data["error"]["param"] == "topic_id"
        finally:
            adapter._session_db.close()

    @pytest.mark.asyncio
    async def test_malformed_json_returns_400(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/topics/switch",
                data=b"not-json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_db_unavailable_returns_503(self):
        adapter = _make_adapter()
        with patch.object(adapter, "_ensure_session_db", return_value=None):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "myapp", "topic_id": "t1"},
                )
                assert resp.status == 503


# ---------------------------------------------------------------------------
# Round-trip: switch then read
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_switch_then_read_matches(self, tmp_path):
        adapter = _make_adapter()
        db = _make_real_db(tmp_path)
        adapter._session_db = db
        set_registered_check(AsyncMock(return_value=True))
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                sw = await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "hermes-agent", "topic_id": "sprint-42"},
                )
                assert sw.status == 200

                rd = await cli.get("/v1/topics/hermes-agent/active")
                assert rd.status == 200
                data = await rd.json()
                assert data["active_topic"]["topic_id"] == "sprint-42"

                # Second switch records the prior
                sw2 = await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "hermes-agent", "topic_id": "backlog"},
                )
                assert sw2.status == 200
                sw2_data = await sw2.json()
                assert sw2_data["prior"] is not None
                assert sw2_data["prior"]["topic_id"] == "sprint-42"

                # Read reflects the new pointer
                rd2 = await cli.get("/v1/topics/hermes-agent/active")
                rd2_data = await rd2.json()
                assert rd2_data["active_topic"]["topic_id"] == "backlog"
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_different_app_ids_are_isolated(self, tmp_path):
        """Two app_ids on the same cockpit principal don't share a pointer."""
        adapter = _make_adapter()
        db = _make_real_db(tmp_path)
        adapter._session_db = db
        set_registered_check(AsyncMock(return_value=True))
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "app-a", "topic_id": "topic-a"},
                )
                await cli.post(
                    "/v1/topics/switch",
                    json={"app_id": "app-b", "topic_id": "topic-b"},
                )

                rd_a = await cli.get("/v1/topics/app-a/active")
                rd_b = await cli.get("/v1/topics/app-b/active")
                data_a = await rd_a.json()
                data_b = await rd_b.json()
                assert data_a["active_topic"]["topic_id"] == "topic-a"
                assert data_b["active_topic"]["topic_id"] == "topic-b"
        finally:
            db.close()
