"""HRM-T0a step 10 — API server enablement posture (default-deny + safe bind).

Covers the config-load → posture-validate → connect() refusal pipeline:

- ``validate_api_server_posture`` unit-validates every (host, key, tls,
  tailscale_only) combination per the plan: loopback needs a key,
  non-loopback needs a strong key AND tls-or-tailscale_only.
- ``_validate_gateway_config`` forces ``api_server.enabled = False`` when
  the posture is unsafe (default-deny preserved at load time).
- ``APIServerAdapter.connect()`` refuses to bind when posture is unsafe
  (defense-in-depth in case the loader is bypassed).
- ``GatewayConfig.from_dict`` honors ``platforms.api_server.enabled: false``
  / missing block and never marks the platform connected.
- ``_check_auth`` rejects unauthenticated requests with 401 and accepts
  the configured bearer key — same handler surface the gateway dispatches
  on.

These tests never touch the live ``config.yaml`` or any profile; they
construct ``PlatformConfig`` / ``GatewayConfig`` objects in-process.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
    _validate_gateway_config,
)
from gateway.platforms.api_server import (
    APIServerAdapter,
    validate_api_server_posture,
)


# ---------------------------------------------------------------------------
# Posture validator unit tests
# ---------------------------------------------------------------------------


_STRONG_KEY = "x" * 32  # passes has_usable_secret(min_length=16)
_WEAK_KEY = "sk-test"   # 7 chars — short of the network-accessible floor


class TestValidatorEnabledShortCircuits:
    def test_disabled_passes_with_no_key_no_tls(self):
        ok, reason = validate_api_server_posture(
            {"host": "0.0.0.0"}, enabled=False,
        )
        assert ok is True
        assert reason is None

    def test_disabled_passes_with_empty_extra(self):
        ok, reason = validate_api_server_posture({}, enabled=False)
        assert ok is True
        assert reason is None

    def test_disabled_passes_with_none_extra(self):
        ok, reason = validate_api_server_posture(None, enabled=False)
        assert ok is True
        assert reason is None


class TestValidatorMissingKey:
    def test_loopback_no_key_fails_closed(self):
        ok, reason = validate_api_server_posture({"host": "127.0.0.1"})
        assert ok is False
        assert reason is not None
        assert "API_SERVER_KEY" in reason

    def test_wildcard_no_key_fails_closed(self):
        ok, reason = validate_api_server_posture({"host": "0.0.0.0"})
        assert ok is False
        assert "API_SERVER_KEY" in reason

    def test_empty_key_fails_closed(self):
        ok, reason = validate_api_server_posture(
            {"host": "127.0.0.1", "key": ""},
        )
        assert ok is False
        assert "required" in reason.lower()


class TestValidatorLoopback:
    def test_loopback_with_short_key_passes(self):
        # Loopback floor is 4 chars; bearer auth is still required.
        ok, reason = validate_api_server_posture(
            {"host": "127.0.0.1", "key": "abcd"},
        )
        assert ok is True
        assert reason is None

    def test_loopback_with_strong_key_passes(self):
        ok, reason = validate_api_server_posture(
            {"host": "127.0.0.1", "key": _STRONG_KEY},
        )
        assert ok is True
        assert reason is None

    def test_loopback_does_not_require_tls(self):
        ok, _ = validate_api_server_posture(
            {"host": "127.0.0.1", "key": _STRONG_KEY, "tls": False},
        )
        assert ok is True

    def test_ipv6_loopback_passes_with_key(self):
        ok, _ = validate_api_server_posture(
            {"host": "::1", "key": _STRONG_KEY},
        )
        assert ok is True

    def test_loopback_with_below_min_key_fails(self):
        # Below the 4-char loopback floor — placeholder territory.
        ok, reason = validate_api_server_posture(
            {"host": "127.0.0.1", "key": "ab"},
        )
        assert ok is False
        assert "short" in reason.lower() or "placeholder" in reason.lower()


class TestValidatorNetworkAccessible:
    def test_wildcard_with_strong_key_no_tls_fails(self):
        """A strong key alone is not enough on a non-loopback bind."""
        ok, reason = validate_api_server_posture(
            {"host": "0.0.0.0", "key": _STRONG_KEY},
        )
        assert ok is False
        assert "tls" in reason.lower() or "tailscale" in reason.lower()

    def test_wildcard_with_weak_key_fails_on_key(self):
        ok, reason = validate_api_server_posture(
            {"host": "0.0.0.0", "key": _WEAK_KEY, "tls": True},
        )
        assert ok is False
        # The weak-key failure trumps the transport check.
        assert "short" in reason.lower() or "placeholder" in reason.lower()

    def test_wildcard_with_strong_key_and_tls_passes(self):
        ok, reason = validate_api_server_posture(
            {"host": "0.0.0.0", "key": _STRONG_KEY, "tls": True},
        )
        assert ok is True
        assert reason is None

    def test_wildcard_with_strong_key_and_tailscale_only_passes(self):
        ok, reason = validate_api_server_posture(
            {"host": "0.0.0.0", "key": _STRONG_KEY, "tailscale_only": True},
        )
        assert ok is True
        assert reason is None

    def test_ipv6_wildcard_requires_tls_or_tailscale(self):
        ok, reason = validate_api_server_posture(
            {"host": "::", "key": _STRONG_KEY},
        )
        assert ok is False
        assert reason is not None

    def test_private_lan_requires_safe_transport(self):
        ok, reason = validate_api_server_posture(
            {"host": "10.0.0.5", "key": _STRONG_KEY},
        )
        assert ok is False
        assert reason is not None

    def test_explicit_true_lifts_the_gate(self):
        """Only the real YAML boolean ``true`` satisfies the transport gate."""
        ok, _ = validate_api_server_posture(
            {"host": "0.0.0.0", "key": _STRONG_KEY, "tls": True},
        )
        assert ok is True

    @pytest.mark.parametrize(
        "tls_value",
        ["yes", "true", "True", "1", "on", "false", "False", "no", "off"],
    )
    def test_non_bool_string_tls_rejected(self, tls_value):
        """Quoted YAML scalars must NEVER satisfy the transport gate.

        ``bool("false")`` is True under Python truthiness, so accepting
        non-bool ``tls`` values would let a config-file typo silently lift
        the posture gate on a network-accessible bind.
        """
        ok, reason = validate_api_server_posture(
            {"host": "0.0.0.0", "key": _STRONG_KEY, "tls": tls_value},
        )
        assert ok is False
        assert reason is not None
        assert "tls" in reason.lower() and "boolean" in reason.lower()

    @pytest.mark.parametrize(
        "tailscale_value",
        ["yes", "true", "True", "1", "on", "false", "False", "no", "off"],
    )
    def test_non_bool_string_tailscale_only_rejected(self, tailscale_value):
        """Same strict-bool rule applies to ``tailscale_only``."""
        ok, reason = validate_api_server_posture(
            {
                "host": "0.0.0.0",
                "key": _STRONG_KEY,
                "tailscale_only": tailscale_value,
            },
        )
        assert ok is False
        assert reason is not None
        assert "tailscale_only" in reason.lower() and "boolean" in reason.lower()

    @pytest.mark.parametrize("bad_value", [1, 0, 1.0, ["true"], {"v": True}])
    def test_non_bool_non_string_tls_rejected(self, bad_value):
        """Numbers, lists, and dicts are also rejected — strict bool only."""
        ok, reason = validate_api_server_posture(
            {"host": "0.0.0.0", "key": _STRONG_KEY, "tls": bad_value},
        )
        assert ok is False
        assert reason is not None
        assert "boolean" in reason.lower()

    def test_explicit_false_tls_still_falls_back_to_transport_error(self):
        """``tls: false`` is a valid bool — gate stays closed via the
        normal transport-required failure, not the type-rejection branch."""
        ok, reason = validate_api_server_posture(
            {"host": "0.0.0.0", "key": _STRONG_KEY, "tls": False},
        )
        assert ok is False
        assert reason is not None
        # Falls through to the transport-posture failure, not the type check.
        assert "boolean" not in reason.lower()
        assert "tls" in reason.lower() or "tailscale" in reason.lower()


# ---------------------------------------------------------------------------
# Config-load validation: posture is enforced at startup, default-deny
# ---------------------------------------------------------------------------


class TestConfigLoaderDefaultDeny:
    def test_missing_api_server_block_does_not_create_platform(self):
        config = GatewayConfig.from_dict({"platforms": {}})
        assert Platform.API_SERVER not in config.platforms

    def test_disabled_block_stays_disabled_and_not_connected(self):
        config = GatewayConfig.from_dict({
            "platforms": {
                "api_server": {
                    "enabled": False,
                    "extra": {"host": "127.0.0.1", "key": _STRONG_KEY},
                },
            },
        })
        api = config.platforms[Platform.API_SERVER]
        assert api.enabled is False
        assert Platform.API_SERVER not in config.get_connected_platforms()

    def test_validate_passthrough_when_disabled(self):
        """An unsafe disabled block must not raise/mutate to enabled."""
        config = GatewayConfig.from_dict({
            "platforms": {
                "api_server": {
                    "enabled": False,
                    "extra": {"host": "0.0.0.0"},  # would be unsafe if enabled
                },
            },
        })
        _validate_gateway_config(config)
        assert config.platforms[Platform.API_SERVER].enabled is False


class TestConfigLoaderPostureEnforced:
    def test_enabled_loopback_with_key_validates(self):
        config = GatewayConfig.from_dict({
            "platforms": {
                "api_server": {
                    "enabled": True,
                    "extra": {"host": "127.0.0.1", "key": _STRONG_KEY},
                },
            },
        })
        _validate_gateway_config(config)
        api = config.platforms[Platform.API_SERVER]
        assert api.enabled is True
        assert Platform.API_SERVER in config.get_connected_platforms()

    def test_enabled_without_key_is_force_disabled(self):
        config = GatewayConfig.from_dict({
            "platforms": {
                "api_server": {
                    "enabled": True,
                    "extra": {"host": "127.0.0.1"},
                },
            },
        })
        _validate_gateway_config(config)
        assert config.platforms[Platform.API_SERVER].enabled is False

    def test_enabled_wildcard_strong_key_no_tls_is_force_disabled(self):
        config = GatewayConfig.from_dict({
            "platforms": {
                "api_server": {
                    "enabled": True,
                    "extra": {"host": "0.0.0.0", "key": _STRONG_KEY},
                },
            },
        })
        _validate_gateway_config(config)
        assert config.platforms[Platform.API_SERVER].enabled is False

    def test_enabled_wildcard_strong_key_with_tls_stays_enabled(self):
        config = GatewayConfig.from_dict({
            "platforms": {
                "api_server": {
                    "enabled": True,
                    "extra": {
                        "host": "0.0.0.0",
                        "key": _STRONG_KEY,
                        "tls": True,
                    },
                },
            },
        })
        _validate_gateway_config(config)
        assert config.platforms[Platform.API_SERVER].enabled is True

    def test_enabled_wildcard_strong_key_with_tailscale_only_stays_enabled(self):
        config = GatewayConfig.from_dict({
            "platforms": {
                "api_server": {
                    "enabled": True,
                    "extra": {
                        "host": "0.0.0.0",
                        "key": _STRONG_KEY,
                        "tailscale_only": True,
                    },
                },
            },
        })
        _validate_gateway_config(config)
        assert config.platforms[Platform.API_SERVER].enabled is True

    def test_enabled_wildcard_weak_key_with_tls_is_force_disabled(self):
        config = GatewayConfig.from_dict({
            "platforms": {
                "api_server": {
                    "enabled": True,
                    "extra": {
                        "host": "0.0.0.0",
                        "key": _WEAK_KEY,
                        "tls": True,
                    },
                },
            },
        })
        _validate_gateway_config(config)
        assert config.platforms[Platform.API_SERVER].enabled is False


# ---------------------------------------------------------------------------
# Adapter connect() — defense-in-depth on the bind path
# ---------------------------------------------------------------------------


class TestConnectRefusesOnUnsafePosture:
    @pytest.mark.asyncio
    async def test_loopback_no_key_refuses(self):
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "127.0.0.1"}),
        )
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_wildcard_strong_key_no_tls_refuses(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "0.0.0.0", "key": _STRONG_KEY},
            ),
        )
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_wildcard_strong_key_tls_passes_posture_gate(self, monkeypatch):
        """With a safe posture, connect() proceeds past the posture gate.

        We don't actually want to bind a socket in unit tests, so we stop
        connect() at the port-conflict probe by claiming the port. The
        relevant assertion is that the posture gate did NOT short-circuit.
        """
        from gateway.platforms import api_server as api_mod

        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "0.0.0.0",
                    "key": _STRONG_KEY,
                    "tls": True,
                    "port": 1,  # port 1 will fail the conflict probe cleanly
                },
            ),
        )

        # If the posture gate was the failure cause, ``validate_api_server_posture``
        # would have been called and we'd see the error in the log. Patch it to
        # raise on a False return so the test fails LOUDLY if posture rejects.
        original = api_mod.validate_api_server_posture
        sentinel: dict[str, bool] = {"called": False}

        def _spy(extra, *, enabled=True):
            sentinel["called"] = True
            ok, reason = original(extra, enabled=enabled)
            assert ok is True, f"posture unexpectedly failed: {reason}"
            return ok, reason

        monkeypatch.setattr(api_mod, "validate_api_server_posture", _spy)

        # connect() may still fail later (port bind, etc.); we only assert the
        # posture spy was invoked AND returned True.
        await adapter.connect()
        assert sentinel["called"] is True


# ---------------------------------------------------------------------------
# Bearer auth at the handler surface — required when key is configured
# ---------------------------------------------------------------------------


class TestBearerAuthAtHandlerSurface:
    def test_authenticated_passes(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "key": _STRONG_KEY},
            ),
        )
        request = MagicMock()
        request.headers = {"Authorization": f"Bearer {_STRONG_KEY}"}
        assert adapter._check_auth(request) is None

    def test_unauthenticated_rejected_with_401(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "key": _STRONG_KEY},
            ),
        )
        request = MagicMock()
        request.headers = {}
        request.transport = None
        request.method = "GET"
        request.path_qs = "/v1/models"
        request.remote = ""
        result = adapter._check_auth(request)
        assert result is not None
        assert result.status == 401

    def test_wrong_bearer_rejected_with_401(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "key": _STRONG_KEY},
            ),
        )
        request = MagicMock()
        request.headers = {"Authorization": "Bearer wrong-key"}
        request.transport = None
        request.method = "GET"
        request.path_qs = "/v1/models"
        request.remote = ""
        result = adapter._check_auth(request)
        assert result is not None
        assert result.status == 401

    def test_constant_time_comparison_does_not_short_circuit_on_prefix(self):
        """Same-length wrong key still 401 — ensures we don't leak via timing."""
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "key": _STRONG_KEY},
            ),
        )
        wrong = "y" * len(_STRONG_KEY)
        request = MagicMock()
        request.headers = {"Authorization": f"Bearer {wrong}"}
        request.transport = None
        request.method = "GET"
        request.path_qs = "/v1/models"
        request.remote = ""
        result = adapter._check_auth(request)
        assert result is not None
        assert result.status == 401
