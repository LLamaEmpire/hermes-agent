"""HRM-T0a JSON topic registry checker.

Reads <registry_root>/<app_id>.json with the format:
    {"app_id": str, "version": 1, "topics": {"topic_id": {...}}}

Returns an async callable (app_id, topic_id) -> bool suitable for passing to
gateway.active_topic.set_registered_check().  All failure modes (missing file,
malformed JSON, schema mismatch, app_id mismatch, missing topic) fail closed
and log at WARNING — they never raise into the caller and never crash gateway
startup.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


def make_json_registry_checker(
    registry_root: Path,
    app_id: str,
) -> Callable[[str, str], Awaitable[bool]]:
    """Return an async checker bound to *registry_root* and *app_id*.

    The returned callable matches set_registered_check's contract:
        async (app_id: str, topic_id: str) -> bool
    """
    _root = Path(registry_root)
    _app_id = app_id

    async def _check(caller_app_id: str, topic_id: str) -> bool:
        if caller_app_id != _app_id:
            logger.warning(
                "topic_registry: app_id mismatch — checker bound to %r but called with %r",
                _app_id,
                caller_app_id,
            )
            return False

        registry_file = _root / f"{_app_id}.json"
        try:
            text = registry_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(
                "topic_registry: registry file not found: %s — failing closed",
                registry_file,
            )
            return False
        except OSError as exc:
            logger.warning(
                "topic_registry: cannot read %s: %s — failing closed",
                registry_file,
                exc,
            )
            return False

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "topic_registry: malformed JSON in %s: %s — failing closed",
                registry_file,
                exc,
            )
            return False

        if not isinstance(data, dict):
            logger.warning(
                "topic_registry: %s root is not a JSON object — failing closed",
                registry_file,
            )
            return False

        file_app_id = data.get("app_id")
        if file_app_id != _app_id:
            logger.warning(
                "topic_registry: %s declares app_id %r but checker expects %r — failing closed",
                registry_file,
                file_app_id,
                _app_id,
            )
            return False

        topics = data.get("topics")
        if not isinstance(topics, dict):
            logger.warning(
                "topic_registry: %s 'topics' field is not a JSON object — failing closed",
                registry_file,
            )
            return False

        return topic_id in topics

    return _check
