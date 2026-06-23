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
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

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


# ── Mutable registry helpers ──────────────────────────────────────────────

TOPIC_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def list_topics_in_registry(
    registry_root: Path,
    app_id: str,
) -> List[str]:
    """Return a sorted list of topic_ids from the registry file for *app_id*.

    Raises on any failure (missing file, malformed JSON, schema mismatch) so
    callers can give the user a concrete error message.
    """
    registry_file = Path(registry_root) / f"{app_id}.json"
    try:
        text = registry_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"no registry file for app {app_id!r} at {registry_file}")
    except OSError as exc:
        raise ValueError(f"cannot read {registry_file}: {exc}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {registry_file}: {exc}")
    if not isinstance(data, dict):
        raise ValueError(f"registry {registry_file} root is not a JSON object")
    if data.get("app_id") != app_id:
        raise ValueError(
            f"registry {registry_file} declares app_id {data.get('app_id')!r}, "
            f"expected {app_id!r}"
        )
    topics = data.get("topics")
    if not isinstance(topics, dict):
        raise ValueError(f"registry {registry_file} 'topics' field is not a JSON object")
    return sorted(topics.keys())


def add_topic_to_registry(
    registry_root: Path,
    app_id: str,
    topic_id: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add *topic_id* to the registry file for *app_id* atomically.

    Idempotent: if *topic_id* already exists the file is not written and the
    return value has ``"created": False``.  On success: ``"created": True``.

    Raises :exc:`ValueError` for invalid slugs, missing files, or schema
    mismatches — never silently corrupts the registry.

    Safety:
    - Write constrained to ``<registry_root>/<app_id>.json`` — no path
      traversal because ``app_id`` is validated against the file's own field.
    - Slug must match :data:`TOPIC_SLUG_RE` (``[a-z0-9][a-z0-9_-]*``).
    - Atomic write: temp file in same dir, then ``os.replace()``.
    """
    if not TOPIC_SLUG_RE.match(topic_id):
        raise ValueError(
            f"invalid topic slug {topic_id!r}: must match [a-z0-9][a-z0-9_-]* "
            "(lowercase, digits, hyphens, underscores only)"
        )
    registry_file = Path(registry_root) / f"{app_id}.json"
    try:
        text = registry_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"no registry file for app {app_id!r} at {registry_file}")
    except OSError as exc:
        raise ValueError(f"cannot read {registry_file}: {exc}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {registry_file}: {exc}")
    if not isinstance(data, dict):
        raise ValueError(f"registry {registry_file} root is not a JSON object")
    if data.get("app_id") != app_id:
        raise ValueError(
            f"registry {registry_file} declares app_id {data.get('app_id')!r}, "
            f"expected {app_id!r}"
        )
    topics = data.get("topics")
    if not isinstance(topics, dict):
        raise ValueError(f"registry {registry_file} 'topics' field is not a JSON object")
    if topic_id in topics:
        return {"created": False, "app_id": app_id, "topic_id": topic_id}
    data["topics"][topic_id] = metadata or {}
    parent = registry_file.parent
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(parent), prefix=f".{app_id}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, str(registry_file))
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return {"created": True, "app_id": app_id, "topic_id": topic_id}


def make_multi_app_registry_checker(
    registry_root: Path,
) -> Callable[[str, str], Awaitable[bool]]:
    """Return an async checker that handles any app_id from *registry_root*.

    Unlike :func:`make_json_registry_checker`, this checker is not bound to a
    specific ``app_id`` at construction time.  For each call it reads
    ``<registry_root>/<caller_app_id>.json``, making it suitable for the API
    server where ``app_id`` comes dynamically from the request body.

    All failure modes (missing file, malformed JSON, schema mismatch,
    ``app_id`` field mismatch in the file) fail closed and log at WARNING.
    """
    _root = Path(registry_root)

    async def _check(caller_app_id: str, topic_id: str) -> bool:
        registry_file = _root / f"{caller_app_id}.json"
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
        if file_app_id != caller_app_id:
            logger.warning(
                "topic_registry: %s declares app_id %r but request used %r — failing closed",
                registry_file,
                file_app_id,
                caller_app_id,
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
