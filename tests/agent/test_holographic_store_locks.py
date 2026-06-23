"""
Focused tests for MemoryStore intra-process SQLite lock robustness.

Proves:
1. Two MemoryStore instances for the same db path share a single RLock.
2. Instances for different db paths get distinct locks.
3. _execute_write retries and succeeds after a simulated 'database is locked'.
4. _execute_write propagates non-lock OperationalErrors immediately.
5. _execute_write raises after exhausting retries on repeated lock errors.
6. Concurrent writes from two MemoryStore instances on the same db succeed
   without hitting unhandled lock errors (functional integration).
"""

import sqlite3
import threading
import time

import pytest

from plugins.memory.holographic.store import (
    MemoryStore,
    _db_lock_registry,
    _db_lock_registry_mu,
    _WRITE_MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_store(tmp_path, name="test.db") -> MemoryStore:
    return MemoryStore(db_path=str(tmp_path / name))


# ---------------------------------------------------------------------------
# Lock registry — shared vs distinct locks
# ---------------------------------------------------------------------------


class TestSharedLockRegistry:
    def test_same_db_path_shares_lock(self, tmp_path):
        """Two MemoryStore instances pointing at the same file share one RLock."""
        s1 = _fresh_store(tmp_path)
        s2 = _fresh_store(tmp_path)
        try:
            assert s1._lock is s2._lock, (
                "Instances for the same db must share a single RLock; "
                "having separate locks allows intra-process convoy deadlocks"
            )
        finally:
            s1.close()
            s2.close()

    def test_different_db_paths_get_distinct_locks(self, tmp_path):
        """MemoryStore instances for different files have independent locks."""
        s1 = _fresh_store(tmp_path, "a.db")
        s2 = _fresh_store(tmp_path, "b.db")
        try:
            assert s1._lock is not s2._lock
        finally:
            s1.close()
            s2.close()

    def test_registry_uses_resolved_canonical_path(self, tmp_path):
        """Even if db_path is given with different representations, it maps to
        the same registry entry after resolve()."""
        db = tmp_path / "canon.db"
        s1 = MemoryStore(db_path=str(db))
        # Pass the same path but as a Path object this time
        s2 = MemoryStore(db_path=db)
        try:
            assert s1._lock is s2._lock
        finally:
            s1.close()
            s2.close()


# ---------------------------------------------------------------------------
# _execute_write retry semantics
# ---------------------------------------------------------------------------


class TestExecuteWriteRetry:
    def test_retries_once_on_locked_error(self, tmp_path):
        """_execute_write re-invokes the callable after a single 'locked' error."""
        store = _fresh_store(tmp_path)
        try:
            call_count = 0

            def flaky():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise sqlite3.OperationalError("database is locked")
                return "ok"

            result = store._execute_write(flaky)
            assert result == "ok"
            assert call_count == 2
        finally:
            store.close()

    def test_retries_on_busy_error(self, tmp_path):
        """'database is busy' is treated the same as 'locked'."""
        store = _fresh_store(tmp_path)
        try:
            call_count = 0

            def flaky():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise sqlite3.OperationalError("database is busy")
                return 42

            assert store._execute_write(flaky) == 42
            assert call_count == 2
        finally:
            store.close()

    def test_raises_immediately_on_non_lock_error(self, tmp_path):
        """Non-lock OperationalErrors propagate without retry."""
        store = _fresh_store(tmp_path)
        try:
            call_count = 0

            def bad():
                nonlocal call_count
                call_count += 1
                raise sqlite3.OperationalError("no such table: missing")

            with pytest.raises(sqlite3.OperationalError, match="no such table"):
                store._execute_write(bad)

            assert call_count == 1, "Should not retry on non-lock errors"
        finally:
            store.close()

    def test_exhausts_retries_and_raises(self, tmp_path):
        """_execute_write raises after _WRITE_MAX_RETRIES consecutive lock errors."""
        store = _fresh_store(tmp_path)
        try:
            call_count = 0

            def always_locked():
                nonlocal call_count
                call_count += 1
                raise sqlite3.OperationalError("database is locked")

            # Speed the test up by patching sleep
            import unittest.mock as mock
            with mock.patch("plugins.memory.holographic.store.time.sleep"):
                with pytest.raises(sqlite3.OperationalError):
                    store._execute_write(always_locked)

            assert call_count == _WRITE_MAX_RETRIES
        finally:
            store.close()

    def test_returns_callable_result(self, tmp_path):
        """_execute_write passes through the callable's return value."""
        store = _fresh_store(tmp_path)
        try:
            assert store._execute_write(lambda: {"key": "value"}) == {"key": "value"}
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Integration: two instances write concurrently to the same db
# ---------------------------------------------------------------------------


class TestConcurrentInstanceWrites:
    def test_two_instances_same_db_no_unhandled_lock(self, tmp_path):
        """Concurrent add_fact calls from two MemoryStore instances on the same
        db complete without raising OperationalError.

        This exercises the shared-lock + busy_timeout + retry path end-to-end.
        """
        db = tmp_path / "concurrent.db"
        errors: list[Exception] = []
        fact_ids: list[int] = []

        def writer(store: MemoryStore, facts: list[str]) -> None:
            try:
                for f in facts:
                    fact_ids.append(store.add_fact(f))
            except Exception as exc:
                errors.append(exc)

        s1 = MemoryStore(db_path=str(db))
        s2 = MemoryStore(db_path=str(db))
        try:
            batch1 = [f"fact from s1 number {i}" for i in range(10)]
            batch2 = [f"fact from s2 number {i}" for i in range(10)]

            t1 = threading.Thread(target=writer, args=(s1, batch1))
            t2 = threading.Thread(target=writer, args=(s2, batch2))
            t1.start()
            t2.start()
            t1.join(timeout=30)
            t2.join(timeout=30)

            assert not errors, f"Concurrent writes raised: {errors}"
            assert len(fact_ids) == 20
        finally:
            s1.close()
            s2.close()

    def test_add_fact_is_idempotent_across_instances(self, tmp_path):
        """Adding the same content via two different MemoryStore instances
        returns the same fact_id (UNIQUE-constraint dedup path)."""
        db = tmp_path / "dedup.db"
        s1 = MemoryStore(db_path=str(db))
        s2 = MemoryStore(db_path=str(db))
        try:
            fid1 = s1.add_fact("Alice prefers dark mode")
            fid2 = s2.add_fact("Alice prefers dark mode")
            assert fid1 == fid2
        finally:
            s1.close()
            s2.close()
