import threading
import time
from collections import defaultdict

from core.config import (
    login_guard_block_seconds,
    login_guard_max_entries,
    login_guard_max_failures,
    login_guard_window_seconds,
)


class LoginGuard:
    """基于 (IP, username) 的登录失败防护。"""

    def __init__(
        self,
        max_failures: int = login_guard_max_failures,
        window_seconds: int = login_guard_window_seconds,
        block_seconds: int = login_guard_block_seconds,
        max_entries: int = login_guard_max_entries,
    ) -> None:
        self.max_failures = max(1, int(max_failures))
        self.window_seconds = max(1, int(window_seconds))
        self.block_seconds = max(1, int(block_seconds))
        self.max_entries = max(1000, int(max_entries))

        self._failures: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._blocked_until: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def _normalize_key(self, ip: str, username: str) -> tuple[str, str]:
        return (str(ip or "unknown").strip(), str(username or "").strip().lower())

    def _trim_failures(self, key: tuple[str, str], now: float) -> None:
        timestamps = self._failures.get(key)
        if not timestamps:
            return

        cutoff = now - self.window_seconds
        valid = [ts for ts in timestamps if ts >= cutoff]
        if valid:
            self._failures[key] = valid
        else:
            self._failures.pop(key, None)

    def _cleanup_expired(self, now: float) -> None:
        expired_blocked = [key for key, ts in self._blocked_until.items() if ts <= now]
        for key in expired_blocked:
            self._blocked_until.pop(key, None)

        if len(self._failures) <= self.max_entries:
            return

        cutoff = now - self.window_seconds
        to_delete: list[tuple[str, str]] = []
        for key, timestamps in self._failures.items():
            valid = [ts for ts in timestamps if ts >= cutoff]
            if valid:
                self._failures[key] = valid
            else:
                to_delete.append(key)

        for key in to_delete:
            self._failures.pop(key, None)

    def get_blocked_seconds(self, ip: str, username: str) -> int:
        now = time.time()
        key = self._normalize_key(ip, username)

        with self._lock:
            self._cleanup_expired(now)
            blocked_until = self._blocked_until.get(key, 0)
            if blocked_until <= now:
                self._blocked_until.pop(key, None)
                return 0
            return int(blocked_until - now)

    def record_failure(self, ip: str, username: str) -> int:
        now = time.time()
        key = self._normalize_key(ip, username)

        with self._lock:
            self._cleanup_expired(now)

            blocked_until = self._blocked_until.get(key, 0)
            if blocked_until > now:
                return int(blocked_until - now)

            self._trim_failures(key, now)
            self._failures[key].append(now)

            if len(self._failures[key]) >= self.max_failures:
                until = now + self.block_seconds
                self._blocked_until[key] = until
                self._failures.pop(key, None)
                return int(self.block_seconds)

            return 0

    def reset(self, ip: str, username: str) -> None:
        key = self._normalize_key(ip, username)
        with self._lock:
            self._failures.pop(key, None)
            self._blocked_until.pop(key, None)
