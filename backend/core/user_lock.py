from threading import Lock


class UserLockManager:
    """按用户 data_path 维度管理写锁，防止同一用户并发写入历史记录。"""

    def __init__(self):
        self._global_lock = Lock()
        self._locks: dict[str, Lock] = {}

    def acquire(self, key: str) -> Lock:
        lock = self._locks.get(key)
        if lock is not None:
            return lock
        with self._global_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = Lock()
                self._locks[key] = lock
        return lock


user_lock_manager = UserLockManager()
