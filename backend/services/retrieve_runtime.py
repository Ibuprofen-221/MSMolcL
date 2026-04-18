import asyncio
import multiprocessing
from datetime import datetime, timedelta
from queue import Empty, Full
from threading import Event, Lock, Thread
from uuid import uuid4

from core.config import (
    retrieve_max_pending_jobs,
    retrieve_mp_start_method,
    retrieve_queue_wait_timeout,
    retrieve_task_keep_seconds,
)
from services.retrieve import init_retrieve_process
from services.retrieve_service import execute_retrieve_job

DEFAULT_ION_MODE = "pos"
VALID_ION_MODES = ("pos", "neg")


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _normalize_ion_mode(ion_mode: str | None) -> str:
    mode = (ion_mode or DEFAULT_ION_MODE).strip().lower()
    if mode not in VALID_ION_MODES:
        raise ValueError(f"ion_mode 仅支持 {','.join(VALID_ION_MODES)}")
    return mode


def _gpu_worker_main(request_queue, response_queue, ion_mode: str) -> None:
    """GPU专用进程：按离子模式常驻模型，串行处理检索任务。"""
    try:
        init_retrieve_process(ion_mode=ion_mode)
    except Exception as exc:
        response_queue.put(
            {
                "type": "worker_error",
                "ion_mode": ion_mode,
                "error": f"GPU进程模型初始化失败({ion_mode}): {exc}",
                "at": _utc_now_iso(),
            }
        )
        return

    while True:
        try:
            message = request_queue.get(timeout=1.0)
        except Empty:
            continue

        if message is None:
            break

        job_id = message.get("job_id")
        job_payload = message.get("payload")

        if not job_id or not isinstance(job_payload, dict):
            response_queue.put(
                {
                    "type": "done",
                    "job_id": job_id,
                    "ion_mode": ion_mode,
                    "status": "failed",
                    "result": None,
                    "error": "任务消息格式无效",
                    "finished_at": _utc_now_iso(),
                }
            )
            continue

        response_queue.put(
            {
                "type": "running",
                "job_id": job_id,
                "ion_mode": ion_mode,
                "started_at": _utc_now_iso(),
            }
        )

        try:
            result = execute_retrieve_job(job_payload)
            response_queue.put(
                {
                    "type": "done",
                    "job_id": job_id,
                    "ion_mode": ion_mode,
                    "status": "success",
                    "result": result,
                    "error": None,
                    "finished_at": _utc_now_iso(),
                }
            )
        except Exception as exc:
            response_queue.put(
                {
                    "type": "done",
                    "job_id": job_id,
                    "ion_mode": ion_mode,
                    "status": "failed",
                    "result": None,
                    "error": str(exc),
                    "finished_at": _utc_now_iso(),
                }
            )


class RetrieveRuntime:
    def __init__(self) -> None:
        self._worker_processes: dict[str, multiprocessing.Process] = {}
        self._request_queues: dict[str, multiprocessing.Queue] = {}
        self._response_queues: dict[str, multiprocessing.Queue] = {}
        self._consumer_threads: dict[str, Thread] = {}
        self._stop_event = Event()

        self._lock = Lock()
        self._jobs: dict[str, dict] = {}
        self._worker_errors: dict[str, str | None] = {mode: None for mode in VALID_ION_MODES}

    async def startup(self) -> None:
        if self._worker_processes and all(proc.is_alive() for proc in self._worker_processes.values()):
            return

        self._stop_event.clear()
        self._worker_errors = {mode: None for mode in VALID_ION_MODES}

        mp_ctx = multiprocessing.get_context(retrieve_mp_start_method)

        for mode in VALID_ION_MODES:
            request_queue = mp_ctx.Queue(maxsize=retrieve_max_pending_jobs)
            response_queue = mp_ctx.Queue(maxsize=retrieve_max_pending_jobs * 2)

            process = mp_ctx.Process(
                target=_gpu_worker_main,
                args=(request_queue, response_queue, mode),
                daemon=False,
                name=f"retrieve-gpu-worker-{mode}",
            )
            process.start()

            consumer_thread = Thread(
                target=self._consume_responses,
                args=(mode,),
                daemon=True,
                name=f"retrieve-response-consumer-{mode}",
            )
            

            self._request_queues[mode] = request_queue
            self._response_queues[mode] = response_queue
            self._worker_processes[mode] = process
            consumer_thread.start()
            self._consumer_threads[mode] = consumer_thread

        await asyncio.sleep(0.2)
        self._sync_worker_state()
        if all(self._worker_errors.get(mode) for mode in VALID_ION_MODES):
            raise RuntimeError("; ".join(err for err in self._worker_errors.values() if err))

    async def shutdown(self) -> None:
        self._stop_event.set()

        for mode, request_queue in self._request_queues.items():
            try:
                request_queue.put_nowait(None)
            except Exception:
                pass

        for mode, proc in self._worker_processes.items():
            proc.join(timeout=8)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

        for mode, th in self._consumer_threads.items():
            if th.is_alive():
                th.join(timeout=2)

        for queue in self._request_queues.values():
            queue.close()
        for queue in self._response_queues.values():
            queue.close()

        self._request_queues.clear()
        self._response_queues.clear()
        self._worker_processes.clear()
        self._consumer_threads.clear()

        self._mark_active_jobs_failed("检索服务已关闭")

    def submit(self, job_payload: dict, task_id: str) -> dict:
        self._cleanup_finished_locked()
        self._sync_worker_state()

        ion_mode = _normalize_ion_mode(job_payload.get("ion_mode"))
        queue = self._request_queues.get(ion_mode)
        proc = self._worker_processes.get(ion_mode)

        if queue is None or proc is None or not proc.is_alive():
            raise RuntimeError(f"{ion_mode} 离子检索执行器尚未初始化")

        worker_error = self._worker_errors.get(ion_mode)
        if worker_error:
            raise RuntimeError(worker_error)

        with self._lock:
            active_jobs = sum(
                1
                for item in self._jobs.values()
                if item["ion_mode"] == ion_mode and item["status"] in {"pending", "running"}
            )
            if active_jobs >= retrieve_max_pending_jobs:
                raise RuntimeError(f"{ion_mode} 离子检索任务过多，请稍后重试")

            job_id = uuid4().hex
            now = _utc_now_iso()
            self._jobs[job_id] = {
                "job_id": job_id,
                "task_id": task_id,
                "ion_mode": ion_mode,
                "worker_key": ion_mode,
                "status": "pending",
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
            }

        try:
            queue.put_nowait(
                {
                    "job_id": job_id,
                    "payload": job_payload,
                    "enqueue_time": _utc_now_iso(),
                    "ion_mode": ion_mode,
                }
            )
        except Full:
            with self._lock:
                self._jobs.pop(job_id, None)
            raise RuntimeError(f"{ion_mode} 离子检索任务过多，请稍后重试") from None

        return {
            "job_id": job_id,
            "task_id": task_id,
            "ion_mode": ion_mode,
            "worker_key": ion_mode,
            "status": "pending",
        }

    def get_job(self, job_id: str) -> dict | None:
        self._cleanup_finished_locked()
        self._sync_worker_state()

        with self._lock:
            item = self._jobs.get(job_id)
            if item is None:
                return None
            return dict(item)

    def _consume_responses(self, ion_mode: str) -> None:
        while not self._stop_event.is_set():
            response_queue = self._response_queues.get(ion_mode)
            if response_queue is None:
                break

            try:
                message = response_queue.get(timeout=retrieve_queue_wait_timeout)
            except Empty:
                continue
            except Exception:
                continue

            self._handle_worker_message(message, ion_mode)

    def _handle_worker_message(self, message: dict, ion_mode: str) -> None:
        msg_type = message.get("type")
        if msg_type == "worker_error":
            error = str(message.get("error") or f"GPU检索进程异常({ion_mode})")
            self._worker_errors[ion_mode] = error
            self._mark_active_jobs_failed(error, ion_mode=ion_mode)
            return

        job_id = message.get("job_id")
        if not job_id:
            return

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return

            if msg_type == "running":
                job["status"] = "running"
                job["started_at"] = message.get("started_at") or _utc_now_iso()
                return

            if msg_type == "done":
                job["status"] = message.get("status") or "failed"
                job["result"] = message.get("result")
                job["error"] = message.get("error")
                job["finished_at"] = message.get("finished_at") or _utc_now_iso()

    def _sync_worker_state(self) -> None:
        for ion_mode, proc in self._worker_processes.items():
            if proc.is_alive():
                continue

            if self._worker_errors.get(ion_mode) is None and not self._stop_event.is_set():
                error = f"GPU检索进程异常退出({ion_mode})"
                self._worker_errors[ion_mode] = error
                self._mark_active_jobs_failed(error, ion_mode=ion_mode)

    def _mark_active_jobs_failed(self, reason: str, ion_mode: str | None = None) -> None:
        finished_at = _utc_now_iso()
        with self._lock:
            for job in self._jobs.values():
                if ion_mode is not None and job.get("ion_mode") != ion_mode:
                    continue
                if job["status"] in {"pending", "running"}:
                    job["status"] = "failed"
                    job["result"] = None
                    job["error"] = reason
                    job["finished_at"] = finished_at

    def _cleanup_finished_locked(self) -> None:
        expire_at = datetime.utcnow() - timedelta(seconds=retrieve_task_keep_seconds)
        with self._lock:
            expired = []
            for job_id, item in self._jobs.items():
                finished_at = item.get("finished_at")
                if not finished_at:
                    continue
                try:
                    dt = datetime.fromisoformat(finished_at)
                except Exception:
                    continue
                if dt < expire_at:
                    expired.append(job_id)

            for job_id in expired:
                self._jobs.pop(job_id, None)


retrieve_runtime = RetrieveRuntime()
