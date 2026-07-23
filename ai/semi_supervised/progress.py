"""Durable, flush-safe progress reporting for long semi-supervised phases."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class ProgressReporter:
    def __init__(
        self,
        phase: str,
        *,
        total: int | None,
        log_path: Path,
        every_items: int = 10,
        heartbeat_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.phase = phase
        self.total = total
        self.log_path = Path(log_path)
        self.every_items = max(1, int(every_items))
        self.heartbeat_seconds = max(0.0, float(heartbeat_seconds))
        self.clock = clock
        self.started_at = 0.0
        self.current = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _emit(self, event: str, detail: str = "") -> None:
        now = self.clock()
        elapsed = max(0.0, now - self.started_at) if self.started_at else 0.0
        rate = self.current / elapsed if elapsed > 0 else 0.0
        eta = None
        if self.total is not None and rate > 0:
            eta = max(0.0, (self.total - self.current) / rate)
        progress = (
            f"{self.current}/{self.total}"
            if self.total is not None
            else str(self.current)
        )
        message = (
            f"[{event}] phase={self.phase} | progress={progress} | "
            f"elapsed={elapsed:.1f}s | rate={rate:.3f}/s | "
            f"ETA={eta:.1f}s" if eta is not None else
            f"[{event}] phase={self.phase} | progress={progress} | "
            f"elapsed={elapsed:.1f}s | rate={rate:.3f}/s | ETA=unknown"
        )
        if detail:
            message += f" | {detail}"
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "phase": self.phase,
            "current": self.current,
            "total": self.total,
            "elapsed_seconds": round(elapsed, 3),
            "items_per_second": round(rate, 6),
            "eta_seconds": None if eta is None else round(eta, 3),
            "detail": detail,
        }
        with self._lock:
            print(message, flush=True)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self._emit("HEARTBEAT", "phase is still running")

    def start(self, *, detail: str = "") -> None:
        self.started_at = self.clock()
        self._emit("PHASE START", detail)
        if self.heartbeat_seconds > 0:
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"progress-{self.phase}",
                daemon=True,
            )
            self._thread.start()

    def advance(self, current: int, *, detail: str = "") -> None:
        self.current = int(current)
        if self.current == 1 or self.current % self.every_items == 0 or (
            self.total is not None and self.current == self.total
        ):
            self._emit("PROGRESS", detail)

    def complete(self, *, detail: str = "") -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.heartbeat_seconds + 1.0))
        self._emit("PHASE COMPLETE", detail)

    def fail(self, error: BaseException) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.heartbeat_seconds + 1.0))
        self._emit("PHASE FAILED", f"{type(error).__name__}: {error}")

