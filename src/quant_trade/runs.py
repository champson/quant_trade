from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from quant_trade.config import AppConfig
from quant_trade.data.storage import DataStore


class RunTracker:
    def __init__(self, config: AppConfig, store: DataStore, task: str, as_of: str):
        self.config, self.store, self.task, self.as_of = config, store, task, as_of
        self.run_id = f"{datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self.started = datetime.now()

    def finish(self, status: str, details: dict | None = None) -> None:
        details = details or {}
        payload = {
            "run_id": self.run_id,
            "task": self.task,
            "as_of": self.as_of,
            "started_at": self.started.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "status": status,
            "details": details,
        }
        path = self.config.paths.runs_dir / f"{self.run_id}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        with self.store.connect() as con:
            con.execute("DELETE FROM runs WHERE run_id = ?", [self.run_id])
            con.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    self.run_id,
                    self.task,
                    self.started,
                    datetime.now(),
                    status,
                    self.as_of,
                    json.dumps(self.config.model_dump(mode="json"), ensure_ascii=False),
                    json.dumps(details, ensure_ascii=False, default=str),
                ],
            )


@contextmanager
def tracked_run(config: AppConfig, store: DataStore, task: str, as_of: str) -> Iterator[RunTracker]:
    tracker = RunTracker(config, store, task, as_of)
    try:
        yield tracker
    except Exception as exc:
        tracker.finish("failed", {"error": str(exc)})
        raise
    else:
        tracker.finish("success")
