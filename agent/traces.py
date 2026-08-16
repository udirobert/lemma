"""Append-only JSONL trace logger.

Every LLM call, tool result, and decision is recorded here. The trace file is
the inspectable reasoning artifact for the demo and is attachable to Trackio
logbooks via ``trackio logbook attach trace``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

TRACES_DIR = Path(__file__).resolve().parent / "traces"


class Trace:
    def __init__(self, run_id: str, path: Path | None = None) -> None:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.path = path or (TRACES_DIR / f"{run_id}.jsonl")
        self._t0 = time.time()

    def log(self, stage: str, event: str, **payload: object) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": self.run_id,
            "stage": stage,
            "event": event,
            "elapsed_s": round(time.time() - self._t0, 2),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def llm_call(
        self,
        stage: str,
        model: str,
        prompt_chars: int,
        response_chars: int,
        duration_s: float,
    ) -> None:
        self.log(
            stage,
            "llm_call",
            model=model,
            prompt_chars=prompt_chars,
            response_chars=response_chars,
            duration_s=round(duration_s, 2),
        )

    def tool_run(
        self,
        stage: str,
        cmd: str,
        exit_code: int,
        duration_s: float,
        output_chars: int,
    ) -> None:
        self.log(
            stage,
            "tool_run",
            cmd=cmd,
            exit_code=exit_code,
            duration_s=round(duration_s, 2),
            output_chars=output_chars,
        )

    def note(self, stage: str, message: str) -> None:
        self.log(stage, "note", message=message)

    def read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def summary(self) -> dict:
        records = self.read()
        stages = sorted({r.get("stage", "?") for r in records})
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "events": len(records),
            "llm_calls": sum(1 for r in records if r.get("event") == "llm_call"),
            "tool_runs": sum(1 for r in records if r.get("event") == "tool_run"),
            "stages": stages,
            "wall_s": round(time.time() - self._t0, 2),
        }
