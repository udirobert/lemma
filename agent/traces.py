"""Append-only JSONL trace logger.

Every LLM call, tool result, and decision is recorded here. The trace file is
the inspectable reasoning artifact for the demo and is attachable to Trackio
logbooks via ``trackio logbook attach trace``.

Each record is hash-chained (SHA-256) so edits, deletions and reorderings are
detectable by ``verify_trace``: ``chain_i = sha256(chain_{i-1} + canonical(record_i))``
where the genesis is bound to the run_id. The final chain value is a single
commitment to every record written so far.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

TRACES_DIR = Path(__file__).resolve().parent / "traces"

_CHAIN_VERSION = "lemma-trace-v1"


def _canonical(record: dict) -> str:
    return json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _genesis(run_id: str) -> str:
    return hashlib.sha256(f"{_CHAIN_VERSION}:{run_id}".encode()).hexdigest()


def _chain_next(prev_chain: str, record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "chain"}
    return hashlib.sha256((prev_chain + "\n" + _canonical(body)).encode()).hexdigest()


def verify_trace(path: Path | str) -> dict:
    """Recompute the hash chain over a trace file.

    Returns {records, chained, valid, head, problems}. Legacy files whose
    records lack chain fields verify as ``valid=False`` with per-line problems.
    """
    path = Path(path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    problems: list[str] = []
    prev_chain: str | None = None
    n_chained = 0
    prev_seq = 0
    prev_i = 0
    for i, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {i}: invalid JSON ({e})")
            continue
        chain = rec.get("chain")
        if chain is None:
            problems.append(f"line {i}: unchained record")
            continue
        if prev_chain is None:
            genesis = _genesis(str(rec.get("run_id", "")))
            if rec.get("seq") != i:
                problems.append(
                    f"line {i}: first chained record has seq {rec.get('seq')}"
                )
            expected = _chain_next(genesis, rec)
        else:
            if rec.get("seq") != prev_seq + (i - prev_i):
                problems.append(
                    f"line {i}: seq gap (seq={rec.get('seq')}, "
                    f"expected {prev_seq + (i - prev_i)})"
                )
            expected = _chain_next(prev_chain, rec)
        if chain != expected:
            problems.append(f"line {i}: chain mismatch")
        prev_chain, prev_seq, prev_i = chain, rec.get("seq", i), i
        n_chained += 1
    return {
        "records": len(lines),
        "chained": n_chained,
        "valid": not problems and n_chained == len(lines) > 0,
        "head": prev_chain if n_chained else None,
        "problems": problems,
    }


class Trace:
    def __init__(self, run_id: str, path: Path | None = None) -> None:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.path = path or (TRACES_DIR / f"{run_id}.jsonl")
        self._t0 = time.time()
        self._lock = threading.Lock()
        self._seq = 0
        self._chain = _genesis(run_id)
        # Resume an existing file: continue the chain from the last record.
        if self.path.is_file():
            last = None
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._seq += 1
                    last = line
            if last is not None:
                try:
                    rec = json.loads(last)
                    if rec.get("chain"):
                        self._chain = rec["chain"]
                        self._seq = int(rec.get("seq", self._seq))
                except (json.JSONDecodeError, ValueError):
                    pass

    def log(self, stage: str, event: str, **payload: object) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": self.run_id,
            "stage": stage,
            "event": event,
            "elapsed_s": round(time.time() - self._t0, 2),
            **payload,
        }
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            self._seq += 1
            record["seq"] = self._seq
            record["chain"] = _chain_next(self._chain, record)
            self._chain = record["chain"]
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @property
    def chain_head(self) -> str:
        """Current chain tip: a commitment to all records written so far."""
        return self._chain

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
