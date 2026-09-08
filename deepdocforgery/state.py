"""Restart-safe state, fingerprints, locks, and append-only JSONL journals.

Long-running stages use one contract: state is reusable only when the inputs
and behaviour-affecting options have the same fingerprint.  Writes are atomic,
and a stale process lock is recoverable after a crash or power loss.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import socket
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from deepdocforgery.telemetry import utc_now, write_json_atomic

STATE_FORMAT_VERSION = 1


class ResumeStateError(RuntimeError):
    """Raised when saved work cannot safely be reused."""


def command_text(parts: Iterable[object]) -> str:
    """Render one copy-safe shell command with platform-safe quoting."""

    return shlex.join(str(part) for part in parts)


def fingerprint(value: Any) -> str:
    """Return a stable SHA-256 for JSON-compatible contract data."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path, *, content: bool = False) -> dict[str, Any]:
    """Describe a file for a resume contract.

    Large model and LMDB files use size plus nanosecond modification time by
    default; small manifests/configuration files may request a content digest.
    """

    resolved = Path(path).resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content:
        result["sha256"] = sha256_file(resolved)
    return result


def load_json_object(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON object, returning ``None`` when the file is absent."""

    source = Path(path)
    if not source.is_file():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ResumeStateError(f"Resume state is not valid JSON: {source}") from error
    if not isinstance(value, dict):
        raise ResumeStateError(f"Resume state must be a JSON object: {source}")
    return value


def validate_state(
    state: Mapping[str, Any],
    *,
    stage: str,
    contract: str,
    path: str | Path,
) -> None:
    """Reject state from another stage, schema, or input contract."""

    source = Path(path)
    if int(state.get("format_version", -1)) != STATE_FORMAT_VERSION:
        raise ResumeStateError(
            f"Unsupported resume-state format in {source}; use --fresh to start a new state"
        )
    if state.get("stage") != stage:
        raise ResumeStateError(f"Resume state {source} belongs to {state.get('stage')!r}")
    if state.get("contract") != contract:
        raise ResumeStateError(
            f"Saved {stage} state does not match the current inputs/configuration: {source}. "
            "Use --fresh only when a new run is intended."
        )


def save_stage_state(
    path: str | Path,
    *,
    stage: str,
    contract: str,
    status: str,
    **fields: Any,
) -> dict[str, Any]:
    """Atomically persist a stage checkpoint."""

    payload = {
        "format_version": STATE_FORMAT_VERSION,
        "stage": stage,
        "contract": contract,
        "status": status,
        "updated_at": utc_now(),
        **fields,
    }
    write_json_atomic(path, payload)
    return payload


def _process_alive(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class StageLock:
    """Exclusive lock with automatic recovery of stale same-host owners."""

    def __init__(self, path: str | Path, *, stage: str) -> None:
        self.path = Path(path)
        self.stage = stage
        self.token = uuid.uuid4().hex
        self.acquired = False

    def __enter__(self) -> StageLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            payload = {
                "stage": self.stage,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "token": self.token,
                "created_at": utc_now(),
            }
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                try:
                    owner = load_json_object(self.path) or {}
                except ResumeStateError:
                    # A process can die after creating the lock inode but
                    # before completing its JSON owner record. Do not remove a
                    # brand-new lock that another process may still be writing.
                    age = time.time() - self.path.stat().st_mtime
                    if age < 30.0:
                        raise ResumeStateError(
                            f"{self.stage} lock is being initialised or is unreadable: "
                            f"{self.path}; retry after 30 seconds"
                        ) from None
                    self.path.unlink(missing_ok=True)
                    continue
                same_host = owner.get("host") == socket.gethostname()
                owner_pid = int(owner.get("pid", -1))
                if same_host and _process_alive(owner_pid):
                    raise ResumeStateError(
                        f"Another {self.stage} process owns {self.path} (pid {owner_pid})"
                    ) from None
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.acquired = True
            return self
        raise ResumeStateError(f"Could not acquire {self.stage} lock: {self.path}")

    def __exit__(self, *_: object) -> bool:
        if self.acquired:
            try:
                owner = load_json_object(self.path)
            except ResumeStateError:
                owner = None
            if owner is None or owner.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        self.acquired = False
        return False


def _recover_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load complete JSONL rows and discard only a torn final write."""

    if not path.is_file():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    valid_bytes = 0
    for index, encoded in enumerate(lines):
        stripped = encoded.strip()
        if not stripped:
            valid_bytes += len(encoded)
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as error:
            is_torn_tail = index == len(lines) - 1 and not raw.endswith(b"\n")
            if not is_torn_tail:
                raise ResumeStateError(f"Corrupt JSONL resume journal: {path}") from error
            break
        if not isinstance(value, dict):
            raise ResumeStateError(f"JSONL resume rows must be objects: {path}")
        records.append(value)
        valid_bytes += len(encoded)
    if valid_bytes != len(raw):
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    elif raw and not raw.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return records


class JsonlJournal:
    """Append-only records plus atomic progress metadata for restartable stages."""

    def __init__(
        self,
        *,
        records_path: str | Path,
        state_path: str | Path,
        stage: str,
        contract: str,
        total_items: int,
        resume: bool = True,
    ) -> None:
        self.records_path = Path(records_path)
        self.state_path = Path(state_path)
        self.stage = stage
        self.contract = contract
        self.total_items = int(total_items)
        if self.total_items < 0:
            raise ValueError("total_items must be non-negative")
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        saved = load_json_object(self.state_path)
        if saved is not None and resume:
            validate_state(saved, stage=stage, contract=contract, path=self.state_path)
            self.records = _recover_jsonl(self.records_path)
        elif resume and self.records_path.exists():
            raise ResumeStateError(
                f"Resume journal exists without compatible state: {self.records_path}. "
                "Use --fresh only when discarding that journal is intended."
            )
        else:
            self.records_path.unlink(missing_ok=True)
            self.state_path.unlink(missing_ok=True)
            self.records = []
        if len(self.records) > self.total_items:
            raise ResumeStateError(
                f"Resume journal has {len(self.records)} rows for only {self.total_items} items"
            )
        self._handle = self.records_path.open("a", encoding="utf-8")
        self.checkpoint("completed" if len(self.records) == self.total_items else "running")

    @property
    def completed_items(self) -> int:
        return len(self.records)

    @property
    def complete(self) -> bool:
        return self.completed_items == self.total_items

    def append(self, value: Mapping[str, Any]) -> None:
        if self.complete:
            raise ResumeStateError(f"Cannot append beyond {self.total_items} items")
        row = dict(value)
        self._handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        self.records.append(row)

    def checkpoint(self, status: str = "running", **fields: Any) -> dict[str, Any]:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return save_stage_state(
            self.state_path,
            stage=self.stage,
            contract=self.contract,
            status=status,
            records_path=str(self.records_path.resolve()),
            completed_items=self.completed_items,
            total_items=self.total_items,
            **fields,
        )

    def finish(self, **fields: Any) -> dict[str, Any]:
        if not self.complete:
            raise ResumeStateError(
                f"Cannot complete {self.stage}: {self.completed_items}/{self.total_items} items"
            )
        return self.checkpoint("completed", **fields)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> JsonlJournal:
        return self

    def __exit__(self, error_type: object, error: BaseException | None, _: object) -> bool:
        try:
            if error is None and self.complete:
                self.finish()
            else:
                status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                self.checkpoint(status, error=None if error is None else str(error))
        finally:
            self.close()
        return False


def remove_stage_state(*paths: str | Path) -> None:
    """Remove only explicit state/journal files for a requested fresh run."""

    for path in paths:
        Path(path).unlink(missing_ok=True)


def elapsed_seconds(started: float) -> float:
    """Small injectable helper used in state reports."""

    return max(0.0, time.perf_counter() - started)
