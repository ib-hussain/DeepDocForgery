"""Consistent command logging, progress reporting, and resource telemetry.

Human-readable events go to stderr, final command results remain JSON on
stdout, and every run also receives text and JSONL logs under ``output/logs``.
This separation keeps the CLI pleasant to watch and safe to automate.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import psutil
import torch
from tqdm.auto import tqdm

T = TypeVar("T")
GIB = 1024**3
CPU_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_PROCESS = psutil.Process()


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with explicit timezone information."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically replace a JSON report so readers never see partial output."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def logical_cpu_count() -> int:
    """Return CPUs available to this process, respecting affinity/cgroups."""

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, psutil.cpu_count(logical=True) or os.cpu_count() or 1)


def physical_cpu_count() -> int:
    detected = psutil.cpu_count(logical=False) or logical_cpu_count()
    return max(1, min(int(detected), logical_cpu_count()))


def resolve_cpu_threads(value: int | str | None) -> int:
    available = logical_cpu_count()
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return available
    threads = int(value)
    if threads < 1:
        raise ValueError("cpu_threads must be 'auto' or a positive integer")
    return min(threads, available)


def resolve_worker_count(value: int | str | None) -> int:
    """Choose safe, high-throughput data workers without CPU oversubscription."""

    if value is not None and not (isinstance(value, str) and value.lower() == "auto"):
        workers = int(value)
        if workers < 0:
            raise ValueError("num_workers must be 'auto' or a non-negative integer")
        return workers
    logical = logical_cpu_count()
    if logical == 1:
        return 0
    # PyTorch owns all logical threads during model compute. Data loading uses
    # independent processes with one Torch thread each, capped at physical
    # cores and leaving one logical CPU for the parent process/OS.
    return max(1, min(physical_cpu_count(), logical - 1))


def configure_compute(
    device: torch.device,
    *,
    cpu_threads: int | str | None = "auto",
) -> dict[str, Any]:
    """Apply process-wide CPU settings and return the resolved configuration."""

    threads = resolve_cpu_threads(cpu_threads)
    for name in CPU_THREAD_ENVIRONMENT:
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    interop_threads = max(1, min(4, physical_cpu_count()))
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        # PyTorch permits this setting only before inter-op work begins. A
        # caller may configure more than one command in a unit-test process.
        interop_threads = torch.get_num_interop_threads()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return {
        "logical_cpus": logical_cpu_count(),
        "physical_cpus": physical_cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": interop_threads,
        "thread_environment": {name: os.environ[name] for name in CPU_THREAD_ENVIRONMENT},
    }


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _optional_number(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def nvidia_smi_snapshot(device_index: int = 0) -> dict[str, Any] | None:
    """Read physical GPU load and competing compute processes when available."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    query = (
        "index,name,memory.used,memory.free,memory.total,utilization.gpu,"
        "temperature.gpu,power.draw,power.limit"
    )
    try:
        completed = subprocess.run(
            [
                executable,
                "-i",
                str(device_index),
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    values = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
    if len(values) != 9:
        return None
    used_mib, free_mib, total_mib = (_optional_number(value) for value in values[2:5])
    result: dict[str, Any] = {
        "index": int(_optional_number(values[0]) or device_index),
        "name": values[1],
        "memory_used_gib": None if used_mib is None else round(used_mib / 1024, 3),
        "memory_free_gib": None if free_mib is None else round(free_mib / 1024, 3),
        "memory_total_gib": None if total_mib is None else round(total_mib / 1024, 3),
        "utilization_percent": _optional_number(values[5]),
        "temperature_c": _optional_number(values[6]),
        "power_draw_w": _optional_number(values[7]),
        "power_limit_w": _optional_number(values[8]),
        "compute_processes": [],
    }
    try:
        processes = subprocess.run(
            [
                executable,
                "-i",
                str(device_index),
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if processes.returncode == 0:
            for line in processes.stdout.splitlines():
                fields = [item.strip() for item in line.split(",", maxsplit=2)]
                if len(fields) != 3:
                    continue
                memory_mib = _optional_number(fields[2])
                result["compute_processes"].append(
                    {
                        "pid": int(_optional_number(fields[0]) or -1),
                        "name": fields[1],
                        "used_vram_gib": (
                            None if memory_mib is None else round(memory_mib / 1024, 3)
                        ),
                    }
                )
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def resource_snapshot(device: torch.device | None = None) -> dict[str, Any]:
    """Return process/system RAM and optional CUDA VRAM statistics."""

    memory = psutil.virtual_memory()
    snapshot: dict[str, Any] = {
        "cpu": {
            "logical": logical_cpu_count(),
            "physical": physical_cpu_count(),
            "process_percent": round(_PROCESS.cpu_percent(interval=None), 1),
            "system_percent": round(psutil.cpu_percent(interval=None), 1),
        },
        "ram": {
            "process_rss_gib": round(_PROCESS.memory_info().rss / GIB, 3),
            "system_used_gib": round(memory.used / GIB, 3),
            "system_available_gib": round(memory.available / GIB, 3),
            "system_total_gib": round(memory.total / GIB, 3),
            "system_percent": round(float(memory.percent), 1),
        },
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        cuda_snapshot: dict[str, Any] = {
            "device": index,
            "name": torch.cuda.get_device_name(index),
            "allocated_gib": round(torch.cuda.memory_allocated(index) / GIB, 3),
            "reserved_gib": round(torch.cuda.memory_reserved(index) / GIB, 3),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(index) / GIB, 3),
            "device_used_gib": round((total_bytes - free_bytes) / GIB, 3),
            "device_free_gib": round(free_bytes / GIB, 3),
            "device_total_gib": round(total_bytes / GIB, 3),
        }
        physical = nvidia_smi_snapshot(index)
        if physical is not None:
            cuda_snapshot["nvidia_smi"] = physical
        snapshot["cuda"] = cuda_snapshot
    return snapshot


def compact_resources(snapshot: Mapping[str, Any]) -> str:
    cpu = snapshot.get("cpu", {})
    ram = snapshot.get("ram", {})
    parts = [
        f"CPU {float(cpu.get('system_percent', 0.0)):.0f}%",
        f"RAM {float(ram.get('system_used_gib', 0.0)):.1f}/"
        f"{float(ram.get('system_total_gib', 0.0)):.1f}G",
        f"RSS {float(ram.get('process_rss_gib', 0.0)):.1f}G",
    ]
    cuda = snapshot.get("cuda")
    if isinstance(cuda, Mapping):
        parts.append(
            f"VRAM {float(cuda.get('device_used_gib', 0.0)):.1f}/"
            f"{float(cuda.get('device_total_gib', 0.0)):.1f}G"
        )
        nvidia_smi = cuda.get("nvidia_smi")
        if isinstance(nvidia_smi, Mapping):
            utilisation = nvidia_smi.get("utilization_percent")
            temperature = nvidia_smi.get("temperature_c")
            if utilisation is not None:
                parts.append(f"GPU {float(utilisation):.0f}%")
            if temperature is not None:
                parts.append(f"temp {float(temperature):.0f}C")
        parts.append(
            f"alloc/reserved/peak "
            f"{float(cuda.get('allocated_gib', 0.0)):.1f}/"
            f"{float(cuda.get('reserved_gib', 0.0)):.1f}/"
            f"{float(cuda.get('peak_allocated_gib', 0.0)):.1f}G"
        )
    return " | ".join(parts)


class _TqdmConsoleHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


class RunLogger:
    """Own all human and machine-readable observability for one command."""

    def __init__(
        self,
        command: str,
        *,
        log_root: str | Path = "output/logs",
        progress_enabled: bool = True,
    ) -> None:
        self.command = command
        self.started_at = utc_now()
        self._started_monotonic = time.perf_counter()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.directory = Path(log_root).resolve() / command
        self.directory.mkdir(parents=True, exist_ok=True)
        self.text_path = self.directory / f"{self.run_id}.log"
        self.events_path = self.directory / f"{self.run_id}.jsonl"
        self.latest_path = self.directory / "latest.json"
        self.progress_enabled = bool(progress_enabled)
        self._lock = threading.Lock()
        self._finished = False
        self._last_resources: Mapping[str, Any] | None = None

        self.logger = logging.getLogger(f"deepdocforgery.{command}.{self.run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        console = _TqdmConsoleHandler()
        console.setFormatter(formatter)
        file_handler = logging.FileHandler(self.text_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.handlers[:] = [console, file_handler]

    @property
    def last_resources(self) -> Mapping[str, Any]:
        """Most recent sample, suitable for failure/status reports."""

        return self._last_resources or resource_snapshot()

    def __enter__(self) -> RunLogger:
        self.event(
            "run_started",
            "STARTED",
            cwd=str(Path.cwd()),
            pid=os.getpid(),
            runtime={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            },
            resources=resource_snapshot(),
        )
        self.info(f"Logs: {self.text_path}", event="log_paths", jsonl=str(self.events_path))
        return self

    def __exit__(self, error_type: object, error: BaseException | None, tb: object) -> bool:
        if error is not None:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            self.event(
                "run_failed",
                f"{status.upper()}: {error}",
                level=logging.ERROR,
                error_type=type(error).__name__,
                traceback="".join(traceback.format_exception(type(error), error, tb)),
                resources=self.last_resources,
            )
            self.finish(status, error=str(error), error_type=type(error).__name__)
        elif not self._finished:
            self.finish("succeeded")
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()
        self.logger.handlers.clear()
        return False

    def event(
        self,
        name: str,
        message: str,
        *,
        level: int = logging.INFO,
        **fields: Any,
    ) -> None:
        supplied_resources = fields.get("resources")
        if isinstance(supplied_resources, Mapping):
            self._last_resources = supplied_resources
        record = {
            "timestamp": utc_now(),
            "command": self.command,
            "run_id": self.run_id,
            "level": logging.getLevelName(level).lower(),
            "event": name,
            "message": message,
            **fields,
        }
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
            if not self._finished:
                running = {
                    "command": self.command,
                    "run_id": self.run_id,
                    "status": "running",
                    "started_at": self.started_at,
                    "updated_at": record["timestamp"],
                    "elapsed_seconds": round(time.perf_counter() - self._started_monotonic, 3),
                    "last_event": name,
                    "message": message,
                    "text_log": str(self.text_path),
                    "events_log": str(self.events_path),
                    "resources": self.last_resources,
                }
                write_json_atomic(self.latest_path, running)
        self.logger.log(level, message)

    def info(self, message: str, *, event: str = "info", **fields: Any) -> None:
        self.event(event, message, **fields)

    def warning(self, message: str, *, event: str = "warning", **fields: Any) -> None:
        self.event(event, message, level=logging.WARNING, **fields)

    def resource(self, device: torch.device | None, *, event: str = "resources") -> dict[str, Any]:
        snapshot = resource_snapshot(device)
        self.event(event, compact_resources(snapshot), resources=snapshot)
        return snapshot

    def progress(
        self,
        iterable: Iterable[T],
        *,
        total: int | None,
        description: str,
        unit: str,
        initial: int = 0,
        leave: bool = True,
    ) -> tqdm[T]:
        return tqdm(
            iterable,
            total=total,
            desc=description,
            unit=unit,
            initial=initial,
            dynamic_ncols=True,
            mininterval=0.5,
            leave=leave,
            disable=not self.progress_enabled,
            file=sys.stderr,
        )

    def finish(self, status: str, **summary: Any) -> dict[str, Any]:
        if self._finished:
            return {}
        self._finished = True
        final_resources = summary.get("resources")
        if not isinstance(final_resources, Mapping):
            final_resources = self.last_resources
            summary["resources"] = final_resources
        result = {
            "command": self.command,
            "run_id": self.run_id,
            "status": status,
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "duration_seconds": round(time.perf_counter() - self._started_monotonic, 3),
            "text_log": str(self.text_path),
            "events_log": str(self.events_path),
            **summary,
        }
        write_json_atomic(self.latest_path, result)
        level = logging.INFO if status == "succeeded" else logging.ERROR
        self.event(
            "run_finished",
            f"{status.upper()} | {compact_resources(final_resources)}",
            level=level,
            summary=result,
        )
        return result


def print_result(value: Mapping[str, Any]) -> None:
    """Print the final machine-readable command result to stdout."""

    print(json.dumps(value, indent=2, sort_keys=True, default=_json_default), flush=True)
