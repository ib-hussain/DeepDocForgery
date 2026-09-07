from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from deepdocforgery.__main__ import COMMANDS
from deepdocforgery.telemetry import (
    RunLogger,
    logical_cpu_count,
    nvidia_smi_snapshot,
    physical_cpu_count,
    resolve_cpu_threads,
    resolve_worker_count,
    resource_snapshot,
    write_json_atomic,
)


def test_auto_cpu_settings_respect_available_cpus() -> None:
    logical = logical_cpu_count()
    assert logical >= 1
    assert 1 <= physical_cpu_count() <= logical
    assert resolve_cpu_threads("auto") == logical
    workers = resolve_worker_count("auto")
    if logical > 1:
        assert 1 <= workers <= logical - 1
    else:
        assert workers == 0
    with pytest.raises(ValueError, match="positive integer"):
        resolve_cpu_threads(0)
    with pytest.raises(ValueError, match="non-negative integer"):
        resolve_worker_count(-1)


def test_resource_snapshot_reports_normal_ram() -> None:
    snapshot = resource_snapshot(torch.device("cpu"))
    assert snapshot["ram"]["system_total_gib"] > 0
    assert snapshot["ram"]["process_rss_gib"] > 0
    assert snapshot["cpu"]["logical"] >= 1
    assert "cuda" not in snapshot


def test_run_logger_writes_text_jsonl_and_latest_status(tmp_path: Path) -> None:
    with RunLogger("unit", log_root=tmp_path, progress_enabled=False) as run:
        run.info("working", event="unit_progress", value=3)
        text_path = run.text_path
        events_path = run.events_path
        latest_path = run.latest_path
    assert "working" in text_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert {event["event"] for event in events} >= {
        "run_started",
        "unit_progress",
        "run_finished",
    }
    assert json.loads(latest_path.read_text(encoding="utf-8"))["status"] == "succeeded"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["duration_seconds"] >= 0.0
    assert latest["resources"]["ram"]["system_total_gib"] > 0


def test_run_logger_persists_failure_status(tmp_path: Path) -> None:
    holder: dict[str, Path] = {}
    with pytest.raises(RuntimeError, match="deliberate"):
        with RunLogger("failure", log_root=tmp_path, progress_enabled=False) as run:
            holder["latest"] = run.latest_path
            raise RuntimeError("deliberate")
    latest = json.loads(holder["latest"].read_text(encoding="utf-8"))
    assert latest["status"] == "failed"
    assert latest["error_type"] == "RuntimeError"


def test_atomic_json_writers_do_not_share_temporary_files(tmp_path: Path) -> None:
    destination = tmp_path / "latest.json"
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(lambda index: write_json_atomic(destination, {"index": index}), range(32))
        )
    assert json.loads(destination.read_text(encoding="utf-8"))["index"] in range(32)
    assert not list(tmp_path.glob("*.tmp"))


def test_testing_and_status_are_first_class_commands() -> None:
    assert COMMANDS["test"] == "deepdocforgery.evaluate"
    assert COMMANDS["status"] == "deepdocforgery.status"


def test_nvidia_smi_snapshot_reports_load_and_competing_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            subprocess.CompletedProcess(
                [],
                0,
                stdout="0, NVIDIA RTX 4000 Ada, 7011, 13464, 20475, 100, 81, 129, 130\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                [],
                0,
                stdout="7607, hashcat, 6386\n",
                stderr="",
            ),
        )
    )
    monkeypatch.setattr("deepdocforgery.telemetry.shutil.which", lambda _: "/bin/nvidia-smi")
    monkeypatch.setattr(
        "deepdocforgery.telemetry.subprocess.run", lambda *args, **kwargs: next(responses)
    )
    snapshot = nvidia_smi_snapshot()
    assert snapshot is not None
    assert snapshot["utilization_percent"] == 100.0
    assert snapshot["temperature_c"] == 81.0
    assert snapshot["memory_free_gib"] == pytest.approx(13.148, abs=0.001)
    assert snapshot["compute_processes"] == [
        {"pid": 7607, "name": "hashcat", "used_vram_gib": pytest.approx(6.236, abs=0.001)}
    ]
