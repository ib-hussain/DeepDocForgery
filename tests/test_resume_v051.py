from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import torch

from deepdocforgery.metrics import StreamingForgeryMetrics
from deepdocforgery.state import (
    JsonlJournal,
    ResumeStateError,
    StageLock,
    command_text,
    fingerprint,
)
from deepdocforgery.train import _training_contract


def test_fingerprint_is_order_independent_and_contract_sensitive() -> None:
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_resume_command_quotes_paths_as_one_shell_argument() -> None:
    command = command_text(("python", "--output", Path("output/model/my run")))
    assert command == "python --output 'output/model/my run'"


def test_training_contract_includes_manifest_content_and_debug_budget(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"sample_id":"one"}\n', encoding="utf-8")
    config = {
        "data": {"manifest": str(manifest)},
        "training": {"epochs": 2, "output_dir": "output/model/first"},
        "logging": {"resource_interval_batches": 10},
    }
    baseline = _training_contract(config)
    # Output/log paths do not alter model semantics, but bounded debug runs do.
    moved = {
        **config,
        "training": {**config["training"], "output_dir": "output/model/second"},
        "logging": {"resource_interval_batches": 1},
    }
    assert _training_contract(moved) == baseline
    assert _training_contract(config, maximum_train_batches=1) != baseline
    assert _training_contract(config, device="cuda") != _training_contract(config, device="cpu")
    manifest.write_text('{"sample_id":"two"}\n', encoding="utf-8")
    assert _training_contract(config) != baseline


def test_jsonl_journal_resumes_and_repairs_a_torn_tail(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    state_path = tmp_path / "state.json"
    contract = fingerprint({"dataset": "fixture"})
    first = JsonlJournal(
        records_path=records_path,
        state_path=state_path,
        stage="prepare/fixture",
        contract=contract,
        total_items=3,
        resume=False,
    )
    first.append({"sample_id": "one"})
    first.checkpoint("running")
    first.close()
    with records_path.open("ab") as handle:
        handle.write(b'{"sample_id":')

    resumed = JsonlJournal(
        records_path=records_path,
        state_path=state_path,
        stage="prepare/fixture",
        contract=contract,
        total_items=3,
        resume=True,
    )
    assert resumed.records == [{"sample_id": "one"}]
    resumed.append({"sample_id": "two"})
    resumed.append({"sample_id": "three"})
    resumed.finish()
    resumed.close()
    assert [json.loads(line) for line in records_path.read_text().splitlines()] == [
        {"sample_id": "one"},
        {"sample_id": "two"},
        {"sample_id": "three"},
    ]
    assert json.loads(state_path.read_text())["status"] == "completed"


def test_jsonl_journal_rejects_changed_inputs(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    state = tmp_path / "state.json"
    journal = JsonlJournal(
        records_path=records,
        state_path=state,
        stage="infer",
        contract=fingerprint({"inputs": ["a.jpg"]}),
        total_items=1,
        resume=False,
    )
    journal.close()
    with pytest.raises(ResumeStateError, match="does not match"):
        JsonlJournal(
            records_path=records,
            state_path=state,
            stage="infer",
            contract=fingerprint({"inputs": ["b.jpg"]}),
            total_items=1,
            resume=True,
        )


def test_stage_lock_recovers_a_dead_same_host_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "prepare.lock"
    lock_path.write_text(
        json.dumps({"host": socket.gethostname(), "pid": 2**30, "token": "stale"}),
        encoding="utf-8",
    )
    with StageLock(lock_path, stage="prepare") as lock:
        assert lock.acquired
        assert lock_path.is_file()
    assert not lock_path.exists()


def test_streaming_metrics_round_trip_without_changing_results() -> None:
    first = StreamingForgeryMetrics(minimum_instance_area=1)
    target = torch.zeros(2, 1, 8, 8)
    target[0, :, 1:3, 1:3] = 1.0
    target[1, :, 3:6, 3:6] = 1.0
    prediction = target.clone()
    first.update(
        mask_probability=prediction,
        tamper_mask=target,
        image_probability=torch.tensor([[0.8], [0.7]]),
        image_label=torch.ones(2, 1),
        valid_mask=torch.ones_like(target),
        image_valid=torch.ones(2, 1),
    )
    resumed = StreamingForgeryMetrics(minimum_instance_area=1)
    resumed.load_state_dict(first.state_dict())
    assert resumed.compute() == first.compute()
