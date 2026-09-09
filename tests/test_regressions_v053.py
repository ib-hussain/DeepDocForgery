from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from deepdocforgery.doctor import _doctor
from deepdocforgery.telemetry import RunLogger


def test_doctor_counter_progress_does_not_require_iterator_protocol(tmp_path: Path) -> None:
    """Regression: tqdm.auto progress bars support update(), not next(progress)."""

    config = tmp_path / "doctor.yaml"
    config.write_text(
        "training:\n  cpu_threads: 1\ndata:\n  num_workers: 0\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        profile="cpu",
        config=config,
        output=tmp_path / "doctor-report.json",
        state=tmp_path / "doctor-state.json",
        log_dir=tmp_path / "logs",
        no_progress=False,
    )
    # No manifest is intentional.  It lets the doctor exercise all four top-level
    # progress increments without doing a large file audit.  The result should be
    # attention_required, but the progress implementation must never crash.
    with RunLogger("doctor-regression", log_root=tmp_path / "logs", progress_enabled=False) as run:
        result = _doctor(args, run)

    assert result["status"] == "attention_required"
    assert result["checks"]["stages"]["runtime"] == "passed"
    assert result["checks"]["stages"]["manifest"] == "attention_required"
    assert result["checks"]["stages"]["protocol"] == "blocked"
    assert (tmp_path / "doctor-report.json").is_file()
    assert (tmp_path / "doctor-state.json").is_file()


def test_pytest_config_blocks_ros_jazzy_entrypoint_autoload(tmp_path: Path) -> None:
    """A sourced ROS install must not inject its pytest11 hook into this suite."""

    plugin_root = tmp_path / "fake-ros-site"
    plugin_root.mkdir()
    (plugin_root / "launch_testing_ros_pytest_entrypoint.py").write_text(
        "def pytest_launch_collect_makemodule(parent, path):\n    return None\n",
        encoding="utf-8",
    )
    dist_info = plugin_root / "fake_ros_launch_testing-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: fake-ros-launch-testing\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[pytest11]\n"
        "launch_testing_ros_pytest_entrypoint = launch_testing_ros_pytest_entrypoint\n",
        encoding="utf-8",
    )
    child_test = tmp_path / "child_test.py"
    child_test.write_text("def test_child():\n    assert True\n", encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    # This specifically verifies repository configuration.  The setup scripts also
    # disable plugin autoload, but that exported variable does not persist after a
    # user later activates the venv in another shell.
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    env["PYTHONPATH"] = os.pathsep.join((str(plugin_root), str(project_root)))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(child_test),
            "-c",
            str(project_root / "pyproject.toml"),
        ],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "1 passed" in completed.stdout
