#!/usr/bin/env python3
"""Install the audited repair after checking every source hash; preserve YAML/state."""

import argparse
import hashlib
import json
import os
import shutil
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install(repo, check_only=False):
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / "repair-manifest.json").read_text())
    repo = repo.resolve()
    config = repo / "configs/cuda/HPO_RESULTS.yaml"
    if not config.is_file() or not (repo / "deepdocforgery/train.py").is_file():
        raise RuntimeError("--repo must point to your existing DeepDocForgery root")
    lock = repo / "output/model/cuda-full/.train.lock"
    if lock.exists():
        owner = json.loads(lock.read_text())
        if owner.get("host") != socket.gethostname():
            raise RuntimeError("A training lock exists for another host; confirm that job has stopped")
        pid = int(owner.get("pid", -1))
        if pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"Training process {pid} is active; stop it before applying the repair")
    changes = []
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe manifest path: {relative}")
        source, target = bundle / "payload" / relative, repo / relative
        if digest(source) != item["repaired_sha256"]:
            raise RuntimeError(f"Bundle integrity check failed: {relative}")
        current = digest(target)
        if current == item["repaired_sha256"]:
            continue
        if current != item["original_sha256"]:
            raise RuntimeError(f"Local edits detected in {relative}; no files changed. Compare with the included diff before merging")
        changes.append(item)
    config_hash = digest(config)
    if check_only or not changes:
        return {"status": "compatible" if changes else "already_installed", "files_to_change": [v["path"] for v in changes], "config_sha256": config_hash}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = repo / "output/recovery" / f"epoch33-code-{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for item in changes:
        target = repo / item["path"]
        if target.exists():
            saved = backup / item["path"]
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    record = {"config_sha256": config_hash, "files": changes, "backup": str(backup)}
    (backup / "installation.json").write_text(json.dumps(record, indent=2) + "\n")
    installed = []
    try:
        for item in changes:
            atomic_copy(bundle / "payload" / item["path"], repo / item["path"])
            installed.append(item)
        if digest(config) != config_hash:
            raise RuntimeError("The YAML changed during installation; code installation rolled back")
    except BaseException:
        for item in reversed(installed):
            saved, target = backup / item["path"], repo / item["path"]
            if saved.is_file():
                atomic_copy(saved, target)
            else:
                target.unlink(missing_ok=True)
        raise
    return {"status": "installed", "backup": str(backup), "config_unchanged": True,
            "files": [item["path"] for item in changes]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(install(args.repo, args.check), indent=2))
    except Exception as error:
        parser.exit(1, f"Repair not installed: {error}\n")


if __name__ == "__main__":
    main()
