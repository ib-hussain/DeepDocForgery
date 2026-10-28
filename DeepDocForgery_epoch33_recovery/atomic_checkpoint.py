"""Atomic, durability-oriented PyTorch checkpoint file writer.
Integrate in existing training code only after preserving its native checkpoint structure.
"""
import os
import tempfile
from pathlib import Path
import torch


def _sync_dir(directory):
    fd=os.open(str(directory),os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_checkpoint_atomic(checkpoint, destination, keep_previous=True):
    """Save checkpoint atomically on Linux filesystems supporting os.replace.

    Never replaces an existing checkpoint until writing + fsync succeeds.
    On success, optionally preserves the previous version as *.previous.pt.
    This cannot protect against a hardware/filesystem failure or semantic bugs.
    """
    dest=Path(destination)
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=None
    backup_tmp=None
    try:
        with tempfile.NamedTemporaryFile(prefix=f'.{dest.name}.',suffix='.tmp',dir=dest.parent,delete=False) as f:
            tmp=Path(f.name)
            torch.save(checkpoint,f)
            f.flush()
            os.fsync(f.fileno())
        if keep_previous and dest.exists():
            previous=dest.with_name(dest.stem+'.previous'+dest.suffix)
            # Hardlink old inode; os.replace below never alters its bytes.
            fd,name=tempfile.mkstemp(prefix=f'.{previous.name}.',suffix='.bak',dir=dest.parent)
            os.close(fd)
            os.unlink(name)
            backup_tmp=Path(name)
            os.link(dest,backup_tmp)
            os.replace(backup_tmp,previous)
            backup_tmp=None
        os.replace(tmp,dest)
        tmp=None
        _sync_dir(dest.parent)
        return dest
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        if backup_tmp is not None:
            backup_tmp.unlink(missing_ok=True)
