#!/usr/bin/env python3
"""
Extract all image-<idx>/label-<idx> entries from the DocTamper LMDB stores
into per-dataset output folders, writing raw bytes directly (no re-encoding)
using a thread pool for speed.

Usage:
    python3 extract_doctamper_lmdb.py
    python3 extract_doctamper_lmdb.py --data-root doctamper-data --output-root extracted --workers 16
"""

import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lmdb

DATASETS = [
    "DocTamperV1-SCD",
    "DocTamperV1-FCD",
    "DocTamperV1-TrainingSet",
    "DocTamperV1-TestingSet",
]

# Minimal magic-byte sniffing (avoids the deprecated stdlib imghdr module).
_SIGNATURES = [
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tif"),
    (b"MM\x00*", "tif"),
]


def guess_ext(buf: bytes) -> str:
    for sig, ext in _SIGNATURES:
        if buf.startswith(sig):
            return ext
    return "bin"


def get_all_keys(env: lmdb.Environment) -> list[str]:
    # keys=True, values=False walks the B+tree index only — it never pages in
    # the (large) image/label bytes. Iterating "for key, _ in cursor" instead
    # silently reads every value too, which is what made this look hung.
    with env.begin(write=False) as txn:
        with txn.cursor() as cursor:
            return [key.decode("utf-8") for key in cursor.iternext(keys=True, values=False)]


def extract_key(env: lmdb.Environment, key: str, out_dir: Path, skip_existing: bool) -> None:
    if "-" not in key:
        return
    prefix, idx = key.split("-", 1)
    sub_dir = out_dir / ("images" if prefix == "image" else "labels")

    if skip_existing and any(sub_dir.glob(f"{idx}.*")):
        return

    with env.begin(write=False) as txn:
        buf = txn.get(key.encode("utf-8"))
    if buf is None:
        return

    ext = guess_ext(buf)
    with open(sub_dir / f"{idx}.{ext}", "wb") as f:
        f.write(buf)


def extract_dataset(data_root: Path, output_root: Path, dataset_name: str, num_workers: int,
                     skip_existing: bool) -> None:
    lmdb_path = data_root / dataset_name
    out_dir = output_root / dataset_name
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=num_workers + 4,
    )

    print(f"[{dataset_name}] listing keys...")
    keys = get_all_keys(env)
    total = len(keys)
    print(f"[{dataset_name}] {total} entries found — extracting with {num_workers} threads")

    done = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(extract_key, env, key, out_dir, skip_existing) for key in keys]
        for future in as_completed(futures):
            future.result()  # re-raise any exception immediately
            done += 1
            if done % 5000 == 0 or done == total:
                print(f"[{dataset_name}] {done}/{total}")

    env.close()
    print(f"[{dataset_name}] done -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DocTamper LMDB entries to disk.")
    parser.add_argument("--data-root", default=".", help="Folder containing the DocTamperV1-* LMDB dirs")
    parser.add_argument("--output-root", default="extracted", help="Where to write extracted files")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count() * 2,
                         help="Thread count (I/O-bound, so 2x cpu_count is a good default)")
    parser.add_argument("--datasets", nargs="*", default=DATASETS,
                         help="Subset of dataset folder names to process")
    parser.add_argument("--skip-existing", action="store_true",
                         help="Skip entries that already have an output file (safe to re-run after Ctrl+C)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        if not (data_root / dataset_name / "data.mdb").exists():
            print(f"[{dataset_name}] skipped — no data.mdb found under {data_root / dataset_name}")
            continue
        extract_dataset(data_root, output_root, dataset_name, args.workers, args.skip_existing)


if __name__ == "__main__":
    main()

