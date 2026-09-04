#!/usr/bin/env python3
"""
copy_dataset_sample.py

Copy a sample of files from a hierarchical dataset into a mirrored directory structure.
For each directory that directly contains files, the script copies the first N files
(sorted alphabetically) into the corresponding destination directory.

Usage:
    python copy_dataset_sample.py --source /path/to/dataset --dest /path/to/sample --num-files 3
"""

import os
import shutil
import argparse
import logging
import time
from pathlib import Path

def setup_logging(log_file=None):
    """
    Configure logging to console and optionally to a file.
    """
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w', encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

def copy_sample(source_dir: str, dest_dir: str, num_files: int):
    """
    Walk through source_dir, find every directory containing files,
    and copy the first `num_files` files (alphabetically) to dest_dir.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        logging.error(f"Source directory does not exist: {source_path}")
        return

    if not source_path.is_dir():
        logging.error(f"Source path is not a directory: {source_path}")
        return

    # Create destination root if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    total_dirs_processed = 0
    total_files_copied = 0
    total_dirs_with_fewer = 0
    start_time = time.time()

    # Walk the directory tree
    for root, dirs, files in os.walk(source_path):
        # Skip directories with no files (we only care about leaves with files)
        if not files:
            continue

        root_path = Path(root)
        # Compute relative path from source root to current directory
        rel_path = root_path.relative_to(source_path)
        # Destination directory: dest_dir / same relative path
        dest_subdir = dest_path / rel_path
        dest_subdir.mkdir(parents=True, exist_ok=True)

        # Sort files alphabetically to get deterministic "first N"
        sorted_files = sorted(files)
        selected_files = sorted_files[:num_files]
        if len(sorted_files) < num_files:
            total_dirs_with_fewer += 1
            logging.warning(
                f"Directory {rel_path} contains only {len(sorted_files)} files, "
                f"which is less than requested {num_files}. Copying all available."
            )

        # Copy each selected file
        for filename in selected_files:
            src_file = root_path / filename
            dst_file = dest_subdir / filename
            try:
                shutil.copy2(src_file, dst_file)
                total_files_copied += 1
                logging.info(f"Copied: {src_file} -> {dst_file}")
            except Exception as e:
                logging.error(f"Failed to copy {src_file}: {e}")

        total_dirs_processed += 1

    end_time = time.time()
    duration = end_time - start_time

    # Summary
    logging.info("=" * 60)
    logging.info("Copy operation completed.")
    logging.info(f"Total directories processed: {total_dirs_processed}")
    logging.info(f"Total files copied: {total_files_copied}")
    logging.info(f"Directories with fewer than {num_files} files: {total_dirs_with_fewer}")
    logging.info(f"Total time: {duration:.2f} seconds")
    logging.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Copy a sample of files from a dataset directory to a destination, "
                    "preserving the folder structure and limiting to N files per directory."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./midv-dm/MIDV-DM",
        help="Path to the source dataset root (default: ./midv-dm/MIDV-DM)"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="./sample-midv",
        help="Path to the destination sample folder (default: ./sample-midv)"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=3,
        help="Number of files to copy from each directory (default: 3)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path (if not provided, logs only to console)"
    )

    args = parser.parse_args()

    setup_logging(args.log_file)
    logging.info("Starting dataset sample copy...")
    logging.info(f"Source: {args.source}")
    logging.info(f"Destination: {args.dest}")
    logging.info(f"Files per directory: {args.num_files}")

    copy_sample(args.source, args.dest, args.num_files)

if __name__ == "__main__":
    main()
    