"""
author: Karol Siegieda
date: 2025-02-24
description: Script that produces digest files from repository
example: core.digest, coder.digest

Usage:
    python digest.py -d "~/wrk/git/rg" -o "/tmp/"
"""

import argparse
import os
from pathlib import Path

from src.rag.readers.repo import RepoReader
from src.rag.common import setup_logger

# Set up the main logger for the script
logger = setup_logger("Digest", debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed documents into Qdrant.")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="~/wrk/git/rg",
        help="Directory with repository (default: ~/wrk/git/rg)",
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging.",
    )
    args = parser.parse_args()
    directory = os.path.expanduser(args.directory)

    reader = RepoReader(repo_path=directory, output_dir=args.output, debug=args.debug)
    digest_files = reader.generate_digests()

    logger.info("\nGenerated digest files:")
    for dir_name, file_path in digest_files.items():
        logger.info(f"{dir_name}: {file_path}")
