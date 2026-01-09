"""
author: Karol Siegieda
date: 2025-02-24
version: 1.0
description: Source files reader and merger
"""

import os
import concurrent.futures

from pathlib import Path
from typing import Dict, Optional, Set, Tuple
from src.rag.common import setup_logger
from src.rag.config import REPOSITORY_CONFIG
from gitingest import ingest


class RepoReader:
    def __init__(
        self,
        repo_path: str,
        output_dir: str,
        debug: bool = False,
        max_workers: int = REPOSITORY_CONFIG["MAX_WORKERS"],
        log_file: Optional[Path] = None
    ):
        """
        Initialize the repository reader.

        Args:
            repo_path: Path to the repository root
            output_dir: Directory where digest files will be saved
            debug: Enable debug logging
            max_workers: Maximum number of worker processes
            log_file: Optional path to log file
        """
        self.repo_path = Path(repo_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        self.logger = setup_logger("RepoReader", debug=debug, log_file=log_file)
        self.target_dirs = REPOSITORY_CONFIG["TARGET_DIRS"]
        self.exclude_patterns: Set[str] = REPOSITORY_CONFIG["EXCLUDE_PATTERNS"]

    def _process_directory(self, target_dir: str, output_name: str) -> Tuple[str, Optional[str]]:
        """
        Process a single directory and generate its digest file.
        """

        dir_path = self.repo_path / target_dir
        if not dir_path.exists():
            self.logger.warning(f"Directory not found: {target_dir}")
            return target_dir, None

        self.logger.info(f"Processing directory: {target_dir}")

        try:
            # Define include pattern to focus only on this directory
            include_pattern = f"{target_dir}/*"

            # Output path for this digest
            output_path = self.output_dir / output_name

            # Let gitingest handle the output directly
            summary, tree, content = ingest(
                source=str(dir_path),
                # include_patterns={include_pattern}, -> Commented, each digest specified path is a source now
                exclude_patterns=self.exclude_patterns,
                output=str(output_path),
            )

            # Check if the file was created
            if output_path.exists() and output_path.stat().st_size > 0:
                self.logger.info(f"Created digest file: {output_path}")
                return target_dir, str(output_path)
            else:
                self.logger.warning(f"No content generated for {target_dir}")
                return target_dir, None

        except Exception as e:
            self.logger.error(f"Error processing directory {target_dir}: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return target_dir, None

    def generate_digests(self) -> Dict[str, str]:
        """
        Generate digest files for each target directory in parallel.
        """

        digest_paths = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_dir = {
                executor.submit(self._process_directory, target_dir, output_name): target_dir
                for target_dir, output_name in self.target_dirs.items()
            }

            # Process results when complete
            for future in concurrent.futures.as_completed(future_to_dir):
                target_dir, output_path = future.result()
                if output_path:
                    digest_paths[target_dir] = output_path

        return digest_paths
