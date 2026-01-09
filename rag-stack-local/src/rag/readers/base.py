from typing import Dict, List, Optional
from pathlib import Path
from llama_index.core.readers.base import BaseReader as LlamaBaseReader
from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader
from src.rag.common import setup_logger
from src.rag.config import CHUNK_CONFIG

CHUNK_SIZE = CHUNK_CONFIG["SIZE"]
CHUNK_OVERLAP = CHUNK_CONFIG["OVERLAP"]


class BaseReader:
    """Base class for all document readers in the RAG system."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        normalize: bool = True,
        debug: bool = False,
    ):
        """Initialize the base reader."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.normalize = normalize
        self.debug = debug
        self.logger = setup_logger(self.__class__.__name__, debug=debug)

    def load_file(self, file: Path, extra_info: Optional[Dict] = None) -> List[Document]:
        """Load and process a single file."""
        raise NotImplementedError("Subclasses must implement the load_file method.")

    def load_data(
        self, directory_path: str, file_extension: str, extra_info: Optional[Dict] = None
    ) -> Dict[str, List[Document]]:
        """Process all files in a directory (to be invoked by subclasses)."""
        raise NotImplementedError("Subclasses must implement the load_data method.")
