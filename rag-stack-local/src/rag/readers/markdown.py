from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime
from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser

from src.rag.common import normalize_text
from src.rag.config import CHUNK_CONFIG
from src.rag.readers import BaseReader

CHUNK_SIZE = CHUNK_CONFIG["SIZE"]
CHUNK_OVERLAP = CHUNK_CONFIG["OVERLAP"]


class MarkdownReader(BaseReader):
    """Markdown document reader."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        normalize: bool = True,
        debug: bool = False,
    ):
        super().__init__(chunk_size, chunk_overlap, normalize, debug)
        self.markdown_parser = MarkdownNodeParser.from_defaults()
        self.sentence_splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def _relative_path(self, file: Path) -> str:
        """Get the relative path of a file."""
        try:
            return str(file.relative_to(Path.cwd()))
        except ValueError:
            return str(file.name)

    def get_file_metadata(self, file: Path) -> Dict[str, str]:
        """Extract additional metadata from the file."""
        file_stat = file.stat()
        return {
            "file_name": file.name,
            "file_path": self._relative_path(file),
            "file_size": file_stat.st_size,
            "creation_date": datetime.fromtimestamp(file_stat.st_ctime).strftime("%Y-%m-%d"),
            "last_modified_date": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d"),
            "file_type": "text/markdown",
        }

    def load_file(self, file: Path, extra_info: Optional[Dict] = None) -> List[Document]:
        """Load and process a single markdown file."""
        try:
            if not file.exists():
                self.logger.error(f"File not found: {file}")
                return []

            with file.open("r", encoding="utf-8") as f:
                text = f.read()

            metadata = self._get_file_metadata(file)
            metadata.update(extra_info or {})

            return self.process_text(text, metadata)

        except Exception as e:
            self.logger.error(f"Error loading file {file}: {e}")
            return []

    def load_data(
        self, directory_path: str, file_extension: str = ".md", extra_info: Optional[Dict] = None
    ) -> Dict[str, List[Document]]:
        """Load and process all markdown files in a directory."""
        reader = SimpleDirectoryReader(input_dir=directory_path, required_exts=[file_extension], recursive=True)
        raw_docs = reader.load_data()

        results = {}
        for doc in raw_docs:
            file_path = doc.metadata.get("file_path", "unknown")
            relative_path = self._relative_path(Path(file_path))
            content = doc.get_content()

            # Generate metadata for chunks
            metadata = {
                **doc.metadata,
                "file_path": relative_path,
            }
            metadata.update(extra_info or {})

            chunks = self.process_text(content, metadata)
            results.setdefault(relative_path, []).extend(chunks)

        return results

    def process_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """Process markdown text into chunked documents."""
        node = Document(text=text, metadata=metadata or {})
        nodes = self.markdown_parser.get_nodes_from_node(node)
        documents = []

        for node in nodes:
            content = node.get_content()
            if self.debug:
                print(f"Node content: {content[:100]}...")

            if self.normalize:
                content = normalize_text(content)

            sentences = self.sentence_splitter.split_text(content)
            if self.debug:
                print(f"Split into {len(sentences)} sentences.")

            # Merge node-specific metadata with base metadata
            node_metadata = {**(metadata or {}), **node.metadata}

            for sentence in sentences:
                if self.debug:
                    print(f"Chunk created: {sentence[:100]}...")
                documents.append(Document(text=sentence, metadata=node_metadata))

        return documents
