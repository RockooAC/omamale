import logging
import spacy
import re

from pathlib import Path
from typing import Dict, List, Optional

from fsspec import AbstractFileSystem
from grobid_client import Client
from grobid_client.api.pdf import process_fulltext_document
from grobid_client.models import Article, ProcessForm
from grobid_client.types import TEI, File
from llama_index.core.readers.file.base import get_default_fs
from llama_index.core.schema import Document

from src.rag.common import normalize_text
from src.rag.config import CHUNK_CONFIG
from src.rag.readers import BaseReader

MIN_SENTENCE_LEN = 10
CHUNK_SIZE = CHUNK_CONFIG["SIZE"]
CHUNK_OVERLAP = CHUNK_CONFIG["OVERLAP"]


def append_docs(documents: List[Document], text: str, metadata: Optional[Dict] = None, normalize: bool = True) -> None:
    """
    Append a new Document to the list of documents if the text meets the minimum sentence length.
    """
    if normalize:
        text = normalize_text(text)
    if len(text) >= MIN_SENTENCE_LEN:
        documents.append(Document(text=text, metadata=metadata))


class GrobidPDFReader(BaseReader):
    """Grobid PDF parser."""

    def __init__(
        self,
        grobid_server: str,
        split_sentence: Optional[bool] = False,
        load_figures: Optional[bool] = False,
        normalize: bool = True,
        debug: bool = False,
    ):
        """
        Initialize GrobidPDFReader.

        Args:
            grobid_server (str): The URL of the Grobid server.
            split_sentence (Optional[bool]): Whether to split sentences into separate documents.
            load_figures (Optional[bool]): Whether to include figures in the parsed output.
            normalize (bool): Whether to normalize text.
            debug (bool): Enable debug logging.
        """
        # Initialize BaseReader
        super().__init__(normalize=normalize, debug=debug)

        # Initialize Grobid-specific attributes
        self.client = Client(base_url=f"{grobid_server}/api", verify_ssl=False, timeout=300)
        self.split_sentence = split_sentence
        self.load_figures = load_figures

    def load_file(
        self, file: Path, extra_info: Optional[Dict] = None, fs: Optional[AbstractFileSystem] = None
    ) -> List[Document]:
        """
        Load and process a single PDF file.
        """
        if self.debug:
            self.logger.info(f"Processing file: {file.name}")

        file = Path(file) if not isinstance(file, Path) else file

        try:
            fs = fs or get_default_fs()
            with fs.open(str(file), "rb") as fin:
                resp = process_fulltext_document.sync_detailed(
                    client=self.client,
                    multipart_data=ProcessForm(
                        generate_ids="0",
                        consolidate_header="0",
                        consolidate_citations="0",
                        include_raw_citations="0",
                        include_raw_affiliations="0",
                        tei_coordinates="0",
                        segment_sentences="1",
                        input_=File(
                            file_name=file.name,
                            payload=fin,
                            mime_type=(
                                extra_info.get("file_type", "application/pdf") if extra_info else "application/pdf"
                            ),
                        ),
                    ),
                )

                if not resp.is_success:
                    self.logger.error(f"Failed to process {file.name}: {resp.content}")
                    return []

                article: Article = TEI.parse(resp.content, figures=self.load_figures)
                metadata = {
                    "file_name": file.name,
                    "file_title": article.title.title(),
                    "file_md5": article.identifier,
                }
                if extra_info:
                    metadata.update(extra_info)

                return self.process_text(article, metadata)

        except Exception as e:
            self.logger.error(f"Error processing file {file.name}: {e}")
            return []

    def load_data(
        self,
        directory_path: str,
        file_extension: str = ".pdf",
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> Dict[str, List[Document]]:
        """
        Load and process all PDF files in a directory.
        """
        if self.debug:
            self.logger.info(f"Processing directory: {directory_path}")
        directory = Path(directory_path)

        if not directory.is_dir():
            self.logger.error(f"Invalid directory: {directory_path}")
            return {}

        fs = fs or get_default_fs()
        results = {}

        for file_path in directory.rglob(f"*{file_extension}"):
            try:
                documents = self.load_file(file_path, extra_info, fs)
                if documents:
                    results[str(file_path)] = documents
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")

        if self.debug:
            self.logger.info(f"Processed {len(results)} files from {directory_path}")
        return results

    def process_text(self, article: Article, metadata: Dict) -> List[Document]:
        """
        Process TEI-parsed content into Document objects.
        """
        documents = []

        # Iterate over sections and paragraphs in the parsed content
        for section_idx, section in enumerate(article.sections):
            if not section.paragraphs:
                continue

            metadata.update(
                {
                    "section_idx": section_idx + 1,
                    "section_name": section.name,
                    "section_num": section.num or "",
                    "section_sentences_len": len(section.paragraphs),
                }
            )

            doc_text = ""
            for sentence_idx, sentence in enumerate(section.paragraphs):
                if not sentence:
                    continue

                if self.split_sentence:
                    metadata["sentence_idx"] = sentence_idx + 1
                    doc_text = ""

                doc_text += " ".join(phrase.text for phrase in sentence) + " "

                if self.split_sentence and len(doc_text) > MIN_SENTENCE_LEN:
                    append_docs(documents, doc_text, metadata)

            if not self.split_sentence:
                append_docs(documents, doc_text, metadata)

        if self.debug:
            self.logger.debug(f"Processed {len(documents)} documents.")

        return documents
