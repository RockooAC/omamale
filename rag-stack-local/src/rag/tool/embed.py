"""
author: Karol Siegieda
date: 2025-02-24
description: Generic script for embedding different documents (pdf, markdown, cpp source code)

Usage:
    python embed.py -d "~/docs/pdf_collection" -c "pdf_documents" -t pdf -e prod --local
"""

import re
import argparse
import logging
import os
import sys
from pathlib import Path

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document

from src.rag.readers.pdf import GrobidPDFReader
from src.rag.readers.markdown import MarkdownReader
from src.rag.readers.cpp import CppReader
from src.rag.qdrant.manager import QdrantManager
from src.rag.common import setup_logger
from src.rag.libs import TokenCountAnnotator

from src.rag.config import (
    EMBEDDER_CONFIG,
    QDRANT_CONFIG,
    GROBID_CONFIG,
)

EMBEDDER_MODELS = EMBEDDER_CONFIG["MODELS"]

# Set up the main logger for the script
logger = setup_logger("Embedding", debug=True)


def get_reader_and_model(doc_type, debug=False):
    if doc_type == "pdf":
        reader = GrobidPDFReader(GROBID_CONFIG["BASE_URL"], split_sentence=GROBID_CONFIG["SPLIT_SENTENCE"])
        embedder_model = EMBEDDER_CONFIG["MODELS"]["QWEN"]
    elif doc_type == "markdown":
        reader = MarkdownReader(debug=debug, normalize=True)
        embedder_model = EMBEDDER_CONFIG["MODELS"]["QWEN"]
    elif doc_type == "cpp":
        reader = CppReader(debug=debug)
        embedder_model = EMBEDDER_CONFIG["MODELS"]["JINA"]
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")

    return reader, embedder_model


def filter_invalid_documents(documents):
    valid_documents = []
    for doc in documents:
        doc_id = getattr(doc, "id_", "unknown")
        text = doc.text if hasattr(doc, "text") else None

        if not text or not text.strip():
            logger.debug(f"Empty text in document {doc_id}, type: {type(doc)}")
            continue

        # Less restrictive binary content check
        if len(text.split()) > 10 and all(c in "0123456789abcdefABCDEF, \n" for c in text):
            logger.warning(f"Binary-like content in document {doc_id}")
            continue

        valid_documents.append(doc)

    logger.info(f"Filtered {len(valid_documents)} valid documents out of {len(documents)} total.")
    return valid_documents


def main(directory, collection_name, doc_type, embedder_url, local, debug=False):
    try:
        # Initialize reader and embedder model
        reader, embedder_model = get_reader_and_model(doc_type, debug)
        directory = os.path.expanduser(directory)

        if not os.path.exists(directory):
            logger.error(f"Directory/file does not exist: {directory}")
            return 1
        if doc_type == "cpp" and not directory.endswith(".digest"):
            logger.error(f"Expected .digest file but got: {directory}")
            return 1

        # Load embedding model
        embedder = OllamaEmbedding(base_url=embedder_url, model_name=embedder_model["ollama"])
        vector_size = len(embedder.get_text_embedding("test"))
        logger.info(f"Embedder loaded: {embedder_model['ollama']} (size: {vector_size})")

        # Set up Qdrant collection
        qdrant_url = QDRANT_CONFIG["BASE_URL_LOCAL"] if local else QDRANT_CONFIG["BASE_URL"]
        qdrant_manager = QdrantManager(qdrant_url, collection_name, vector_size)
        qdrant_manager.setup_collection()

        # Load and filter documents
        raw_data = reader.load_data(directory, extra_info={"source": collection_name})
        logger.debug(f"Raw data files: {list(raw_data.keys())}")

        # Convert to documents based on reader type
        documents = []
        if doc_type == "cpp":
            # Handle CPP reader
            for file_name, file_info in raw_data.items():
                chunks = file_info["chunks"]
                source_code = file_info["source"]
                extra_info = file_info.get("extra_info")

                for chunk in chunks:
                    doc = chunk.to_document(source_code, extra_info)
                    if doc is not None:
                        documents.append(doc)
        else:
            # Handle PDF/Markdown reader
            for file_path, docs in raw_data.items():
                documents.extend(docs)

        documents = filter_invalid_documents(documents)

        # Mandatory since it's a valuable metric in pipelines
        # We are doing it after chunks are created, because we want to have matadata count also
        try:
            annotator = TokenCountAnnotator(embedder_model["hf"], debug, logger)
            annotator.annotate(documents)
        except Exception as e:
            logger.error("Unable to annotate documents")
            return 1

        # Proceed with embedding and indexing
        vector_store = qdrant_manager.get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex.from_documents(
            documents,
            embed_model=embedder,
            storage_context=storage_context,
            show_progress=True,
        )

        logger.info("Embedding completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Embedding error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed documents into Qdrant.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Directory or file path for the documents.")
    parser.add_argument("-c", "--collection", type=str, required=True, help="Qdrant collection name.")
    parser.add_argument(
        "-t", "--type", type=str, choices=["pdf", "markdown", "cpp"], required=True, help="Type of documents."
    )
    parser.add_argument(
        "-e",
        choices=["prod", "ks"],
        default="prod",
        help="Select the embedder environment: 'prod' (default) or 'ks'.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Qdrant database (default is remote).",
    )
    parser.add_argument(
        "--remote",
        dest="local",
        action="store_false",
        help="Use remote Qdrant database (default).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging.",
    )
    parser.set_defaults(local=False)
    args = parser.parse_args()

    # Map embedder environment to URL
    embedder_url = EMBEDDER_CONFIG["KS_BASE_URL"] if args.e == "ks" else EMBEDDER_CONFIG["BASE_URL"]

    exit_code = main(
        args.directory,
        args.collection,
        args.type,  # Use args.type as document type
        embedder_url,  # Pass the resolved embedder URL
        args.local,
        args.debug,
    )
    sys.exit(exit_code)
