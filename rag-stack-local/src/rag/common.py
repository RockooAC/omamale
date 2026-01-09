# Function to detect encoding of a file
import re
import logging
import spacy
from qdrant_client import QdrantClient
from pathlib import Path
from typing import Optional, List


# Load the spaCy model
# Install the spaCy model with: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


def setup_logger(name: str, debug: bool = False, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting and optional file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Check if a collection exists in Qdrant
def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        # Check if the collection exists
        exists = client.collection_exists(collection_name)
        return exists
    except Exception as e:
        logging.error(f"An error occurred while checking for collection existence: {e}")
        return False


def normalize_text(text: str) -> str:
    """
    Normalize text by removing newlines, extra spaces, and converting to UTF-8 encoding.
    """
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # convert to UTF-8 encoding
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    return preprocess_text(text)


def preprocess_text(text) -> str:
    """
    Preprocess the input text by lemmatizing and removing stop words.
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])
