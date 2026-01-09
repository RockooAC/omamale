"""
author: Karol Siegieda
date: 2025-03-10
description: Helper class QdrantManager for retrival purposes
"""

from qdrant_client import QdrantClient, models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from src.rag.common import setup_logger


class QdrantManager:
    def __init__(self, base_url: str, collection_name: str, vector_size: int = None):
        """
        base_url (str): Qdrant base URL.
        collection_name (str): Name of the Qdrant collection.
        vector_size (int, optional): Dimension of vectors for the collection.
        """
        self.client = QdrantClient(base_url)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.logger = setup_logger("QdrantManager", debug=True)

    def setup_collection(self):
        if self.client.collection_exists(self.collection_name):
            # Get existing collection configuration
            collection_info = self.client.get_collection(self.collection_name)
            existing_distance = collection_info.config.params.vectors.distance

            # If the existing distance is not COSINE, recreate the collection
            if existing_distance != models.Distance.COSINE:
                self.logger.warning(f"Recreating '{self.collection_name}' with COSINE similarity.")
                self.client.delete_collection(self.collection_name)
                self.create_collection()
            else:
                self.logger.info(f"Collection '{self.collection_name}' already exists with COSINE similarity.")
        else:
            self.create_collection()

    def create_collection(self):
        if not self.vector_size:
            raise ValueError("Vector size must be specified to create a new collection.")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
        )
        self.logger.info(f"Collection '{self.collection_name}' created with COSINE similarity.")

    def index_documents(self, documents):
        if not self.collection_exists():
            raise ValueError(f"Collection '{self.collection_name}' does not exist.")

        # Filter out invalid documents
        documents = self.filter_invalid_chunks(documents)

        if not documents:
            self.logger.error("No valid documents to index.")
            return

        # Add documents to the collection
        self.logger.info(f"Indexing {len(documents)} documents into '{self.collection_name}'...")
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(id=doc["id"], vector=doc["vector"], payload=doc["metadata"]) for doc in documents
                ],
            )
            self.logger.info(f"Indexing complete.")
        except Exception as e:
            self.logger.error(f"Error during indexing: {e}")

    def filter_invalid_chunks(self, documents):
        valid_documents = []
        for doc in documents:
            text = doc.get("text", "").strip()
            if not text:
                self.logger.warning(f"Skipping document with empty text: {doc.get('id', 'unknown')}")
                continue
            # Exclude binary-like content
            if all(c in "0123456789abcdefABCDEF, \n" for c in text):
                self.logger.warning(f"Skipping document with binary-like content: {doc.get('id', 'unknown')}")
                continue
            valid_documents.append(doc)
        return valid_documents

    def get_vector_store(self):
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            batch_size=50,  # Smaller batch size for better monitoring
            metadata_payload_keys=["token_count"],
        )

    def collection_exists(self):
        return self.client.collection_exists(self.collection_name)

    def get_collection_info(self):
        if not self.collection_exists():
            raise ValueError(f"Collection '{self.collection_name}' does not exist.")
        return self.client.get_collection(self.collection_name)
