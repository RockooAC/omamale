"""
Example:
    python retrieve.py -c "test-embed-pdf" -q "Explain the encoding process in detail" --local
"""

import argparse
import time
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from rag.qdrant.manager import QdrantManager
from rag.config import QDRANT_CONFIG, EMBEDDER_CONFIG
from rag.common import setup_logger

# Set up logging
logger = setup_logger("Retriever", debug=True)


def detect_embedder(client, collection_name):
    """
    Detect the embedder model based on the vector size in the Qdrant collection.
    """
    # Fetch collection info
    collection_info = client.get_collection(collection_name)
    logger.debug(f"Collection info: {collection_info}")

    # Access vector size from the nested structure
    try:
        vector_size = collection_info.config.params.vectors.size
    except AttributeError as e:
        logger.error(f"Unable to find vector size in collection info: {e}")
        raise ValueError("Could not determine vector size from collection info.")

    # Map vector size to embedder model
    vector_size_mapping = EMBEDDER_CONFIG["VECTOR_SIZES"]

    if vector_size not in vector_size_mapping:
        raise ValueError(f"Unknown vector size {vector_size}. Cannot determine embedder model.")

    embedder_key = vector_size_mapping[vector_size]
    embedder_model = EMBEDDER_CONFIG["MODELS"][embedder_key]

    logger.info(f"Detected embedder model: {embedder_model['ollama']} based on vector size {vector_size}.")
    return embedder_model["ollama"]


def retrieve_chunks(collection_name: str, query: str, local: bool, environment: str, top_k: int = 10):
    """
    Retrieve relevant chunks from a Qdrant collection based on a query.
    """
    try:
        # Determine Qdrant URL
        qdrant_url = QDRANT_CONFIG["BASE_URL_LOCAL"] if local else QDRANT_CONFIG["BASE_URL"]

        # Initialize Qdrant Manager
        qdrant_manager = QdrantManager(qdrant_url, collection_name)
        logger.info(f"Connected to Qdrant at {qdrant_url}.")

        # Verify collection exists
        if not qdrant_manager.collection_exists():
            logger.error(f"Collection '{collection_name}' does not exist.")
            return

        # Detect embedder model dynamically
        embedder_model_name = detect_embedder(qdrant_manager.client, collection_name)

        # Determine embedder URL based on environment
        embedder_url = EMBEDDER_CONFIG["KS_BASE_URL"] if environment == "ks" else EMBEDDER_CONFIG["BASE_URL"]

        # Create embedding instance
        embedder = OllamaEmbedding(
            base_url=embedder_url,
            model_name=embedder_model_name,
        )
        logger.info(f"Using embedder: {embedder.model_name} at {embedder_url}")

        # Create vector store and storage context
        vector_store = qdrant_manager.get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create a retriever using VectorStoreIndex
        index = VectorStoreIndex.from_vector_store(
            storage_context=storage_context,
            vector_store=vector_store,  # Pass vector_store explicitly
            embed_model=embedder,
        )
        retriever = index.as_retriever(
            similarity_top_k=top_k,
            search_type="dense",
            include_metadata=True,
        )

        # Measure retrieval timing
        retrieval_start_time = time.time()

        # Retrieve relevant chunks
        chunks = retriever.retrieve(query)

        retrieval_end_time = time.time()
        retrieval_time = retrieval_end_time - retrieval_start_time

        if not chunks:
            logger.info("No relevant chunks retrieved.")
            return

        # Perform manual metadata filtering
        filtered_chunks = [chunk for chunk in chunks if chunk.metadata.get("code_type") == "method"]

        if not filtered_chunks:
            logger.info("No chunks matched the metadata filter.")
            return

        # Print retrieved chunks
        logger.info("Retrieved chunks:")
        for i, chunk in enumerate(filtered_chunks, start=1):
            logger.info(f"Chunk {i}:")
            logger.info(f"Text: {chunk.text}")
            logger.info(f"Metadata: {chunk.metadata}")
            logger.info("===" * 20)

        logger.info(f"Qdrant retrieval time: {retrieval_time:.2f} seconds")
        return filtered_chunks

    except Exception as e:
        logger.error(f"An error occurred during retrieval: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks from a Qdrant collection.")
    parser.add_argument("-c", "--collection", type=str, required=True, help="Qdrant collection name.")
    parser.add_argument("-q", "--query", type=str, required=True, help="Query for retrieval.")
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
        "--topk",
        type=int,
        default=10,
        help="How many similar chunks to retrieve from db.",
    )
    parser.set_defaults(local=False)
    args = parser.parse_args()

    # Call the retrieval function
    retrieve_chunks(
        collection_name=args.collection,
        query=args.query,
        local=args.local,
        environment=args.e,
        top_k=args.topk,
    )
