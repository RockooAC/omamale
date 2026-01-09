import argparse
import time
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import QueryBundle
from rag.qdrant.manager import QdrantManager
from rag.config import QDRANT_CONFIG, EMBEDDER_CONFIG, LLM_CONFIG
from rag.common import setup_logger

# Set up logging
logger = setup_logger("RAG Chat", debug=True)


def detect_embedder(client, collection_name):
    """
    Detect the embedder model based on the vector size in the Qdrant collection.
    """
    collection_info = client.get_collection(collection_name)
    logger.debug(f"Collection info: {collection_info}")

    try:
        vector_size = collection_info.config.params.vectors.size
    except AttributeError as e:
        logger.error(f"Unable to find vector size in collection info: {e}")
        raise ValueError("Could not determine vector size from collection info.")

    vector_size_mapping = EMBEDDER_CONFIG["VECTOR_SIZES"]

    if vector_size not in vector_size_mapping:
        raise ValueError(f"Unknown vector size {vector_size}. Cannot determine embedder model.")

    embedder_key = vector_size_mapping[vector_size]
    embedder_model = EMBEDDER_CONFIG["MODELS"][embedder_key]

    logger.info(f"Detected embedder model: {embedder_model['ollama']} based on vector size {vector_size}.")
    return embedder_model["ollama"]


def initialize_llm(embedder_model):
    """
    Initialize the LLM based on the detected embedder model.
    """
    if "jina" in embedder_model.lower():  # Use CODE_MODEL for Jina
        model_name = LLM_CONFIG["CODE_MODEL"]
        context_window = LLM_CONFIG["CONTEXT_WINDOW"]
        logger.info("Using CODE_MODEL for reasoning tasks.")
    else:  # Use DEFAULT_MODEL otherwise
        model_name = LLM_CONFIG["DEFAULT_MODEL"]
        context_window = LLM_CONFIG["CONTEXT_WINDOW"]
        logger.info("Using DEFAULT_MODEL for general tasks.")

    llm = Ollama(
        base_url=LLM_CONFIG["BASE_URL"],
        model=model_name,
        temperature=0.3,
        request_timeout=120,
        context_window=context_window,
    )
    logger.info(f"Initialized LLM: {llm.model} at {llm.base_url}")
    return llm


def query_documents(
    collection_name: str, query: str, local: bool, environment: str, top_k: int = 20, rerank: bool = False
):
    """
    Query the documents in Qdrant using a chat-based interaction.
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

        # Initialize LLM based on embedder model
        llm = initialize_llm(embedder_model_name)

        # Create index and query engine
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embedder,
        )

        # Measure time for Qdrant retrieval
        start_retrieval_time = time.time()
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=top_k)
        retrieval_time = time.time() - start_retrieval_time
        logger.info(f"Time taken for retrieving documents from Qdrant: {retrieval_time:.2f} seconds")

        # Measure time for LLM query
        logger.info("Querying documents...")
        start_llm_time = time.time()
        response = query_engine.query(query)
        llm_time = time.time() - start_llm_time
        logger.info(f"Time taken for querying LLM: {llm_time:.2f} seconds")

        # Optional reranking
        if rerank:
            logger.info("Applying reranking to retrieved documents...")
            reranker = LLMRerank(
                llm=llm,
                top_n=10,  # Adjust as needed
                choice_batch_size=5,  # Adjust as needed
                choice_select_prompt=PromptTemplate(
                    "Rank the following documents based on their relevance to the query:\n\n{context_str}\n\nQuery: {query_str}\nAnswer:\n"
                ),
            )
            response_with_rerank = reranker.postprocess_nodes(response.source_nodes, QueryBundle(query_str=query))
            response.response = response_with_rerank

        logger.info(f"Query response: {response.response}")
        return response.response

    except Exception as e:
        logger.error(f"An error occurred during query: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query documents in Qdrant using a chat-based interaction.")
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
        "--rerank",
        action="store_true",
        help="Enable reranking of retrieved documents.",
    )
    parser.set_defaults(local=False)
    args = parser.parse_args()

    # Call the query function
    query_documents(
        collection_name=args.collection,
        query=args.query,
        local=args.local,
        environment=args.e,
        rerank=args.rerank,
    )
