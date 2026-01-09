"""
title: Qdrant Retriever Pipeline
author: Paweł Jasionowski
date: 2024-08-02
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a qdrant collection using the Llama Index library with Ollama embeddings.
"""
import os
from typing import List, Union, Generator, Iterator

import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.indices.list.base import ListRetrieverMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.core.prompts import PromptType
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.qdrant.utils import fastembed_sparse_encoder

from libs.tools import (SimilarityCutoffPostprocessor,
                        MultiCollectionRetriever, BuildReference, NameRetriever, Observer, parse_nodes_to_markdown,
                        setup_logger)

from libs.variables import (
    GlobalRepository,
    DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_SENTENCE_TRANSFORMER_RERANK_MODEL,
)

# Logger Configuration
logger = setup_logger(name="Qdrant Retriever", debug=False)

DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (
    "A list of documents is shown below. Each document has a number next to it along "
    "with a summary of the document. A question is also provided. \n"
    "Respond with the numbers of the documents "
    "you should consult to answer the question, in order of relevance, as well \n"
    "as the relevance score. The relevance score is a number from 1-100 based on "
    "how relevant you think the document is to the question.\n"
    "Do not include any documents that are not relevant to the question. \n"
    "Only response the ranking results, do not say any word or explain.\n"
    "\n"
    "Example:\n"
    "Doc: 9, Relevance: 74\n"
    "Doc: 3, Relevance: 43\n"
    "Doc: 7, Relevance: 32\n"
    "\n"
    "Let's try this now: \n\n"
    "{context_str}\n"
    "Question: {query_str}\n"
    "Answer:\n"
)

DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(DEFAULT_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT)
ENV_SUFFIX = "_QDRANT_RETRIEVER"

class Pipeline:
    class Valves(BaseModel):
        OLLAMA_EMBEDDING_BASE_URL: str
        OLLAMA_EMBEDDING_MODEL_NAME: str
        OLLAMA_CHUNK_SIZE: int
        OLLAMA_CHUNK_OVERLAP: int
        OLLAMA_RERANK_ACTIVE: bool
        OLLAMA_RERANK_BASE_URL: str
        OLLAMA_RERANK_MODEL_NAME: str
        OLLAMA_RERANK_TEMPERATURE: float
        OLLAMA_RERANK_TOP_N: int
        OLLAMA_RERANK_CHOICE_BATCH_SIZE: int
        QDRANT_BASE_URL: str
        QDRANT_COLLECTION_NAME: str
        QDRANT_VECTOR_STORE_PARALLEL: int
        QDRANT_SIMILARITY_TOP_K: int
        QDRANT_HYBRID_SEARCH: bool
        QDRANT_SIMILARITY_CUTOFF_ACTIVE: bool
        QDRANT_SIMILARITY_CUTOFF: float
        SENTENCE_TRANSFORMER_RERANK_ACTIVE: bool
        SENTENCE_TRANSFORMER_RERANK_MODEL: str
        SENTENCE_TRANSFORMER_RERANK_TOP_N: int
        PIPELINE_TIME_RECORDING: bool
        SPARSE_TEXT_EMBEDDING_MODEL: str

        def __init__(self, **data):
            logger.info(f"Initializing Valves with data: {data}")
            super().__init__(**data)

    def __init__(self):
        logger.info("Initializing Qdrant Retriever")
        self.name = "Qdrant Retriever"
        self.index = None
        self.qdrant = None
        self.ollama_embedder = None
        self.text_splitter = None
        self.multi_retriever = None
        self.ollama_rerank = None
        self.query_engine_postprocessors = []
        self.retrievers = []
        self.sparse_text_embedding = None

        self.valves = self.Valves(
            **{
                # Ollama settings
                "OLLAMA_EMBEDDING_BASE_URL":            os.getenv(f"OLLAMA_EMBEDDING_BASE_URL{ENV_SUFFIX}", "http://10.255.240.149:11434"),
                "OLLAMA_EMBEDDING_MODEL_NAME":          os.getenv(f"OLLAMA_EMBEDDING_MODEL_NAME{ENV_SUFFIX}", "gte-qwen2.5-instruct-q5"),
                "OLLAMA_CHUNK_SIZE":                    os.getenv(f"OLLAMA_CHUNK_SIZE{ENV_SUFFIX}", "1024"),
                "OLLAMA_CHUNK_OVERLAP":                 os.getenv(f"OLLAMA_CHUNK_OVERLAP{ENV_SUFFIX}", "256"),
                # Ollama rerank settings
                "OLLAMA_RERANK_ACTIVE":                 os.getenv(f"OLLAMA_RERANK_ACTIVE{ENV_SUFFIX}", "True"),
                "OLLAMA_RERANK_BASE_URL":               os.getenv(f"OLLAMA_RERANK_BASE_URL{ENV_SUFFIX}", "http://10.255.240.156:11434"),
                "OLLAMA_RERANK_MODEL_NAME":             os.getenv(f"OLLAMA_RERANK_MODEL_NAME{ENV_SUFFIX}", "llama3.2:3b"),
                "OLLAMA_RERANK_TEMPERATURE":            os.getenv(f"OLLAMA_RERANK_TEMPERATURE{ENV_SUFFIX}", "0.5"),
                "OLLAMA_RERANK_TOP_N":                  os.getenv(f"OLLAMA_RERANK_TOP_N{ENV_SUFFIX}", "10"),
                "OLLAMA_RERANK_CHOICE_BATCH_SIZE":      os.getenv(f"OLLAMA_RERANK_CHOICE_BATCH_SIZE{ENV_SUFFIX}", "5"),
                # Qdrant settings
                "QDRANT_BASE_URL":                      os.getenv(f"QDRANT_BASE_URL{ENV_SUFFIX}", "http://10.255.240.18:6333"),
                "QDRANT_COLLECTION_NAME":               os.getenv(f"QDRANT_COLLECTION_NAME{ENV_SUFFIX}", "confluence_prod_bm42,docusaurus_prod_bm42,pdf_prod_bm42"),
                "QDRANT_SIMILARITY_TOP_K":              os.getenv(f"QDRANT_SIMILARITY_TOP_K{ENV_SUFFIX}", "7"),
                "QDRANT_VECTOR_STORE_PARALLEL":         os.getenv(f"QDRANT_VECTOR_STORE_PARALLEL{ENV_SUFFIX}", "4"),
                "QDRANT_HYBRID_SEARCH":                 os.getenv(f"QDRANT_HYBRID_SEARCH{ENV_SUFFIX}", "True"),
                "QDRANT_SIMILARITY_CUTOFF_ACTIVE":      os.getenv(f"QDRANT_SIMILARITY_CUTOFF_ACTIVE{ENV_SUFFIX}", "True"),
                "QDRANT_SIMILARITY_CUTOFF":             os.getenv(f"QDRANT_SIMILARITY_CUTOFF{ENV_SUFFIX}", "0.5"),
                # Sentence transformer rerank settings
                "SENTENCE_TRANSFORMER_RERANK_ACTIVE":   os.getenv(f"SENTENCE_TRANSFORMER_RERANK_ACTIVE{ENV_SUFFIX}", "True"),
                "SENTENCE_TRANSFORMER_RERANK_MODEL":    os.getenv(f"SENTENCE_TRANSFORMER_RERANK_MODEL{ENV_SUFFIX}", DEFAULT_SENTENCE_TRANSFORMER_RERANK_MODEL),
                "SENTENCE_TRANSFORMER_RERANK_TOP_N":    os.getenv(f"SENTENCE_TRANSFORMER_RERANK_TOP_N{ENV_SUFFIX}", "10"),
                # Pipeline settings
                "PIPELINE_TIME_RECORDING":              os.getenv(f"PIPELINE_TIME_RECORDING{ENV_SUFFIX}", "True"),
                "SPARSE_TEXT_EMBEDDING_MODEL":          os.getenv(f"SPARSE_TEXT_EMBEDDING_MODEL{ENV_SUFFIX}", DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL),
            }
        )

    def load_config(self):
        """
        Load the configuration. This function is called when the server is started and when the valves are updated.
        It sets various settings for the Ollama model and Qdrant client, and initializes the vector store index.
        """
        logger.info("Loading configuration")

        # Initialize the text splitter with chunk size and overlap
        self.text_splitter = SentenceSplitter(
            chunk_size=self.valves.OLLAMA_CHUNK_SIZE,
            chunk_overlap=self.valves.OLLAMA_CHUNK_OVERLAP,
        )
        # Initialize the embedding model with the specified model name and base URL
        self.ollama_embedder = OllamaEmbedding(
            model_name=self.valves.OLLAMA_EMBEDDING_MODEL_NAME,
            base_url=self.valves.OLLAMA_EMBEDDING_BASE_URL,
        )
        # Initialize the rerank model with the specified model name, base URL, temperature, and context window
        self.ollama_rerank = Ollama(
            base_url=self.valves.OLLAMA_RERANK_BASE_URL,
            model=self.valves.OLLAMA_RERANK_MODEL_NAME,
            temperature=self.valves.OLLAMA_RERANK_TEMPERATURE,
            context_window=self.valves.OLLAMA_RERANK_CHOICE_BATCH_SIZE*self.valves.OLLAMA_CHUNK_SIZE+500,
            request_timeout=DEFAULT_REQUEST_TIMEOUT,
        )
        # Connect to the Qdrant client using the base URL
        self.qdrant = qdrant_client.QdrantClient(url=self.valves.QDRANT_BASE_URL)
        logger.info(f"Initialized rerank client with URL: {self.valves.QDRANT_BASE_URL}")

        # Initialize the query engine postprocessors
        self.query_engine_postprocessors = []
        # Set the similarity cutoff for the query engine
        if self.valves.QDRANT_SIMILARITY_CUTOFF_ACTIVE:
            self.query_engine_postprocessors.append(
                SimilarityCutoffPostprocessor(similarity_cutoff=self.valves.QDRANT_SIMILARITY_CUTOFF)
            )

        # Set the sentence transformer rerank model and top n for the query engine
        if self.valves.SENTENCE_TRANSFORMER_RERANK_ACTIVE:
            self.query_engine_postprocessors.append(
                GlobalRepository.get_or_create(
                    name="sentence_transformer_rerank",
                    factory=SentenceTransformerRerank,
                    model=self.valves.SENTENCE_TRANSFORMER_RERANK_MODEL,
                    top_n=self.valves.SENTENCE_TRANSFORMER_RERANK_TOP_N,
                    keep_retrieval_score=True,
                )
            )
        # Set the LLM rerank model and top n for the query engine
        if self.valves.OLLAMA_RERANK_ACTIVE:
            self.query_engine_postprocessors.append(
                LLMRerank(
                    llm=self.ollama_rerank,
                    top_n=self.valves.OLLAMA_RERANK_TOP_N,
                    choice_batch_size=self.valves.OLLAMA_RERANK_CHOICE_BATCH_SIZE,
                    choice_select_prompt=DEFAULT_CHOICE_SELECT_PROMPT,
                )
            )

        # Add the reference postprocessor to the query engine
        self.query_engine_postprocessors.append(BuildReference(reference_key="reference"))


        self.sparse_text_embedding = None
        if self.valves.QDRANT_HYBRID_SEARCH:
            self.sparse_text_embedding = GlobalRepository.get_or_create(
                name="fastembed_sparse_encoder",
                factory=fastembed_sparse_encoder,
                model_name=self.valves.SPARSE_TEXT_EMBEDDING_MODEL,
            )

        # Build the retrievers for the specified collections
        self.retrievers = []
        collections = self.valves.QDRANT_COLLECTION_NAME.split(",")
        for collection_name in collections:
            collection_retriever = self._build_retriever(collection_name)
            if collection_retriever is not None:
                logger.info(f"Loaded collection: {collection_name}")
                self.retrievers.append(NameRetriever(name=collection_name, retriever=collection_retriever))
        pass

    def _build_retriever(self, collection_name: str):
        """
        Build a retriever for the specified collection name.
        Args:
            collection_name (str): The name of the collection.
        Returns:
            IndexNode: The index node for the collection.
        """
        if not self.qdrant.collection_exists(collection_name):
            logger.info(f"Collection {collection_name} does not exist. Creating collection.")
            return None

        # Connect to the Qdrant vector store collection with parallelism and collection name
        vector_store = QdrantVectorStore(
            client=self.qdrant,
            parallel=self.valves.QDRANT_VECTOR_STORE_PARALLEL,
            collection_name=collection_name,
            sparse_doc_fn=self.sparse_text_embedding,
            sparse_query_fn=self.sparse_text_embedding,
            enable_hybrid=self.sparse_text_embedding is not None,
        )
        # Create a storage context from the vector store
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
        )
        retriever = VectorStoreIndex.from_vector_store(
            storage_context=storage_context,
            vector_store=vector_store,
            embed_model=self.ollama_embedder,
        ).as_retriever(
            retriever_mode=ListRetrieverMode.DEFAULT,
            similarity_top_k=self.valves.QDRANT_SIMILARITY_TOP_K,
        )
        return retriever

    async def on_startup(self):
        """
        This function is called when the server is started.
        It loads the configuration for the Assistant Redge Pipeline.
        """
        logger.info("Loading Assistant Redge Pipeline")
        self.load_config()

    async def on_valves_updated(self):
        """
        This function is called when the valves are updated.
        It reloads the configuration and closes the existing Qdrant client connection.
        """
        logger.info("Valves updated")
        self.qdrant.close()
        self.load_config()

    async def on_shutdown(self):
        """
        This function is called when the server is stopped.
        Currently, it does not perform any actions.
        """
        self.qdrant.close()

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """
        Process the user message through the pipeline.

        Args:
            user_message (str): The message from the user.
            model_id (str): The ID of the model to use.
            messages (List[dict]): A list of message dictionaries.
            body (dict): Additional data for processing.

        Returns:
            Union[str, Generator, Iterator]: The response from the query engine, which can be a string, generator, or iterator.
        """
        # Create an observer for the query engine
        observer = Observer()

        # Create a multi retriever for retrieving nodes from the collections
        multi_retriever = MultiCollectionRetriever(
            retrievers=self.retrievers,
            node_postprocessors=self.query_engine_postprocessors,
            observer=observer,
        )

        # Retrieve relevant information using the index retriever
        logger.info(f"Retrieving nodes for user message: {user_message}")
        nodes = multi_retriever.retrieve(user_message)
        logger.info(f"Retrieved nodes: {len(nodes)}")
        return parse_nodes_to_markdown(user_message, nodes, observer)