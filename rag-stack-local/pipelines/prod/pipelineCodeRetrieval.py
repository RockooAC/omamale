"""
title: Code Retrieval Pipeline
author: Karol Siegieda
date: 2025-02-11
version: 1.0
license: MIT
description: A pipeline for analyzing source code using LlamaIndex and Deepseek Coder.
"""

import asyncio
import os
import time
import hashlib
import re

from typing import List, Union, Generator, Iterator, Optional

import qdrant_client
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptType
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.llms.ollama import Ollama
from llama_index.legacy.vector_stores.qdrant_utils import SparseEncoderCallable
from llama_index.vector_stores.qdrant.utils import fastembed_sparse_encoder

from libs.tools import (
    Observer,
    NameRetriever,
    CodeEntityExtractor,
    FilterCodeCollectionRetriever,
    CodeMultiCollectionRetriever,
    SimilarityCutoffPostprocessor,
    TokenCounterPostprocessor,
    parse_code_nodes_to_markdown,
    setup_logger,
)

from libs.variables import (
    DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL,
    GlobalRepository,
)

ENV_SUFFIX = "_CODE_RETRIEVAL"
logger = setup_logger(name="Code Retrieval", debug=False)


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_EMBEDDING_BASE_URL: str
        OLLAMA_EMBEDDING_MODEL_NAME: str
        OLLAMA_CHUNK_SIZE: int
        OLLAMA_CHUNK_OVERLAP: int
        OLLAMA_REQUEST_TIMEOUT: float
        QDRANT_SIMILARITY_TOP_K: int
        QDRANT_BASE_URL: str
        QDRANT_COLLECTION_NAME: str
        QDRANT_VECTOR_STORE_PARALLEL: int
        QDRANT_HYBRID_SEARCH: bool
        QDRANT_SIMILARITY_CUTOFF: float
        SPARSE_TEXT_EMBEDDING_MODEL: str
        CONTEXT_INPUT_OUTPUT_RATIO: str
        CONTEXT_WINDOW: int

        def __init__(self, **data):
            logger.info(f"Initializing Code Retrieval Valves with data: {data}")
            super().__init__(**data)

    def __init__(self):
        print("Initializing Code Retrieval")
        self.name = "Code Retrieval"
        self.index = None
        self.qdrant = None
        self.multi_retriever = None
        self.text_splitter = None
        self.ollama_embedder = None
        self.code_retriever = None
        self.query_engine_postprocessors = []

        self.valves = self.Valves(
            **{
                # Ollama settings
                "OLLAMA_EMBEDDING_BASE_URL": os.getenv(
                    f"OLLAMA_EMBEDDING_BASE_URL{ENV_SUFFIX}", "http://10.255.240.149:11435"
                ),
                "OLLAMA_EMBEDDING_MODEL_NAME": os.getenv(
                    f"OLLAMA_EMBEDDING_MODEL_NAME{ENV_SUFFIX}", "jina/jina-embeddings-v2-base-en:latest"
                ),
                "OLLAMA_CHUNK_SIZE": os.getenv(f"OLLAMA_CHUNK_SIZE{ENV_SUFFIX}", "1024"),
                "OLLAMA_CHUNK_OVERLAP": os.getenv(f"OLLAMA_CHUNK_OVERLAP{ENV_SUFFIX}", "256"),
                "OLLAMA_REQUEST_TIMEOUT": os.getenv(f"OLLAMA_REQUEST_TIMEOUT{ENV_SUFFIX}", "120"),
                # Qdrant settings
                "QDRANT_SIMILARITY_TOP_K": os.getenv(f"QDRANT_SIMILARITY_TOP_K{ENV_SUFFIX}", "50"),
                "QDRANT_BASE_URL": os.getenv(f"QDRANT_BASE_URL{ENV_SUFFIX}", "http://10.255.240.18:6333"),
                "QDRANT_COLLECTION_NAME": os.getenv(
                    f"QDRANT_COLLECTION_NAME{ENV_SUFFIX}",
                    "codebase_prod_bm42",
                ),
                "QDRANT_VECTOR_STORE_PARALLEL": os.getenv(f"QDRANT_VECTOR_STORE_PARALLEL{ENV_SUFFIX}", "4"),
                "QDRANT_HYBRID_SEARCH": os.getenv(f"QDRANT_HYBRID_SEARCH{ENV_SUFFIX}", "False").lower() == "true",
                "QDRANT_SIMILARITY_CUTOFF": os.getenv(f"QDRANT_SIMILARITY_CUTOFF{ENV_SUFFIX}", "0.5"),
                "SPARSE_TEXT_EMBEDDING_MODEL": os.getenv(
                    f"SPARSE_TEXT_EMBEDDING_MODEL{ENV_SUFFIX}",
                    DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL,
                ),
                "CONTEXT_INPUT_OUTPUT_RATIO": os.getenv(f"CONTEXT_INPUT_OUTPUT_RATIO{ENV_SUFFIX}", "1:1"),
                "CONTEXT_WINDOW": os.getenv(f"CONTEXT_WINDOW{ENV_SUFFIX}", "32768"),
            }
        )

    def load_config(self):
        """Load configuration and initialize components."""
        logger.info("Loading Code Retrieval configuration")

        # Initialize the text splitter with chunk size and overlap
        self.text_splitter = SentenceSplitter(
            chunk_size=self.valves.OLLAMA_CHUNK_SIZE,
            chunk_overlap=int(self.valves.OLLAMA_CHUNK_SIZE * 0.2),  # 20% overlap
        )

        # Initialize the embedding model with the specified model name and base URL
        self.ollama_embedder = OllamaEmbedding(
            model_name=self.valves.OLLAMA_EMBEDDING_MODEL_NAME,
            base_url=self.valves.OLLAMA_EMBEDDING_BASE_URL,
        )

        # Initialize Qdrant client
        self.qdrant = qdrant_client.QdrantClient(url=self.valves.QDRANT_BASE_URL)

        # Initialize sparse text embedding model if hybrid search is enabled
        sparse_text_embedding: Optional[SparseEncoderCallable] = None
        if self.valves.QDRANT_HYBRID_SEARCH:
            sparse_text_embedding = GlobalRepository.get_or_create(
                name="fastembed_sparse_encoder",
                factory=fastembed_sparse_encoder,
                model_name=self.valves.SPARSE_TEXT_EMBEDDING_MODEL,
            )

        # Create collection-specific retrievers using FilterCodeCollectionRetriever
        filter_retrievers = []
        collections = self.valves.QDRANT_COLLECTION_NAME.split(",")
        for collection_name in collections:
            if not self.qdrant.collection_exists(collection_name):
                logger.warn(f"Collection '{collection_name}' does not exist.")
                continue

            # Create FilterCodeCollectionRetriever for each collection
            code_retriever = FilterCodeCollectionRetriever(
                client=self.qdrant,
                collection_name=collection_name,
                embed_model=self.ollama_embedder,
                similarity_top_k=self.valves.QDRANT_SIMILARITY_TOP_K,
                parallel=self.valves.QDRANT_VECTOR_STORE_PARALLEL,
                sparse_text_embedding=sparse_text_embedding,
            )

            # Use the existing NameRetriever implementation from llama_index
            filter_retrievers.append(NameRetriever(name=f"code_{collection_name}", retriever=code_retriever))

        query_engine_postprocessors = [
            SimilarityCutoffPostprocessor(similarity_cutoff=self.valves.QDRANT_SIMILARITY_CUTOFF),
            TokenCounterPostprocessor(
                context_size=self.valves.CONTEXT_WINDOW,
                input_output_ratio=self.valves.CONTEXT_INPUT_OUTPUT_RATIO,
                logger=logger,
            ),
        ]

        # Create CodeMultiCollectionRetriever
        self.code_retriever = CodeMultiCollectionRetriever(
            retrievers=filter_retrievers,
            node_postprocessors=query_engine_postprocessors,
            logger=logger,
        )

    async def on_startup(self):
        """Initialize the pipeline on startup."""
        print("Loading Code Retrieval Pipeline")
        self.load_config()

    async def on_valves_updated(self):
        """Handle configuration updates."""
        print("Valves updated")
        if self.qdrant:
            self.qdrant.close()
        self.load_config()

    async def on_shutdown(self):
        """Clean up resources on shutdown."""
        if self.qdrant:
            self.qdrant.close()

    def retrieve_with_entities(self, query: str, code_entities: dict, observer: Observer) -> List[NodeWithScore]:
        logger.info(f"Retrieval entities: {code_entities}")
        nodes = self.code_retriever.retrieve_code_multi(
            query=query,
            apply_filters=code_entities.get("apply_filters", False),
            class_names=code_entities.get("class_names", []),
            method_names=code_entities.get("method_names", []),
            namespaces=code_entities.get("namespaces", []),
            file_paths=code_entities.get("file_paths", []),
            search_context=code_entities.get("search_context"),
            observer=observer,
        )
        logger.info(f"Found {len(nodes)} nodes.")
        return nodes

    def extract_entities(self, query: str, observer: Observer) -> dict:
        try:
            entities = CodeEntityExtractor.from_query(query, observer=observer, logger=logger)
            return entities.to_dict(query=query)
        except Exception as e:
            result = {"query": query}
            return result

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Extract entities from the query
        observer = Observer()
        code_entities = self.extract_entities(user_message, observer=observer)
        nodes = self.retrieve_with_entities(user_message, code_entities, observer)

        return parse_code_nodes_to_markdown(user_message, nodes, observer)
