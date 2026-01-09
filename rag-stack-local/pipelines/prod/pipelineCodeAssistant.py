"""
title: Code Assistant Pipeline
author: Karol Siegieda
date: 2025-02-11
version: 1.0
license: MIT
description: A pipeline for analyzing source code using LlamaIndex and Deepseek Coder.
"""

import os
import types
from typing import List, Union, Generator, Iterator, Optional

import qdrant_client
from llama_index.core import PromptTemplate
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptType
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore

from typing import List, Union, Generator, Iterator, Optional
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama

from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatResponse
from llama_index.vector_stores.qdrant.utils import fastembed_sparse_encoder, SparseEncoderCallable

from libs.tools import (
    Observer,
    NameRetriever,
    CodeEntityExtractor,
    CodePromptDetector,
    NodeRetriever,
    FilterCodeCollectionRetriever,
    CodeMultiCollectionRetriever,
    SimilarityCutoffPostprocessor,
    TokenCounterPostprocessor,
    setup_logger,
)
from libs.variables import (
    DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL,
    GlobalRepository,
)

from libs.template import (
    CODE_ASSISTANT_TEMPLATE_STR
)

DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(CODE_ASSISTANT_TEMPLATE_STR, prompt_type=PromptType.CHOICE_SELECT)

ENV_SUFFIX = "_CODE_ASSISTANT"
logger = setup_logger(name="Code Assistant", debug=False)


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_MODEL_BASE_URL: str
        OLLAMA_MODEL_NAME: str
        OLLAMA_CONTEXT_WINDOW: int
        OLLAMA_REQUEST_TIMEOUT: float
        OLLAMA_TEMPERATURE: float
        OLLAMA_EMBEDDING_BASE_URL: str
        OLLAMA_EMBEDDING_MODEL_NAME: str
        OLLAMA_CHUNK_SIZE: int
        OLLAMA_CHUNK_OVERLAP: int
        QDRANT_SIMILARITY_TOP_K: int
        QDRANT_BASE_URL: str
        QDRANT_COLLECTION_NAME: str
        QDRANT_VECTOR_STORE_PARALLEL: int
        QDRANT_HYBRID_SEARCH: bool
        QDRANT_SIMILARITY_CUTOFF: float
        SPARSE_TEXT_EMBEDDING_MODEL: str
        CONTEXT_INPUT_OUTPUT_RATIO: str

        def __init__(self, **data):
            print(f"Initializing Code Assistant Valves with data: {data}")
            super().__init__(**data)

    def __init__(self):
        print("Initializing Code Assistant")
        self.name = "Code Assistant"
        self.qdrant = None
        self.ollama_llm = None
        self.ollama_embedder = None
        self.text_splitter = None
        self.code_detector = None
        self.filter_retrievers = []
        self.query_engine_postprocessors = []

        self.valves = self.Valves(
            **{
                "OLLAMA_MODEL_BASE_URL": os.getenv(f"OLLAMA_MODEL_BASE_URL{ENV_SUFFIX}", "http://10.255.240.156:11434"),
                "OLLAMA_MODEL_NAME": os.getenv(f"OLLAMA_MODEL_NAME{ENV_SUFFIX}", "qwen3-coder:30b"),
                "OLLAMA_CONTEXT_WINDOW": os.getenv(f"OLLAMA_CONTEXT_WINDOW{ENV_SUFFIX}", "61440"),
                "OLLAMA_REQUEST_TIMEOUT": os.getenv(f"OLLAMA_REQUEST_TIMEOUT{ENV_SUFFIX}", "120"),
                "OLLAMA_TEMPERATURE": os.getenv(f"OLLAMA_TEMPERATURE{ENV_SUFFIX}", "0.5"),
                "OLLAMA_EMBEDDING_BASE_URL": os.getenv(
                    f"OLLAMA_EMBEDDING_BASE_URL{ENV_SUFFIX}", "http://10.255.240.149:11435"
                ),
                "OLLAMA_EMBEDDING_MODEL_NAME": os.getenv(
                    f"OLLAMA_EMBEDDING_MODEL_NAME{ENV_SUFFIX}", "jina/jina-embeddings-v2-base-en:latest"
                ),
                "OLLAMA_CHUNK_SIZE": os.getenv(f"OLLAMA_CHUNK_SIZE{ENV_SUFFIX}", "1024"),
                "OLLAMA_CHUNK_OVERLAP": os.getenv(f"OLLAMA_CHUNK_OVERLAP{ENV_SUFFIX}", "256"),
                "QDRANT_SIMILARITY_TOP_K": os.getenv(f"QDRANT_SIMILARITY_TOP_K{ENV_SUFFIX}", "30"),
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
            }
        )

    def load_config(self):
        """Load configuration and initialize components."""
        print("Loading Code Assistant configuration")

        # Initialize the text splitter with chunk size and overlap
        self.text_splitter = SentenceSplitter(
            chunk_size=self.valves.OLLAMA_CHUNK_SIZE,
            chunk_overlap=int(self.valves.OLLAMA_CHUNK_SIZE * 0.1),  # 10% overlap
        )

        # Initialize the embedding model with the specified model name and base URL
        self.ollama_embedder = OllamaEmbedding(
            model_name=self.valves.OLLAMA_EMBEDDING_MODEL_NAME,
            base_url=self.valves.OLLAMA_EMBEDDING_BASE_URL,
        )

        self.ollama_llm = Ollama(
            model=self.valves.OLLAMA_MODEL_NAME,
            base_url=self.valves.OLLAMA_MODEL_BASE_URL,
            temperature=self.valves.OLLAMA_TEMPERATURE,
            request_timeout=self.valves.OLLAMA_REQUEST_TIMEOUT,
            context_window=self.valves.OLLAMA_CONTEXT_WINDOW,
            # num_gpu=1,
        )

        # Initialize Qdrant client
        self.qdrant = qdrant_client.QdrantClient(url=self.valves.QDRANT_BASE_URL)

        # Initialize Code Detector
        self.code_detector = CodePromptDetector(logger=logger)

        # Initialize sparse text embedding model if hybrid search is enabled
        sparse_text_embedding: Optional[SparseEncoderCallable] = None
        if self.valves.QDRANT_HYBRID_SEARCH:
            sparse_text_embedding = GlobalRepository.get_or_create(
                name="fastembed_sparse_encoder",
                factory=fastembed_sparse_encoder,
                model_name=self.valves.SPARSE_TEXT_EMBEDDING_MODEL,
            )

        # Create collection-specific retrievers using FilterCodeCollectionRetriever
        collections = self.valves.QDRANT_COLLECTION_NAME.split(",")
        for collection_name in collections:
            if not self.qdrant.collection_exists(collection_name):
                logger.warn(f"Collection '{collection_name}' does not exist.")
                continue

            # Create FilterCodeCollectionRetriever for each collection
            filter_retriever = FilterCodeCollectionRetriever(
                client=self.qdrant,
                collection_name=collection_name,
                embed_model=self.ollama_embedder,
                similarity_top_k=self.valves.QDRANT_SIMILARITY_TOP_K,
                parallel=self.valves.QDRANT_VECTOR_STORE_PARALLEL,
                sparse_text_embedding=sparse_text_embedding,
            )

            # Use the existing NameRetriever implementation from llama_index
            self.filter_retrievers.append(NameRetriever(name=f"code_{collection_name}", retriever=filter_retriever))

        self.query_engine_postprocessors = [
            SimilarityCutoffPostprocessor(similarity_cutoff=self.valves.QDRANT_SIMILARITY_CUTOFF),
            TokenCounterPostprocessor(
                context_size=self.valves.OLLAMA_CONTEXT_WINDOW,
                input_output_ratio=self.valves.CONTEXT_INPUT_OUTPUT_RATIO,
            ),
        ]

    async def on_startup(self):
        print("Loading Code Assistant Pipeline")
        self.load_config()

    async def on_valves_updated(self):
        print("Valves updated")
        self.load_config()

    async def on_shutdown(self):
        if self.qdrant:
            self.qdrant.close()

    async def inlet(self, body: dict, user: dict) -> dict:
        print("Processing inlet request")
        if "files" in body:
            for file in body["files"]:
                if "collection_name" in file:
                    if not "external_collections" in body:
                        body["external_collections"] = []
                    body["external_collections"].append(f"open-webui_{file['collection_name']}")
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        try:
            if not body or "messages" not in body or not body["messages"]:
                return {"messages": [{"content": "Invalid response format"}]}

            if body["messages"][-1]["content"] == "Empty Response":
                body["messages"][-1]["content"] = (
                    "Sorry, I couldn't find any relevant information. " "Please try asking in a different way."
                )
            return body
        except Exception as e:
            print(f"Error in outlet: {str(e)}")
            return {"messages": [{"content": "Error processing response"}]}

    def _extend_stream(self, response, event_key: str, observer: Observer) -> Generator:
        """
        Extend the response stream by yielding chunks from the response generator,
        stopping the observer, and appending references and summary.
        """
        generator = response.response_gen() if callable(response.response_gen) else response.response_gen
        for chunk in generator:
            yield chunk
        observer.stop(key=event_key)
        logger.info(f"Finished processing response with event key: {event_key}")
        yield observer.summary()

    def retrieve_with_entities(self, query: str, code_entities: dict, observer: Observer) -> List[NodeWithScore]:
        logger.info(f"Retrieval entities: {code_entities}")
        code_retriever = CodeMultiCollectionRetriever(
            retrievers=self.filter_retrievers,
            node_postprocessors=self.query_engine_postprocessors,
            logger=logger,
        )

        nodes = code_retriever.retrieve_code_multi(
            query=query,
            apply_filters=code_entities.get("apply_filters", False),
            class_names=code_entities.get("class_names", []),
            method_names=code_entities.get("method_names", []),
            namespaces=code_entities.get("namespaces", []),
            file_paths=code_entities.get("file_paths", []),
            search_context=code_entities.get("search_context"),
            observer=observer,
        )

        # This is workaroud for now (in case if applied filters returns no nodes, we don't want return empty response)
        # CodeMultiCollectionRetriever needs further refactor
        if len(nodes) == 0:
            logger.warning(f"Retrieval with filter does not found nodes. Switching to distance-based retrieval.")
            nodes = code_retriever.retrieve_code_multi(
                query=query,
                apply_filters=False,
                class_names=None,
                method_names=None,
                namespaces=None,
                file_paths=None,
                search_context=None,
                observer=observer,
            )
        logger.info(f"Results filtered to {len(nodes)} nodes in total for further processing.")
        return nodes

    def extract_entities(self, query: str, observer: Observer) -> dict:
        try:
            entities = CodeEntityExtractor.from_query(query, observer=observer, logger=logger)
            return entities.to_dict(query=query)
        except Exception as e:
            result = {"query": query}
            return result

    def _process_response(self, response, event_key, observer) -> Union[str, Generator, Iterator]:
        """
        Process the response from the query engine.
        """
        if isinstance(response, (StreamingResponse, types.GeneratorType)):
            return self._extend_stream(response, event_key, observer)
        else:
            content = response.response
            observer.stop(key=event_key)
            content += observer.summary()
            logger.info(f"Finished processing response with event key: {event_key}")
            return content

    def _extend_stream(self, stream, event_key, observer):
        try:
            if hasattr(stream, "response_gen"):
                for token in stream.response_gen:
                    yield token
            else:
                for chunk in stream:
                    if hasattr(chunk, "delta") and chunk.delta:
                        yield chunk.delta
        finally:
            observer.stop(key=event_key)
            yield observer.summary()
            logger.info(f"Finished streaming response with event key: {event_key}")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:

        observer = Observer()
        event_key = observer.start(name="LLM Query Engine")
        is_streaming = body.get("stream", True)

        if not self.code_detector.is_code(query=user_message, use_llm=True, observer=observer):
            code_entities = self.extract_entities(user_message, observer=observer)
            nodes = self.retrieve_with_entities(user_message, code_entities, observer)

            retriever = NodeRetriever(nodes)
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                llm=self.ollama_llm,
                text_qa_template=PromptTemplate(CODE_ASSISTANT_TEMPLATE_STR),
                streaming=is_streaming,
            )
            response = query_engine.query(user_message)
        else:
            if is_streaming:
                # Returns StreamingResponse
                response = self.ollama_llm.stream_chat(messages=[ChatMessage(role="user", content=user_message)])
            else:
                # Returns ChatResponse
                response = self.ollama_llm.chat(messages=[ChatMessage(role="user", content=user_message)])

        return self._process_response(response, event_key, observer)
