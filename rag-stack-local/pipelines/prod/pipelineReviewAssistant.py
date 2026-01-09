#!/usr/bin/python3

"""
title: Review Assistant Pipeline
author: Oskar Bukowski
date: inf
version: 1.0
license: MIT
description: A pipeline for comprehensive C++ and Golang projects code review
"""

import os
import sys
from typing import List, Union, Generator, Iterator, Optional, Dict, AnyStr

import qdrant_client
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant.utils import fastembed_sparse_encoder

from libs.tools import (
    Observer,
    NameRetriever,
    NodeRetriever,
    FilterCodeCollectionRetriever,
    CodeMultiCollectionRetriever,
    SimilarityCutoffPostprocessor,
    CodeMethodsExtractor,
    setup_logger
)
from libs.variables import GlobalRepository, DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL


class Pipeline:
    CODE_QA_PROMPT_STR = (
        "You are a Code Review Assistant helping a backend systems team working on high-performance multimedia streaming, distributed systems, and network software."
        "The team develops services in C++, Go, and Python, often dealing with networking protocols (TCP, UDP, RTMP, HLS, DASH, SS), multimedia formats and codecs (e.g., MPEG-TS, H.264, AAC), and performance-critical components."
        "You assist in reviewing system-level code across multiple repositories."
        "\n"
        "Your Responsibilities:"
        "1. Analyze the provided code changes in context."
        "2. Highlight potential bugs, undefined behavior, memory issues, or bad concurrency adhering to rules stated in 'Rules' point below."
        "3. Recommend improvements in readability, efficiency, and modern best practices (e.g., idiomatic Go, modern C++20) adhering to sorting rules stated in 'Rules' point below."
        "4. Explain reasoning with examples. Reference functions, symbols, or code structures explicitly."
        "\n"
        "Rules:"
        "- Do not guess – if context is missing, state what is unclear."
        "- Do not warn about multithreading/synchronization issues if there are no synchronization primitives used in current method."
        "- Sort issues in the following order: memory safety issues, security issues, logic errors, less important issues (typos, posssible refactors, more idiomatic code) at the end."
        "\n"
        "Formatting:"
        "- Use points a), b), c) and so on when reviewing multiple issues."
        "- Provide concise refactor suggestions where appropriate (adhering to sorting rules)."
        "\n"
        "Context information is below:"
        "{context_str}"
        "\n"
        "Question:"
        "{query_str}"
        "\n"
        "Answer:"
    )

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

        def __init__(self, **data):
            print(f"Initializing Review Assistant Valves with data: {data}")
            super().__init__(**data)

    def __init__(self):
        self.name = "Review Assistant"

        self.llm = None
        self.qdrant = None
        self.embedder = None
        self.text_splitter = None
        self.sparse_text_embedding = None
        self.logger = setup_logger(name="Review Assistant", debug=True)

        self.valves = self.Valves(
            **{
                "OLLAMA_MODEL_BASE_URL": os.getenv(f"OLLAMA_MODEL_BASE_URL", "http://10.255.240.156:11434"),
                "OLLAMA_MODEL_NAME": os.getenv(f"OLLAMA_MODEL_NAME", "qwen3-coder:30b"),
                "OLLAMA_CONTEXT_WINDOW": os.getenv(f"OLLAMA_CONTEXT_WINDOW", "32000"),
                "OLLAMA_REQUEST_TIMEOUT": os.getenv(f"OLLAMA_REQUEST_TIMEOUT", "120"),
                "OLLAMA_TEMPERATURE": os.getenv(f"OLLAMA_TEMPERATURE", "0.5"),
                "OLLAMA_EMBEDDING_BASE_URL": os.getenv(f"OLLAMA_EMBEDDING_BASE_URL", "http://10.255.240.149:11435"),
                "OLLAMA_EMBEDDING_MODEL_NAME": os.getenv(f"OLLAMA_EMBEDDING_MODEL_NAME", "jina/jina-embeddings-v2-base-en:latest"),
                "OLLAMA_CHUNK_SIZE": os.getenv(f"OLLAMA_CHUNK_SIZE", "1024"),
                "OLLAMA_CHUNK_OVERLAP": os.getenv(f"OLLAMA_CHUNK_OVERLAP", "256"),
                "QDRANT_SIMILARITY_TOP_K": os.getenv(f"QDRANT_SIMILARITY_TOP_K", "15"),
                "QDRANT_BASE_URL": os.getenv(f"QDRANT_BASE_URL", "http://10.255.240.18:6333"),
                "QDRANT_COLLECTION_NAME": os.getenv(f"QDRANT_COLLECTION_NAME", "codebase_prod_bm42", ),
                "QDRANT_VECTOR_STORE_PARALLEL": os.getenv(f"QDRANT_VECTOR_STORE_PARALLEL", "4"),
                "QDRANT_HYBRID_SEARCH": os.getenv(f"QDRANT_HYBRID_SEARCH", "False").lower() == "true",
                "QDRANT_SIMILARITY_CUTOFF": os.getenv(f"QDRANT_SIMILARITY_CUTOFF", "0.5"),
                "SPARSE_TEXT_EMBEDDING_MODEL": os.getenv(f"SPARSE_TEXT_EMBEDDING_MODEL", DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL),
            }
        )

    def load_config(self):
        self.logger.debug("Running load_config() method")
        self.logger.info("Loading Review Assistant configuration")

        self.text_splitter = SentenceSplitter(
            chunk_size=self.valves.OLLAMA_CHUNK_SIZE,
            chunk_overlap=int(self.valves.OLLAMA_CHUNK_SIZE * 0.1),  # 10% overlap
        )

        self.embedder = OllamaEmbedding(
            model_name=self.valves.OLLAMA_EMBEDDING_MODEL_NAME,
            base_url=self.valves.OLLAMA_EMBEDDING_BASE_URL,
        )

        self.llm = Ollama(
            model=self.valves.OLLAMA_MODEL_NAME,
            base_url=self.valves.OLLAMA_MODEL_BASE_URL,
            temperature=self.valves.OLLAMA_TEMPERATURE,
            request_timeout=self.valves.OLLAMA_REQUEST_TIMEOUT,
            context_window=self.valves.OLLAMA_CONTEXT_WINDOW,
        )

        self.qdrant = qdrant_client.QdrantClient(url=self.valves.QDRANT_BASE_URL)

        if self.valves.QDRANT_HYBRID_SEARCH:
            self.sparse_text_embedding = GlobalRepository.get_or_create(
                name="fastembed_sparse_encoder",
                factory=fastembed_sparse_encoder,
                model_name=self.valves.SPARSE_TEXT_EMBEDDING_MODEL,
            )
        else:
            self.sparse_text_embedding = fastembed_sparse_encoder(DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL)

    async def on_startup(self):
        self.logger.debug("Running on_startup() method")
        self.logger.info("Loading Review Assistant Pipeline")
        self.load_config()

    async def on_valves_updated(self):
        self.logger.debug("Running on_valves_updated() method")
        self.logger.info("Valves updated")
        self.load_config()

    async def on_shutdown(self):
        self.logger.debug("Running on_shutdown() method")
        self.logger.info("Shutdown")
        if self.qdrant:
            self.qdrant.close()

    async def inlet(self, body: dict, user: dict) -> Dict:
        self.logger.debug("Running inlet() method")
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> Dict:
        self.logger.debug("Running outlet() method")
        try:
            if not body or "messages" not in body:
                return {"messages": [{"content": "Invalid format."}]}
            if body["messages"][-1]["content"] == "Empty Response":
                body["messages"][-1]["content"] = "Sorry, no review comments were generated. Try again."
            return body
        except Exception as e:
            self.logger.error("[outlet] Failed to process body: %s", str(e))
            return {"messages": [{"content": "Internal error."}]}

    def build_dual_retriever(self, commit_collection: AnyStr) -> CodeMultiCollectionRetriever:
        self.logger.debug("Running build_dual_retriever() method")
        return CodeMultiCollectionRetriever(
            retrievers=[
                NameRetriever(
                    name="commit",
                    retriever=FilterCodeCollectionRetriever(
                        client=self.qdrant,
                        collection_name=commit_collection,
                        embed_model=self.embedder,
                        similarity_top_k=self.valves.QDRANT_SIMILARITY_TOP_K,
                        parallel=self.valves.QDRANT_VECTOR_STORE_PARALLEL,
                        sparse_text_embedding=None,
                    )
                ),
                NameRetriever(
                    name="codebase",
                    retriever=FilterCodeCollectionRetriever(
                        client=self.qdrant,
                        collection_name=self.valves.QDRANT_COLLECTION_NAME,
                        embed_model=self.embedder,
                        similarity_top_k=self.valves.QDRANT_SIMILARITY_TOP_K,
                        parallel=self.valves.QDRANT_VECTOR_STORE_PARALLEL,
                        sparse_text_embedding=self.sparse_text_embedding,
                    )
                )
            ],
            node_postprocessors=[
                SimilarityCutoffPostprocessor(
                    similarity_cutoff=self.valves.QDRANT_SIMILARITY_CUTOFF
                )
            ],
            logger=self.logger,
        )

    def pipe(self, user_message: AnyStr, model_id: AnyStr, messages: List[Dict], body: Dict) -> Union[AnyStr, Generator, Iterator]:
        self.logger.debug("Running pipe() method")
        self.logger.info(f"User message: {user_message}")

        observer = Observer()
        event_key = observer.start(name="Code Review Pipeline")
        commit_collection = body.get("env").get("COMMIT_COLLECTION")
        diff = body.get("env").get("RAW_DIFF")
        file_name = body.get("env").get("FILE_NAME")

        self.logger.info(f"Commit collection: {commit_collection}")

        if not commit_collection:
            raise ValueError("Missing COMMIT_COLLECTION in env")
        if not diff:
            raise ValueError("Missing RAW_DIFF in env")

        identifiers, removed_identifiers = CodeMethodsExtractor().llm_extract(diff=diff)
        filtered_identifiers = [i for i in identifiers if i not in removed_identifiers]

        dual_retriever = self.build_dual_retriever(commit_collection)
        nodes = dual_retriever.retrieve_code_multi(
            query=user_message,
            apply_filters=True,
            method_names=filtered_identifiers,
            class_names=filtered_identifiers,
            namespaces=[],
            file_paths=[file_name],
            observer=observer,
        )

        if removed_identifiers:  # here we use it to avoid model being deletion biased
            removed_note = (
                "Note: The following identifiers appeared in removed lines in this commit, "
                "but they may still exist elsewhere in the codebase. Avoid assuming they were deleted completely:\n"
                f"{removed_identifiers}\n"
            )
            user_message = removed_note + user_message

        engine = RetrieverQueryEngine.from_args(
            retriever=NodeRetriever(nodes),
            llm=self.llm,
            text_qa_template=PromptTemplate(self.CODE_QA_PROMPT_STR),
            streaming=body.get("stream", False),
        )

        response = engine.query(user_message)
        observer.stop(key=event_key)

        return response.response + "\n" + observer.summary() + "\n" + f"Identifiers ->  ADDED: {identifiers} ; REMOVED: {removed_identifiers}"
