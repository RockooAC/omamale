"""
title: Qdrant Ollama Pipeline
description: A pipeline for retrieving relevant information from a qdrant collection using the Llama Index library with Ollama embeddings.
"""
import os
from typing import List, Optional
from typing import Union, Generator, Iterator

import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.indices import EmptyIndex
from llama_index.core.indices.empty import EmptyIndexRetriever
from llama_index.core.indices.list.base import ListRetrieverMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.schema import IndexNode, NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.qdrant.utils import fastembed_sparse_encoder
from pydantic import Field

from libs.tools import (Observer, SimilarityCutoffPostprocessor, BuildReference,
                        NameRetriever, MultiCollectionRetriever, setup_logger, MessagesRetriever, ExpandQueries,
                        TokenCounterPostprocessor)
from libs.variables import (
    GlobalRepository,
    DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_SENTENCE_TRANSFORMER_RERANK_MODEL,
)

# Logger Configuration
logger = setup_logger(name="Assistant Redge", debug=False)

# Prompt template for question answering
qa_prompt_str = (
    "You are an advanced language model and personal assistant to a system software team focused on multimedia streaming and networking. "
    "The team is developing products like Redge Media Coder and Redge Media Origin/CDN (packager) and deals with topics such as codecs, containers, streaming protocols (HLS, DASH, SS), "
    "networking (UDP, TCP/IP, RTMP, MPEG-TS, multicast), server performance, and video/audio codecs (e.g., H.264, H.265, AAC, MPEG-2), as well as NVIDIA technologies like CUDA and NVENC.\n"
    "\n"
    "Your Task: Analyze the embedded documents and provide precise, detailed, and context-based answers to user questions. "
    "Your answers must be accurate, comprehensive, and focused on clarity. "
    "Aim for both depth and readability, connecting high-level facts when relevant.\n"
    "\n"
    "Requirements for Each Response:\n"
    "Answer Precision: Respond accurately and thoroughly based on the provided context. Avoid improvising; if the answer is not within the context, say: \"Sorry, the provided query is not clear enough for me to answer from the provided research papers.\"\n"
    "Detail-Oriented: Include detailed explanations and provide additional relevant information when appropriate, even if loosely related. Prioritize information from newer versions if conflicting details exist.\n"
    "Conciseness and Clarity: Be concise yet comprehensive. Make your answers clear and easy to understand for researchers.\n"
    "\n"
    "Context information is below:\n"
    "{context_str}\n"
    "Question: {query_str}\n"
    "\n"
    "Answer:"
)

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

DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(
    DEFAULT_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)

ENV_SUFFIX = "ASSISTANT_REDGE"

class Pipeline:

    class Valves(BaseModel):
        # Ollama main settings
        OLLAMA_MODEL_BASE_URL: str = Field(default="http://10.255.240.156:11434", description="Ollama base URL")
        OLLAMA_MODEL_NAME: str = Field(default="qwen3:30b", description="Ollama base model name")
        OLLAMA_CONTEXT_WINDOW: int = Field(default="61440", description="Ollama context window size")
        OLLAMA_TEMPERATURE: float = Field(default="0.5", description="Ollama temperature")
        # Ollama embedding settings
        OLLAMA_EMBEDDING_BASE_URL: str = Field(default="http://10.255.240.149:11434", description="Ollama embedding base URL")
        OLLAMA_EMBEDDING_MODEL_NAME: str = Field(default="gte-qwen2.5-instruct-q5", description="Ollama embedding model name")
        # Ollama chunk settings
        OLLAMA_CHUNK_SIZE: int = Field(default="1024", description="Ollama chunk size")
        OLLAMA_CHUNK_OVERLAP: int = Field(default="128", description="Ollama chunk overlap")
        CONTEXT_INPUT_OUTPUT_RATIO: str = Field(default="1:1", description="Context input output ratio")
        # Ollama queries settings
        OLLAMA_QUERIES_ACTIVE: bool = Field(default=True, description="Ollama queries active")
        OLLAMA_QUERIES_BASE_URL: str = Field(default="http://10.255.240.161:11434", description="Ollama queries base URL")
        OLLAMA_QUERIES_MODEL_NAME: str = Field(default="mistral", description="Ollama queries model name")
        OLLAMA_QUERIES_STRATEGY: str = Field(default='diversify', description="Ollama queries strategy", examples=['diversify', 'specify', 'broaden', 'rephrase', 'diagnose'])
        OLLAMA_QUERIES_SIMILARITY_THRESHOLD: float = Field(default=0.8, description="Ollama queries similarity threshold")
        OLLAMA_QUERIES_TEMPERATURE: float = Field(default=0.5, description="Ollama queries temperature")
        OLLAMA_QUERIES_NUM: int = Field(default=3, description="Ollama queries num")
        OLLAMA_QUERIES_CONTEXT_WINDOW: int = Field(default=8192, description="Ollama queries context window")
        OLLAMA_QUERIES_MIN_QUERY_LENGTH: int = Field(default=32, description="Ollama queries min query length")
        OLLAMA_QUERIES_MAX_QUERY_LENGTH: int = Field(default=256, description="Ollama queries max query length")
        # QDRANT settings
        QDRANT_BASE_URL: str = Field(default="http://10.255.240.18:6333", description="QDRANT base URL")
        QDRANT_COLLECTION_NAME: str = Field(default="confluence_prod_bm42,docusaurus_prod_bm42,pdf_prod_bm42", description="QDRANT collection name")
        QDRANT_VECTOR_STORE_PARALLEL: int = Field(default=4, description="QDRANT vector store parallel")
        QDRANT_SIMILARITY_TOP_K: int = Field(default=20, description="QDRANT similarity top k")
        QDRANT_HYBRID_SEARCH: bool = Field(default=True, description="QDRANT hybrid search")
        SPARSE_TEXT_EMBEDDING_MODEL: str = Field(default=DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL, description="Model name for sparse text embedding")
        # RERANK SETTINGS
        # Similarity cutoff
        QDRANT_SIMILARITY_CUTOFF_ACTIVE: bool = Field(default=True, description="QDRANT similarity cutoff active")
        QDRANT_SIMILARITY_CUTOFF: float = Field(default=0.3, description="QDRANT similarity cutoff")
        # Sentence Transformer Rerank
        SENTENCE_TRANSFORMER_RERANK_ACTIVE: bool = Field(default=True, description="SENTENCE_TRANSFORMER_RERANK_ACTIVE")
        SENTENCE_TRANSFORMER_RERANK_MODEL: str = Field(default=DEFAULT_SENTENCE_TRANSFORMER_RERANK_MODEL, description="Model name for Sentence Transformer Rerank")
        SENTENCE_TRANSFORMER_RERANK_TOP_N: int = Field(default=20, description="Sentence Transformer Rerank top n")
        # Ollama rerank settings
        OLLAMA_RERANK_ACTIVE: bool = Field(default=False, description="Ollama rerank active")
        OLLAMA_RERANK_BASE_URL: str = Field(default="http://10.255.240.161:11434", description="Ollama rerank base URL")
        OLLAMA_RERANK_MODEL_NAME: str = Field(default="mistral", description="Ollama rerank model name")
        OLLAMA_RERANK_TEMPERATURE: float = Field(default="0.5", description="Ollama rerank temperature")
        OLLAMA_RERANK_TOP_N: int = Field(default=15, description="Ollama rerank top n")
        OLLAMA_RERANK_CHOICE_BATCH_SIZE: int = Field(default="5", description="Ollama rerank choice batch size")
        # Other settings
        PIPELINE_TIME_RECORDING: bool = Field(default=True, description="Enable pipeline time recording")
        CHAT_HISTORY_ACTIVE: bool = Field(default=True, description="Enable chat history")
        FORCE_ONLY_EXTERNAL_SOURCES: bool = Field(default=False, description="Enable force only external sources. If enabled, only external sources are used for the answer.")

        def __init__(self, **data):
            logger.info(f"Initializing Valves with data: {data}")
            super().__init__(**data)

    def __init__(self):
        logger.info("Initializing Assistant Redge")
        self.name = "Assistant Redge"
        self.query_engine = None
        self.qdrant = None
        self.ollama_llm = None
        self.ollama_rerank = None
        self.expand_queries = None
        self.ollama_embedder = None
        self.text_splitter = None
        self.retrievers = []
        self.llm_rerank = None
        self.query_engine_postprocessors = []
        self.sparse_text_embedding = None

        # Initialize valve parameters from environment variables
        self.valves = self.Valves(
            **{k: os.getenv(f"{k}_{ENV_SUFFIX}", v.default) for k, v in self.Valves.model_fields.items()}
        )

    def load_config(self):
        """
        Load the configuration. This function is called when the server is started and when the valves are updated.
        It sets various settings for the Ollama model and Qdrant client, and initializes the vector store index.
        """

        logger.info("Loading configuration")
        try:
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

            # Set the LLM settings with model name, base URL, temperature, and request timeout
            self.ollama_llm = Ollama(
                model=self.valves.OLLAMA_MODEL_NAME,
                base_url=self.valves.OLLAMA_MODEL_BASE_URL,
                temperature=self.valves.OLLAMA_TEMPERATURE,
                context_window=self.valves.OLLAMA_CONTEXT_WINDOW,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
            )

            self.ollama_rerank = Ollama(
                base_url=self.valves.OLLAMA_RERANK_BASE_URL,
                model=self.valves.OLLAMA_RERANK_MODEL_NAME,
                temperature=self.valves.OLLAMA_RERANK_TEMPERATURE,
                context_window=self.valves.OLLAMA_RERANK_CHOICE_BATCH_SIZE * self.valves.OLLAMA_CHUNK_SIZE + self.valves.OLLAMA_CHUNK_OVERLAP,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
            )

            if self.valves.OLLAMA_QUERIES_ACTIVE:
                self.expand_queries = ExpandQueries(
                    llm=Ollama(
                        base_url=self.valves.OLLAMA_QUERIES_BASE_URL,
                        model=self.valves.OLLAMA_QUERIES_MODEL_NAME,
                        temperature=self.valves.OLLAMA_QUERIES_TEMPERATURE,
                        context_window=self.valves.OLLAMA_QUERIES_CONTEXT_WINDOW,
                        request_timeout=DEFAULT_REQUEST_TIMEOUT,
                    ),
                    num_expansions=self.valves.OLLAMA_QUERIES_NUM,
                    similarity_threshold=self.valves.OLLAMA_QUERIES_SIMILARITY_THRESHOLD,
                    expansion_strategy=self.valves.OLLAMA_QUERIES_STRATEGY,   #Strategy for query expansion: 'diversify', 'specify', 'broaden', 'rephrase', 'diagnose'
                    min_query_length=self.valves.OLLAMA_QUERIES_MIN_QUERY_LENGTH,
                    max_query_length=self.valves.OLLAMA_QUERIES_MAX_QUERY_LENGTH,
                )

            # Connect to the Qdrant client using the base URL
            self.qdrant = qdrant_client.QdrantClient(url=self.valves.QDRANT_BASE_URL)
            logger.info(f"Initialized rerank client with URL: {self.valves.QDRANT_BASE_URL}")
            # Create a reranker using the Ollama model

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

            # Add the token counter postprocessor to the query engine
            # This postprocesor protects against context overflow
            self.query_engine_postprocessors.append(
                TokenCounterPostprocessor(
                    context_size=self.valves.OLLAMA_CONTEXT_WINDOW,
                    input_output_ratio=self.valves.CONTEXT_INPUT_OUTPUT_RATIO,
                ),
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

            self.retrievers = []
            collections = self.valves.QDRANT_COLLECTION_NAME.split(",")
            for collection_name in collections:
                collection_retriever = self._build_retriever(collection_name)
                if collection_retriever is not None:
                    logger.info(f"Loaded collection: {collection_name}")
                    self.retrievers.append(NameRetriever(name=collection_name, retriever=collection_retriever))
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def _build_retriever(self, collection_name: str) -> BaseRetriever:
        """
        Build a retriever for the specified collection name.
        Args:
            collection_name (str): The name of the collection.
        Returns:
            IndexNode: The index node for the collection.
        """
        if not self.qdrant.collection_exists(collection_name):
            logger.info(f"Collection {collection_name} does not exist.")
            return EmptyIndexRetriever(index=EmptyIndex())

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
        logger.info("Valves updated. Reloading configuration.")
        self.load_config()

    async def on_shutdown(self):
        """
        This function is called when the server is stopped.
        Currently, it does not perform any actions.
        """
        logger.info("Shutting down Assistant Redge Pipeline")
        self.qdrant.close()

    async def inlet(self, body: dict, user: dict) -> dict:
        """Modifies form data before the OpenAI API request."""
        logger.info("Processing inlet request")
        if 'files' in body:
            for file in body['files']:
                if 'collection_name' in file:
                    if 'external_collections' not in body:
                        body['external_collections'] = []
                    body['external_collections'].append(f"open-webui_{file['collection_name']}")
        elif 'metadata' in body and body['metadata'].get('files', None) is not None:
            for file in body['metadata']['files']:
                if 'collection_name' in file:
                    if 'external_collections' not in body:
                        body['external_collections'] = []
                    body['external_collections'].append(f"open-webui_{file['collection_name']}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Modifies the response body before sending it to the user."""
        # Check if the last message in body is an 'Empty Response' and replace it with a 'No Results Found' message
        if body.get("messages"):
            # "Empty Response"
            content = body["messages"][-1].get("content", "")
            if content.startswith("Empty Response"):
                content = content.replace("Empty Response", "Sorry, I couldn't find any relevant information. Please try asking in a different way.")
                body["messages"][-1]["content"] = content
        return body

    @staticmethod
    def _prepare_references(source_nodes: list[NodeWithScore]) -> str:
        """Prepare the references for the response."""
        references_info = f"\n\n***References:***\n"
        # Add unique references to the response
        references = set()
        # Sort nodes by score in place
        source_nodes.sort(key=lambda x: x.score, reverse=True)
        for node in source_nodes:
            if 'reference' in node.metadata and node.metadata['reference'] not in references:
                references.add(node.metadata['reference'])
                references_info += f"\n- {node.metadata['reference']}  (Score: {node.score:.2f})"
        return references_info

    def _extend_stream(self, response, event_key: str, observer: Observer) -> Generator:
        """
        Extend the response stream by yielding chunks from the response generator,
        stopping the observer, and appending references and summary.

        Args:
            response: The response object containing the response generator.
            event_key (str): The event key for the observer.
            observer (Observer): The observer instance to track the processing.

        Yields:
            Generator: Chunks of the response, followed by references and summary.
        """
        generator = response.response_gen() if callable(response.response_gen) else response.response_gen
        for chunk in generator:
            yield chunk
        observer.stop(key=event_key)
        logger.info(f"Finished processing response with event key: {event_key}")
        yield self._prepare_references(response.source_nodes)
        yield observer.summary()

    def _process_response(self, response, event_key, observer) -> Union[str, Generator, Iterator]:
        """
        Process the response from the query engine.

        Args:
            response: The response object from the query engine.
            event_key: The event key for the observer.
            observer: The observer instance to track the processing.

        Returns:
            Union[str, Generator, Iterator]: The processed response, which can be a string, generator, or iterator.
        """
        if isinstance(response, StreamingResponse):
            return self._extend_stream(response, event_key, observer)
        else:
            content = response.response
            observer.stop(key=event_key)
            content += self._prepare_references(response.source_nodes)
            content += observer.summary()
            logger.info(f"Finished processing response with event key: {event_key}")
            return content

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) \
            -> Union[str, Generator, Iterator]:
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
        logger.info(f"Processing message: '{user_message}'")
        retrievers = self.retrievers.copy()
        if 'external_collections' in body:
            for external_collection in body['external_collections']:
                collection_retriever = self._build_retriever(external_collection)
                if collection_retriever:
                    if self.valves.FORCE_ONLY_EXTERNAL_SOURCES:
                        retrievers.clear()
                    retrievers.append(NameRetriever(name=external_collection, retriever=collection_retriever))

        # Add messages to the retrievers if chat history is active
        if len(messages) > 1 and self.valves.CHAT_HISTORY_ACTIVE:
            retrievers.append(MessagesRetriever(messages=messages))

        observer = Observer()
        query_engine = RetrieverQueryEngine.from_args(
            retriever=MultiCollectionRetriever(
                retrievers=retrievers,
                observer=observer,
                expand_queries=self.expand_queries,
                node_postprocessors=self.query_engine_postprocessors,
            ),
            llm=self.ollama_llm,
            text_qa_template=PromptTemplate(qa_prompt_str),
            streaming=body.get('stream', False),
        )

        event_key = observer.start(name="LLM Query Engine")
        logger.info(f"Send message: '{user_message}' to query engine with event key: {event_key}")

        response = query_engine.query(user_message)

        return self._process_response(response, event_key, observer)