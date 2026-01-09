import hashlib
import json
import logging
import os
import re
import textwrap
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from itertools import accumulate, takewhile
from operator import sub
from pathlib import Path
from typing import List, Optional, cast, Dict, Any, Callable, Tuple
from uuid import uuid4

import mistune
import requests
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.list.base import ListRetrieverMode
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType, TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.legacy.postprocessor import BaseNodePostprocessor
from llama_index.legacy.vector_stores.qdrant_utils import SparseEncoderCallable
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pygments.lexers import guess_lexer
from pygments.token import Token
from pygments.util import ClassNotFound
from qdrant_client import QdrantClient

try:
    from libs.template import EXTRACTOR_PROMPT_TEMPLATE, CODE_DETECTOR_TEMPLATE, CODE_METHODS_EXTRACTOR_TEMPLATE, \
        QUERIES_TEMPLATES, DEFAULT_QUERY_TEMPLATE
except ImportError:
    from template import EXTRACTOR_PROMPT_TEMPLATE, CODE_DETECTOR_TEMPLATE, CODE_METHODS_EXTRACTOR_TEMPLATE


#################################
# Helper classes
#################################


class NodeRetriever(BaseRetriever):
    def __init__(self, nodes):
        self.nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle):
        return self.nodes


class Event(BaseModel):
    """Event class for the retriever."""

    data: Dict[str, Any] = None
    name: str = ""
    start: datetime = datetime.now()
    end: datetime = None

    def __init__(self, name: str, **data):
        super().__init__(**data)
        self.data = data
        self.name = name
        self.start = datetime.now()

    def __str__(self):
        if self.end is None:
            return f"{self.name}, Not completed."
        return f"{self.name}: {(self.end - self.start).total_seconds():.3f} seconds."


class Observer(BaseModel):
    """Observer class for the retriever."""

    events: Dict[str, Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.events = {}

    def start(self, key: Optional[str] = None, name: Optional[str] = None, **event_data) -> str:
        if not key:
            key = str(uuid4())
        event = Event(name=name, **event_data)
        self.events[key] = event
        return key

    def stop(self, key: str, **event_data) -> str:
        if key and key in self.events:
            event = self.events[key]
            event.end = datetime.now()
            event.data.update(event_data)
            return key
        else:
            raise ValueError(f"Event with key {key} not found.")

    @contextmanager
    def measure(self, name, **event_data):
        key = self.start(name=name, **event_data)
        try:
            yield
        finally:
            self.stop(key)

    def summary(self) -> str:
        summary = "\n".join([f"- {str(event)}" for key, event in self.events.items()])
        return f"\n\n---\n***Times:***\n{summary}"

    def clear(self):
        self.events = {}


class TokenCounterPostprocessor(BaseNodePostprocessor):
    """Node processor for filtering chunks based on context limit."""

    context_size: int = Field(...)
    input_output_ratio: str | Tuple[int, int] = Field(default="1:1")
    logger: Optional[logging.Logger] = Field(default=None, exclude=True)

    def _parse_ratio(self, ratio: str | Tuple[int, int]) -> Tuple[int, int]:
        if isinstance(ratio, str):
            parts = ratio.split(":")
            if len(parts) != 2:
                raise ValueError("Ratio must be in format 'input:output' ('2:1', '1:1', ...)")
            return int(parts[0]), int(parts[1])
        return ratio

    def _get_input_max_tokens(self, context_size: int, input_output_ratio: Tuple[int, int]) -> int:
        first, second = input_output_ratio
        return int(context_size * first / (first + second))

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        nodes = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
        ratio_tuple = self._parse_ratio(self.input_output_ratio)
        max_input_tokens = self._get_input_max_tokens(self.context_size, ratio_tuple)
        tokens_left = max_input_tokens

        logging.info(f"Tokens left: {tokens_left}")  # FIXME -> Have to fix logger initialization -> 'FieldInfo' object has no attribute 'info'

        nodes_tokens_count = (node.metadata.get("token_count", 0) for node in nodes)
        tokens_left_iter = accumulate(nodes_tokens_count, sub, initial=tokens_left)
        tokens_left_iter = takewhile(lambda tokens_left: tokens_left > 0, tokens_left_iter)
        
        return list(node for node, _ in zip(nodes, tokens_left_iter))


class SimilarityCutoffPostprocessor(BaseNodePostprocessor):
    """Similarity-based Node processor with max chunks limit."""

    # Similarity cutoff for filtering nodes
    similarity_cutoff: Optional[float] = None
    # Maximum number of chunks to return, ordered by score
    max_chunks: Optional[int] = None

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        sim_cutoff_exists = self.similarity_cutoff is not None
        max_chunks_exists = self.max_chunks is not None

        filtered_nodes = []
        for node in nodes:
            should_use_node = True
            if sim_cutoff_exists:
                similarity = node.score
                if similarity is None:
                    should_use_node = False
                elif cast(float, similarity) < cast(float, self.similarity_cutoff):
                    should_use_node = False

            if should_use_node:
                filtered_nodes.append(node)

        if max_chunks_exists and filtered_nodes:
            sorted_nodes = sorted(
                filtered_nodes, key=lambda x: float("-inf") if x.score is None else cast(float, x.score), reverse=True
            )
            return sorted_nodes[: self.max_chunks]

        return filtered_nodes


class BuildReference(BaseNodePostprocessor):
    reference_key: str = Field(description="Reference key in metadata.")

    def __init__(self, reference_key: str) -> None:
        super().__init__()
        self.reference_key = reference_key

    @classmethod
    def class_name(cls) -> str:
        return "Citations"

    @staticmethod
    def _pdf_reference(metadata: Dict[str, Any]) -> str:
        """Get the PDF references from metadata."""
        reference = []
        if "file_path" in metadata:
            file_path = metadata["file_path"]
            # Remove everything before the last 'chat-ai-embedding-sources/' in the file path
            reference.append(f"File path: '{file_path.split('chat-ai-embedding-sources/')[-1]}'")
        if "section_name" in metadata and metadata["section_name"]:
            section_name = "Paragraph: '"
            if "section_num" in metadata and metadata["section_num"]:
                section_name += f"{metadata['section_num']} - "
            section_name += metadata["section_name"]
            section_name += "'"
            reference.append(section_name)
        if "sentence_idx" in metadata:
            reference.append(f"Sentence: {metadata['sentence_idx']}")

        return ", ".join(reference)

    @staticmethod
    def _html_reference(metadata: Dict[str, Any]) -> str:
        """Get the HTML references from metadata."""
        if "title" in metadata and "url" in metadata:
            return f"Title: [{metadata['title']}]({metadata['url']})"
        elif "title" in metadata and not "url" in metadata:
            return f"Title: '{metadata['title']}'"
        elif not "title" in metadata and "url" in metadata:
            return f"URL: '{metadata['url']}'"
        else:
            return "No reference available"

    @staticmethod
    def _docusaurus_reference(metadata: Dict[str, Any]) -> str:
        """Get the HTML references from metadata."""
        if "file_path" in metadata:
            file_path = metadata["file_path"]
            file_path = file_path.split("/rg/doc/docusaurus/redgemedia/")[-1]
            file_path = os.path.dirname(file_path)
            return f"URL: https://docs.redge.media/{file_path}"

        return ""

    @staticmethod
    def _text_reference(metadata: Dict[str, Any]) -> str:
        """Get the HTML references from metadata."""
        if "file_path" in metadata:
            file_path = metadata["file_path"]
            file_path = file_path.split("chat-ai-embedding-sources/")[-1]
            return f"File path: {file_path}"

        return ""

    @staticmethod
    def _websearch_reference(metadata: Dict[str, Any]) -> str:
        """Get the websearch references from metadata."""
        reference = []
        if "title" in metadata:
            reference.append(f"Title: '{metadata['title']}'")
        if "source" in metadata:
            reference.append(f"Source: '{metadata['source']}'")
        return ", ".join(reference)

    def postprocess_node(self, node: NodeWithScore):
        try:
            if "file_type" in node.metadata and node.metadata["file_type"] == "application/pdf":
                node.metadata[self.reference_key] = self._pdf_reference(node.metadata)
            elif "file_type" in node.metadata and node.metadata["file_type"] == "text/plain":
                node.metadata[self.reference_key] = self._text_reference(node.metadata)
            elif "url" in node.metadata:
                node.metadata[self.reference_key] = self._html_reference(node.metadata)
            elif "file_path" in node.metadata and "/rg/doc/docusaurus/redgemedia/" in node.metadata["file_path"]:
                node.metadata[self.reference_key] = self._docusaurus_reference(node.metadata)
            elif "reference" in node.metadata:
                node.metadata[self.reference_key] = node.metadata["reference"]
            elif "metadata" in node.metadata:
                # Copy the metadata from the 'metadata' key to the node metadata
                node.metadata[self.reference_key] = self._websearch_reference(node.metadata["metadata"])
            else:
                node.metadata[self.reference_key] = "No reference available"
        except Exception as e:
            node.metadata[self.reference_key] = "N/A"
            logging.error(f"Error in BuildReference: {e}")

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        for node in nodes:
            self.postprocess_node(node)

        return nodes


class NameRetriever(BaseRetriever):
    """A retriever that returns a single node with a name."""

    name: str = Field(description="The name of the node.")
    retriever: BaseRetriever = Field(description="The retriever to use.")

    def __init__(self, name: str, retriever: BaseRetriever, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if name is None:
            raise ValueError("Name cannot be None.")
        self.name = name
        if retriever is None:
            raise ValueError("Retriever cannot be None.")
        self.retriever = retriever

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes."""
        return self.retriever.retrieve(query_bundle)

    def __str__(self):
        return self.name


class ExpandQueries(BaseModel):
    """
    Enhanced class for expanding queries using LLM with improved functionality.

    Features:
    - Customizable prompt templates
    - Multiple expansion strategies
    - Query validation and quality control
    - Deduplication mechanisms
    - Error handling and fallback strategies
    - Caching for performance
    - Flexible parsing
    """

    llm: Any = Field(description="LLM to use for query expansion.")
    num_expansions: int = Field(default=3, description="Number of query expansions to generate.")

    # New configuration options
    expansion_strategy: str = Field(default=DEFAULT_QUERY_TEMPLATE, description="Strategy for query expansion: 'diversify', 'specify', 'broaden', 'rephrase', 'diagnose'")
    min_query_length: int = Field(default=30, description="Minimum length for generated queries.")
    max_query_length: int = Field(default=200, description="Maximum length for generated queries.")
    similarity_threshold: float = Field(default=0.8, description="Threshold for deduplicating similar queries.")
    fallback_strategies: List[str] = Field(default_factory=lambda: ["synonyms", "keywords"], description="Fallback strategies when LLM fails.")

    def __init__(self, **data):
        super().__init__(**data)

    def _get_prompt_template(self, strategy: str) -> str:
        """Get prompt template based on expansion strategy."""
        return QUERIES_TEMPLATES.get(strategy, QUERIES_TEMPLATES[DEFAULT_QUERY_TEMPLATE])

    def _parse_llm_response(self, response_text: str) -> List[str]:
        """Enhanced parsing of LLM response with multiple fallback strategies."""
        if not response_text or not response_text.strip():
            return []

        text = response_text.strip()
        # Remove <think>...</think> sections (including multiline, case-insensitive)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        questions = []

        # Strategy 1: Split by newlines and clean bullet points
        lines = text.split('\n')
        for line in lines:
            cleaned = line.strip()
            # Remove various bullet point formats
            cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
            cleaned = re.sub(r'^\d+\.\s*', '', cleaned)  # Remove numbered lists
            cleaned = cleaned.strip('"\'')  # Remove quotes

            if cleaned and len(cleaned) >= self.min_query_length:
                questions.append(cleaned)

        # Strategy 2: If no questions found, try splitting by common separators
        if not questions:
            separators = ['. ', '? ', '! ', '; ']
            for sep in separators:
                if sep in text:
                    parts = text.split(sep)
                    for part in parts:
                        cleaned = part.strip().strip('"\'')
                        if cleaned and len(cleaned) >= self.min_query_length:
                            # Add question mark if missing
                            if not cleaned.endswith('?'):
                                cleaned += '?'
                            questions.append(cleaned)
                    break

        # Strategy 3: If still no questions, treat entire response as single query
        if not questions and len(text) >= self.min_query_length:
            questions.append(text)

        return questions

    def _validate_query(self, query: str) -> bool:
        """Validate generated query quality."""
        if not query or len(query.strip()) < self.min_query_length:
            return False

        if len(query) > self.max_query_length:
            return False

        # Check for reasonable content (not just punctuation or numbers)
        if not re.search(r'[a-zA-Z]', query):
            return False

        # Check for minimum word count
        words = query.split()
        if len(words) < 2:
            return False

        return True

    def _deduplicate_queries(self, queries: List[str], original_query: str) -> List[str]:
        """Remove duplicate and very similar queries."""
        if not queries:
            return []

        unique_queries = []
        original_lower = original_query.lower()

        for query in queries:
            query_lower = query.lower()

            # Skip if identical to original
            if query_lower == original_lower:
                continue

            # Skip if too similar to original (simple word overlap check)
            original_words = set(original_lower.split())
            query_words = set(query_lower.split())

            if original_words and query_words:
                overlap = len(original_words.intersection(query_words))
                similarity = overlap / max(len(original_words), len(query_words))

                if similarity > self.similarity_threshold:
                    continue

            # Check similarity with already selected queries
            is_duplicate = False
            for existing in unique_queries:
                existing_words = set(existing.lower().split())
                if query_words and existing_words:
                    overlap = len(query_words.intersection(existing_words))
                    similarity = overlap / max(len(query_words), len(existing_words))

                    if similarity > self.similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_queries.append(query)

        return unique_queries

    def _apply_fallback_strategies(self, original_query: str) -> List[str]:
        """Apply fallback strategies when LLM fails."""
        fallback_queries = []

        for strategy in self.fallback_strategies:
            if strategy == "synonyms":
                # Simple synonym-based expansion
                words = original_query.split()
                if len(words) > 1:
                    # Create variations by replacing key terms
                    synonyms = {
                        "how": ["what", "which way"],
                        "what": ["how", "which"],
                        "why": ["what reason", "how come"],
                        "when": ["at what time", "during which"],
                        "where": ["in which location", "at what place"]
                    }

                    for word, syns in synonyms.items():
                        if word in original_query.lower():
                            for syn in syns:
                                new_query = original_query.lower().replace(word, syn, 1)
                                fallback_queries.append(new_query.capitalize())

            elif strategy == "keywords":
                # Extract key terms and create focused queries
                words = original_query.split()
                if len(words) >= 3:
                    # Create queries focusing on different parts
                    mid_point = len(words) // 2
                    first_half = " ".join(words[:mid_point])
                    second_half = " ".join(words[mid_point:])

                    if len(first_half) >= self.min_query_length:
                        fallback_queries.append(f"What about {first_half}?")
                    if len(second_half) >= self.min_query_length:
                        fallback_queries.append(f"How does {second_half} work?")

        return fallback_queries[:self.num_expansions]

    def _expand(self, query: QueryBundle) -> List[QueryBundle]:
        """Generate expanded queries with enhanced functionality."""
        query_str = query.query_str

        try:
            # Get appropriate prompt template
            template = self._get_prompt_template(self.expansion_strategy)
            prompt = template.format(query=query_str, num_expansions=self.num_expansions)

            # Generate response
            response = self.llm.complete(prompt)

            # Parse response
            questions = self._parse_llm_response(response.text)

            # Validate queries
            valid_questions = [q for q in questions if self._validate_query(q)]

            # Deduplicate
            unique_questions = self._deduplicate_queries(valid_questions, query_str)

            # If we don't have enough questions, apply fallback strategies
            if len(unique_questions) < self.num_expansions:
                fallback_questions = self._apply_fallback_strategies(query_str)
                # Add fallback questions that aren't duplicates
                for fq in fallback_questions:
                    if fq not in unique_questions and len(unique_questions) < self.num_expansions:
                        unique_questions.append(fq)

            # Limit to requested number
            final_questions = unique_questions[:self.num_expansions]

            # Convert to QueryBundle objects
            result = [QueryBundle(query_str=q) for q in final_questions]

            return result

        except Exception as e:
            logging.warning(f"Error in query expansion: {e}. Using fallback strategies.")
            # Use fallback strategies
            fallback_questions = self._apply_fallback_strategies(query_str)
            return [QueryBundle(query_str=q) for q in fallback_questions]

    def expand(self, query: QueryBundle, observer: Optional[Observer] = None) -> List[QueryBundle]:
        """Generate expanded queries with optional observability."""
        if not observer:
            return self._expand(query)

        with observer.measure(name="Generating expanded queries",
                              query=query.query_str,
                              strategy=self.expansion_strategy,
                              num_expansions=self.num_expansions):
            return self._expand(query)

class MultiCollectionRetriever(BaseRetriever):
    """A retriever that combines multiple retrievers."""

    retrievers: List[NameRetriever] = Field(description="A list of retrievers to combine.")
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = Field(description="A list of node postprocessors.")

    def __init__(
        self,
        retrievers: List[NameRetriever],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        observer: Optional[Observer] = None,
        expand_queries: Optional[ExpandQueries] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:

        if retrievers is None:
            raise ValueError("Retrievers cannot be None.")

        self.retrievers = retrievers or []
        self.node_postprocessors = node_postprocessors or []
        self.observer = observer
        self.expand_queries = expand_queries
        super().__init__(callback_manager=callback_manager, object_map=object_map, verbose=verbose, **kwargs)

    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        return super().retrieve(str_or_query_bundle)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes."""
        results = []
        queries = [query_bundle]
        seen_ids = set()
        # If expand_queries is provided, generate expanded queries
        if self.expand_queries:
            expanded_queries = self.expand_queries.expand(query=query_bundle, observer=self.observer)
            logging.info("Expanded queries generated:\n" + "\n".join(q.query_str for q in expanded_queries))
            queries.extend(expanded_queries)
        # Retrieve nodes from each retriever
        for retriever in self.retrievers:
            event_key = self.observer.start(name=f"Retrieving from '{retriever.name}'")
            for query in queries:
                nodes = retriever.retrieve(query)
                for node in nodes:
                    if node.node.node_id not in seen_ids:
                        results.append(node)
                        seen_ids.add(node.node.node_id)
            self.observer.stop(key=event_key)
        # Postprocess nodes
        for postprocessor in self.node_postprocessors:
            event_key = self.observer.start(name=f"Postprocessing with '{postprocessor.class_name()}'")
            results = postprocessor.postprocess_nodes(results, query_bundle=query_bundle)
            self.observer.stop(event_key)
        return results


class CodeMultiCollectionRetriever(MultiCollectionRetriever):
    """Multi retriever for code collections"""

    def __init__(
            self,
            retrievers: List[NameRetriever],
            node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
            observer: Optional[Observer] = None,
            logger: logging.Logger = None,
            **kwargs,
    ):
        self.logger = logger or setup_logger("CodeMultiCollectionRetriever", debug=False)
        super().__init__(retrievers=retrievers, node_postprocessors=node_postprocessors, observer=observer, **kwargs)

    def retrieve_code_multi(
        self,
        query: str,
        apply_filters: Optional[bool] = False,
        class_names: Optional[List[str]] = None,
        method_names: Optional[List[str]] = None,
        namespaces: Optional[List[str]] = None,
        file_paths: Optional[List[str]] = None,
        search_context: Optional[str] = None,
        observer: Optional[Observer] = None,
    ) -> List[NodeWithScore]:
        """
        Function that collect and returns chunks (NodeWithScore).
        It uses retrieve_code() function since it iterates through data entities.
        Example:
            {
                "class_names": "X265Encoder", "AudioEncoder", ...
            }
        """
        class_names = class_names or []
        method_names = method_names or []
        namespaces = namespaces or []
        file_paths = file_paths or []

        all_results = []
        search_count = 0

        def search_with_entity(class_name=None, mth_name=None, ns=None, fp=None):
            nonlocal search_count
            search_count += 1

            results = self.retrieve_code(
                query=query,
                apply_filters=apply_filters,
                class_name=class_name,
                method_name=mth_name,
                namespace=ns,
                file_path=fp,
                search_context=search_context,
                observer=observer,
            )
            return results

        for class_name in class_names:
            all_results.extend(search_with_entity(class_name=class_name))
        for mth_name in method_names:
            all_results.extend(search_with_entity(mth_name=mth_name))
        for ns in namespaces:
            all_results.extend(search_with_entity(ns=ns))
        for fp in file_paths:
            all_results.extend(search_with_entity(fp=fp))

        if not (class_names or method_names or namespaces or file_paths):
            all_results = search_with_entity()

        return all_results

    def retrieve_code(
        self,
        query: str,
        apply_filters: Optional[bool] = False,
        class_name: Optional[str] = None,
        method_name: Optional[str] = None,
        namespace: Optional[str] = None,
        file_path: Optional[str] = None,
        search_context: Optional[str] = None,
        observer: Optional[Observer] = None,
    ) -> List[NodeWithScore]:
        """Function that collect and returns chunks (NodeWithScore) based on structural filters (FilterCodeCollectionRetriever)"""

        # Prioritize search_context over query (which is postprocessed by LLM)
        search_query = search_context if search_context else query
        query_bundle = QueryBundle(query_str=search_query)
        results = []

        for retriever in self.retrievers:
            if observer:
                event_key = observer.start(name=f"CodeMultiCollectionRetriever retriever[{retriever.name}]")
            if isinstance(retriever.retriever, FilterCodeCollectionRetriever):
                code_results = retriever.retriever.search(
                    query=search_query,
                    apply_filters=apply_filters,
                    class_name=class_name,
                    method_name=method_name,
                    namespace=namespace,
                    file_path=file_path,
                    observer=observer,
                )
                self.logger.debug(
                    f"Found {len(code_results)} nodes in collection {retriever.retriever.collection_name}."
                )
                results.extend(code_results)
                if observer:
                    observer.stop(key=event_key, result_count=len(code_results))
            else:
                code_results = retriever.retrieve(query_bundle)
                results.extend(code_results)
                if observer:
                    observer.stop(key=event_key, result_count=len(code_results))

        for postprocessor in self.node_postprocessors:
            event_key = observer.start(name=f"Postprocessing [{postprocessor.__class__.__name__}]")
            results = postprocessor.postprocess_nodes(results, query_bundle=query_bundle)
            if observer:
                observer.stop(key=event_key)

        self.logger.debug(f"Returning {len(results)} nodes after postprocessing.")
        return results


class MessagesRetriever(BaseRetriever):
    name: str = Field(default="MessagesRetriever", description="The name of the node.")
    messages: List[dict] = Field(description="A list of messages.")
    default_score: int = Field(default=100, description="Default score for retrieved messages.")
    reference_label: str = Field(default="Previous exchange", description="Reference label for metadata.")

    def __init__(self, messages: List[dict], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.messages = messages
        self.name = kwargs.get("name", "MessagesRetriever")
        self.default_score = kwargs.get("default_score", 100)
        self.reference_label = kwargs.get("reference_label", "Previous exchange")

    def _to_node_with_score(self, question: dict, answer: dict) -> Optional[NodeWithScore]:
        q_role, q_content = question.get("role"), question.get("content")
        a_role, a_content = answer.get("role"), answer.get("content")

        if not q_content or not a_content:
            return None
        if q_role != "user" or a_role != "assistant":
            return None

        combined_text = f"Question: {normalize_message(q_content)}\n\nAnswer: {normalize_message(a_content)}"

        node = TextNode(
            id_=str(uuid4()),
            text=combined_text,
            metadata={
                "question_role": q_role,
                "answer_role": a_role,
                "reference": self.reference_label,
            },
        )
        return NodeWithScore(
            node=node,
            score=self.default_score,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes_with_score: List[NodeWithScore] = []
        for i, message in enumerate(self.messages):
            current_msg = self.messages[i]
            next_msg = self.messages[i + 1]

            if (current_msg.get("role"), next_msg.get("role")) == ("user", "assistant"):
                node_with_score = self._to_node_with_score(current_msg, next_msg)
                if node_with_score:
                    nodes_with_score.append(node_with_score)
                i += 2
            else:
                i += 1

        return nodes_with_score


class FilterCodeCollectionRetriever(BaseRetriever):
    """Class for retrieving data from collections based on structural filters"""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embed_model: Any,
        similarity_top_k: int,
        parallel: int = 1,
        sparse_text_embedding: Optional[SparseEncoderCallable] = None,
        logger: logging.Logger = None,
    ):
        super().__init__()

        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            parallel=parallel,
            sparse_doc_fn=sparse_text_embedding,
            sparse_query_fn=sparse_text_embedding,
            enable_hybrid=sparse_text_embedding is not None,
        )
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, storage_context=storage_context, embed_model=embed_model
        )

        self.logger = logger or setup_logger("FilterCodeCollectionRetriever", debug=False)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self.search(query=query_bundle.query_str)

    def search(
        self,
        query: str,
        apply_filters: bool,
        observer: Optional[Observer] = None,
        file_path: Optional[str] = None,
        **kwargs: Any,
    ) -> List[NodeWithScore]:

        self.logger.debug(f"Query: {query}, apply filters: {apply_filters}")
        if observer:
            event_key = observer.start(name=f"FilterCodeCollectionRetriever collection[{self.collection_name}]")

        query_str = query if query else "code"

        if apply_filters:
            metadata_filters = []
            for key, value in kwargs.items():
                if value is not None:
                    metadata_filters.append(MetadataFilter(key=key, operator=FilterOperator.TEXT_MATCH, value=value))

            if metadata_filters:
                filters = MetadataFilters(filters=metadata_filters, condition=FilterCondition.AND)
                retriever = self.index.as_retriever(
                    filters=filters, similarity_top_k=self.similarity_top_k, retriever_mode=ListRetrieverMode.DEFAULT
                )
            else:
                retriever = self.index.as_retriever(
                    similarity_top_k=self.similarity_top_k, retriever_mode=ListRetrieverMode.DEFAULT
                )
        else:
            retriever = self.index.as_retriever(
                similarity_top_k=self.similarity_top_k, retriever_mode=ListRetrieverMode.DEFAULT
            )

        results = retriever.retrieve(query_str)

        if file_path:
            path_filtered = []
            for result in results:
                fp = result.node.metadata.get("file_path", "")
                if fp and file_path in fp:
                    path_filtered.append(result)

            if path_filtered:
                results = path_filtered

        if observer:
            observer.stop(key=event_key, result_count=len(results))

        return results


@dataclass
class CodeEntities:
    """Dataclass for storing structural data (extracted from prompt by LLM)"""

    class_names: List[str] = field(default_factory=list)
    method_names: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    namespaces: List[str] = field(default_factory=list)
    search_context: Optional[str] = None
    exact_match_required: bool = False
    apply_filters: bool = False

    def to_dict(self, query: str) -> dict:
        return {
            "query": query,
            "apply_filters": self.apply_filters,
            "class_names": self.class_names,
            "method_names": self.method_names,
            "file_paths": self.file_paths,
            "namespaces": self.namespaces,
            "exact_match_required": self.exact_match_required,
            "search_context": self.search_context,
        }


class CodePromptDetector:
    """Uses pygments and LLM (enabled by default for quality) to check whether there is any code provided in text snippet"""

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://10.255.240.161:11434",
        temperature: float = 0.1,
        request_timeout: int = 60,
        pygments_threshold: float = 0.35,
        debug: bool = False,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.pygments_threshold = pygments_threshold
        self.logger = logger or setup_logger("CodePromptDetector", debug=debug)

    def is_code(self, query: str, use_llm: bool = True, observer: Optional[object] = None) -> bool:
        """Returns true if provided prompt looks like code snippet provided, else false."""
        if not query or not query.strip():
            return False

        query = query.strip()
        if observer:
            event_key = observer.start(name="CodePromptDetector")

        try:
            # 1. Pygments token counting
            lexer = guess_lexer(query)
            tokens = list(lexer.get_tokens(query))
            code_ratio = self._token_ratio(tokens)
            token_report = self._generate_token_report(tokens)

            self.logger.debug(
                f"Pygments analysis: {lexer.__class__.__name__}, ratio={code_ratio:.2f} token_report: {token_report}"
            )

            # 2. Pygments decision
            if not use_llm:
                result = code_ratio >= self.pygments_threshold
                self.logger.info(f"Pygments-only decision: {result}")
            else:  # Full verification with LLM
                token_report = self._generate_token_report(tokens)
                result = self._llm_verify(
                    query=query, lexer_name=lexer.__class__.__name__, code_ratio=code_ratio, token_report=token_report
                )
                self.logger.info(f"LLM-verified decision: is code: {result}")

            return result

        except ClassNotFound:
            self.logger.info("No lexer found - assuming not code")
            return False
        finally:
            if observer:
                observer.stop(key=event_key)

    def _llm_verify(self, query: str, lexer_name: str, code_ratio: float, token_report: str) -> bool:
        """Additional LLM verification"""
        try:
            prompt = CODE_DETECTOR_TEMPLATE.format(
                query=query, lexer=lexer_name, ratio=f"{code_ratio:.2f}", tokens=token_report
            )
            llm = Ollama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                request_timeout=self.request_timeout,
            )
            response = llm.complete(prompt)
            self.logger.debug(f"LLM response: {response.text}")

            cleaned_response = strip_whitespace(response.text)
            return cleaned_response == "FOUND"

        except Exception as e:
            self.logger.error(f"Error during entity extraction: {e}")
            return code_ratio >= self.pygments_threshold

    def _generate_token_report(self, tokens) -> str:
        """Create human-readable token analysis"""
        counts = defaultdict(int)
        for ttype, _ in tokens:
            counts[ttype] += 1
        return "\n".join(f"{str(k)}: {v}" for k, v in counts.items())

    def _token_ratio(self, tokens) -> float:
        """Estimate how code-like the snippet is based on token types."""
        total, code_like = 0, 0
        for ttype, _ in tokens:
            total += 1
            if ttype in (
                Token.Keyword,
                Token.Name,
                Token.Operator,
                Token.Literal,
                Token.Punctuation,
                Token.String,
                Token.Number,
                Token.Generic,
                Token.Name.Builtin,
                Token.Name.Attribute,
                Token.Name.Decorator,
                Token.Name.Entity,
                Token.Name.Exception,
                Token.Name.Function,
                Token.Name.Label,
                Token.Name.Namespace,
                Token.Name.Property,
                Token.Name.Tag,
                Token.Keyword.Constant,
                Token.Keyword.Declaration,
                Token.Keyword.Namespace,
                Token.Keyword.Pseudo,
                Token.Keyword.Reserved,
                Token.Keyword.Type,
                Token.Operator.Word,
                Token.Comment.Preproc,
                Token.Comment.Special,
            ):
                code_like += 1
        return code_like / total if total > 0 else 0


class CodeEntityExtractor:
    """Extracts code entities from query using LLM."""

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://10.255.240.161:11434",
        temperature: float = 0.1,
        request_timeout: int = 60,
        debug: bool = False,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.logger = logger or setup_logger("CodeEntityExtractor", debug=debug)

    @classmethod
    def from_query(
        cls,
        query: str,
        model: str = "mistral",
        base_url: str = "http://10.255.240.161:11434",
        temperature: float = 0.1,
        request_timeout: int = 60,
        debug: bool = False,
        logger: logging.Logger = None,
        observer: Optional[Observer] = None,
    ) -> CodeEntities:
        """Factory method"""
        extractor = cls(
            model=model,
            base_url=base_url,
            temperature=temperature,
            request_timeout=request_timeout,
            debug=debug,
            logger=logger,
        )
        return extractor.extract(query, observer=observer)

    def extract(self, query: str, observer: Optional[Observer] = None) -> CodeEntities:
        """
        Using LLM to extract metadata keys and create JSON
        NOTE! Can raise ValueError
        """
        self.logger.info(f"Extracting entities from query: {query}")
        prompt = EXTRACTOR_PROMPT_TEMPLATE.format(query=query)

        if observer:
            event_key = observer.start(name=f"CodeEntityExtractor model[{self.model}]")

        try:
            llm = Ollama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                request_timeout=self.request_timeout,
            )
            response = llm.complete(prompt)
            self.logger.debug(f"LLM response: {response.text}")

            data = self.extract_json(response.text)
            if observer:
                observer.stop(key=event_key)

            if self.validate_extracted_data(data):
                code_entities = CodeEntities(**data)
                code_entities.apply_filters = True
                return code_entities
            else:
                raise ValueError("Extracted data failed validation")

        except Exception as e:
            if observer:
                observer.stop(key=event_key)
            self.logger.error(f"Error during entity extraction: {e}")
            raise ValueError(f"Failed to extract entities: {e}")

    def extract_json(self, text: str) -> dict:
        """Extract JSON data from text using regex pattern matching."""

        # Look for JSON pattern
        matches = re.search(r"(\{[\s\S]*\})", text)
        if not matches:
            raise ValueError("No JSON object found in response")

        try:
            return json.loads(matches.group(1))
        except json.JSONDecodeError:
            return json.loads(matches.group(1).replace("'", '"'))

    def validate_extracted_data(self, data: Dict) -> bool:
        """Function to verify whether structural answer (from LLM) contains valid data."""
        expected_keys = {
            "class_names": list,
            "method_names": list,
            "file_paths": list,
            "namespaces": list,
            "exact_match_required": bool,
            "search_context": (str, type(None)),
        }
        for key, expected_type in expected_keys.items():
            if key not in data:
                self.logger.warning(f"Missing key in LLM output: {key}")
                return False

            value = data[key]
            if key == "search_context":
                if not (isinstance(value, str) or value is None):
                    self.logger.warning(f"Invalid type for {key}: expected str or None, got {type(value)}")
                    return False
            elif not isinstance(value, expected_type):
                self.logger.warning(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
                return False
        return True


class CodeMethodsExtractor:
    """
    Extracts method names from git diff
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://10.255.240.151:11434",  # ob host
        temperature: float = 0.1,
        request_timeout: int = 120,  # 2min timeout
        debug: bool = True,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.logger = logger or setup_logger("CodeMethodsExtractor", debug=debug)
        self.user_prompt = """### CODE BEGIN\n{diff}\n### CODE END"""

    def llm_extract(self, diff: str) -> Tuple:
        """LLM extraction"""
        try:
            res = requests.post(
                url=f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": CODE_METHODS_EXTRACTOR_TEMPLATE},
                        {"role": "user", "content": self.user_prompt.format(diff=diff)},
                    ],
                },
                timeout=60,
            )
            self.logger.info(f"Helper LLM raw response:  {res.json()}")

            if not res.ok:
                self.logger.error(f"LLM HTTP error {res.status_code}: {res.text}")
                return [], []

            reply = res.json()["choices"][0].get("message", {}).get("content", "{}")
            parsed = json.loads(re.sub(r"\\_", "_", reply[reply.find("{") : reply.rfind("}") + 1]))
            identifiers = parsed["identifiers"] if isinstance(parsed, dict) and "identifiers" in parsed else []
            removed_identifiers = (
                parsed["removed_identifiers"] if isinstance(parsed, dict) and "removed_identifiers" in parsed else []
            )

            self.logger.info(f"Helper LLM final return: {identifiers}")
            return identifiers, removed_identifiers
        except Exception as e:
            self.logger.exception("Error extracting identifiers: %s", e)
            return [], []


class ObjectRepository:
    """
    A thread-safe repository for managing shared objects. This class ensures that objects
    are created or retrieved in a thread-safe manner and supports object deletion.
    """

    def __init__(self):
        """
        Initializes the ObjectRepository with internal dictionaries for storing objects,
        locks, and factory arguments, as well as a global lock for thread safety.
        """
        self._objects = {}  # name -> object
        self._locks = {}  # name -> threading.Lock
        self._factory_args = {}  # name -> hashed factory arguments
        self._global_lock = threading.Lock()

    def _get_lock_for_key(self, name: str) -> threading.Lock:
        """
        Retrieves or creates a lock for a given object name.

        Args:
            name (str): The name of the object.

        Returns:
            threading.Lock: A lock associated with the given name.
        """
        with self._global_lock:
            if name not in self._locks:
                self._locks[name] = threading.Lock()
            return self._locks[name]

    @staticmethod
    def _hash_args(kwargs: dict) -> str:
        """
        Generates a hash for the given arguments.

        Args:
            kwargs (dict): The arguments to hash.

        Returns:
            str: A SHA-256 hash of the serialized arguments.
        """
        try:
            serialized = json.dumps(kwargs, sort_keys=True)
        except TypeError:
            # If serialization fails, use a simple string representation
            serialized = str(kwargs)

        # Generate a SHA-256 hash of the serialized string
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get_or_create(self, name: str, factory: Callable[..., Any], **kwargs):
        """
        Retrieves an existing object or creates a new one using the provided factory function.

        Args:
            name (str): The name of the object.
            factory (callable): A factory function to create the object if it does not exist.
            **kwargs: Additional arguments to pass to the factory function.

        Returns:
            Any: The retrieved or newly created object.
        """
        lock = self._get_lock_for_key(name)
        with lock:
            new_hash = self._hash_args(kwargs)
            existing_hash = self._factory_args.get(name)

            # Check if the object already exists and if the hash matches
            # If not, create a new object
            if name not in self._objects or new_hash != existing_hash:
                self._objects[name] = factory(**kwargs)
                self._factory_args[name] = new_hash

            return self._objects[name]

    def delete(self, name: str):
        """
        Deletes an object and its associated metadata from the repository.

        Args:
            name (str): The name of the object to delete.
        """
        lock = self._get_lock_for_key(name)
        with lock:
            if name in self._objects:
                del self._objects[name]
                del self._factory_args[name]

        with self._global_lock:
            self._locks.pop(name, None)


#################################
# Helper functions
#################################


def normalize_message(text):
    # Remove the References section and everything below it
    text = re.sub(r"\*\*\*References:\*\*\*.*?(?=\n---|\Z)", "", text, flags=re.DOTALL)

    # Remove the Times section and everything below it
    text = re.sub(r"\*\*\*Times:\*\*\*.*", "", text, flags=re.DOTALL)

    # Convert from markdown to text
    text = mistune.html(text)

    return text


def setup_logger(name: str, debug: bool = False, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting and optional file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.propagate = False  # Prevent duplicate logs

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


def wrap_text_in_box(text: str, max_line_length: int = 90) -> str:
    """Wrap long text into a box with line breaks, ensuring words are not split."""
    wrapped_lines = []
    for line in text.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=max_line_length))

    # If there are no lines, return an empty string
    if not wrapped_lines:
        return ""

    formatted_lines = [wrapped_lines[0]]  # First line without `>`
    if len(wrapped_lines) > 1:
        formatted_lines[0] += " \\"  # If there are more lines, add `\` to the first line
        formatted_lines += [f"> {line} \\" for line in wrapped_lines[1:-1]]  # Lines in the middle with `>`
        formatted_lines.append(f"> {wrapped_lines[-1]}")  # Last line without `\`

    return "\n".join(formatted_lines)


def metadata_to_string(metadata: dict) -> str:
    return "\n".join([f"- **{key}**: `{value}`" for key, value in metadata.items()])


def parse_nodes_to_markdown(user_message: str, nodes: list, observer: Observer) -> str:
    output = [f"## 🔍 Query", f"_{user_message}_", ""]

    for i, node in enumerate(nodes):
        output.extend(
            [
                f"---",
                f"### 📌 ID: `{node.id_}`",
                f"### 🎯 Score: `{node.score:.0f}%`",
                "",
                f"### 📂 Metadata",
                metadata_to_string(node.metadata),
                "",
                f"### 📝 Text",
                f"```text",
                wrap_text_in_box(node.text),
                f"```",
                "",
            ]
        )

        if i == len(nodes) - 1:
            output.append("---")

    if observer is not None:
        output.extend(["", observer.summary()])

    return "\n".join(output)


def parse_code_nodes_to_markdown(user_message: str, nodes: list, observer: Observer) -> str:
    output = f"**Code Analysis Query**: _{user_message}_\n\n"
    output += f"**Total Relevant Snippets**: {len(nodes)}\n"

    sorted_nodes = sorted(nodes, key=lambda x: x.score or 0, reverse=True)

    for node in sorted_nodes:
        metadata = node.metadata
        details = []

        if node.score is not None:
            details.append(f"**Relevance**: {node.score:.2f}%")
        if file_path := metadata.get("file_path"):
            details.append(f"**File**: `{file_path}`")
        if class_name := metadata.get("class_name"):
            details.append(f"**Class**: `{class_name}`")
        if namespace := metadata.get("namespace"):
            details.append(f"**Namespace**: `{namespace}`")
        if token_count := metadata.get("token_count"):
            details.append(f"**Token count**: `{token_count}`")

        method_or_func = metadata.get("method_name") or metadata.get("function_name")
        if method_or_func:
            details.append(f"**Function/Method**: `{method_or_func}`")

        if details:
            output += "\n".join(details) + "\n\n"

        # defaulted to cpp
        language = metadata.get("programming_language", "cpp")
        output += f"**Code**:\n```{language}\n{node.text}\n```\n\n"

    if observer is not None:
        output += observer.summary()

    return output


def strip_whitespace(text: str) -> str:
    """Removes all spaces, tabs, and newlines from the input string."""
    return re.sub(r"[\s\t\n]+", "", text)
