#! /usr/bin/python3

import os
import sys
import logging
from typing import List, NoReturn, Union
from qdrant_client import QdrantClient, models
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.readers.confluence import ConfluenceReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from rag.embedding.prod.GlobalsProd import *


class DataIngestion:
    def __init__(self, collection_name: str, cache_collection_name: Union[str, None]):
        self.pipeline = None
        self.insert_strategy = DocstoreStrategy.UPSERTS
        self.__setup_logger__()
        self.collection = collection_name
        self.cache_collection = cache_collection_name
        self.dynamic_vector_size = self.__embedder__.get_text_embedding("test").__len__()

    def __setup_logger__(self) -> logging:
        self.logger = logging
        self.logger.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.DEBUG
        )

    @property
    def __embedder__(self) -> OllamaEmbedding:
        return OllamaEmbedding(
            base_url=EMBEDDER_BASE_URL,
            model_name=EMBEDDER_MODEL_GWEN
        )

    @property
    def __embedder_jina__(self) -> OllamaEmbedding:
        return OllamaEmbedding(
            base_url=EMBEDDER_BASE_JINA_URL,
            model_name=EMBEDDER_MODEL_JINA
        )

    @property
    def __doc_splitter__(self) -> SentenceSplitter:
        return SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    @property
    def __dynamic_doc_splitter__(self) -> SentenceSplitter:
        return SentenceSplitter(
            chunk_size=round(0.8 * self.dynamic_vector_size),
            chunk_overlap=round(0.15 * self.dynamic_vector_size)
        )

    @property
    def __vector_store__(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=QdrantClient(QDRANT_BASE_URL, timeout=30),
            collection_name=self.collection
        )

    @property
    def __bm42_hybrid_vector_store__(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=QdrantClient(QDRANT_BASE_URL, timeout=30),
            collection_name=self.collection,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
            vectors_config={
                "text-dense": models.VectorParams(
                    # size=self.dynamic_vector_size // Use dynamic if needed
                    size=CHUNK_SIZE,
                    distance=models.Distance.DOT,
                )
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams()
                )
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            )
        )

    @property
    def __hybrid_vector_store__(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=QdrantClient(QDRANT_BASE_URL, timeout=30),
            collection_name=self.collection,
            enable_hybrid=True,
            vectors_config={
                "text-dense": models.VectorParams(
                    # size=self.dynamic_vector_size // Use dynamic if needed
                    size=CHUNK_SIZE,
                    distance=models.Distance.DOT,
                )
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams()
                )
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            )
        )

    @property
    def __cache__(self) -> IngestionCache:
        return IngestionCache(
            cache=RedisCache.from_host_and_port(
                host=REDIS_HOST,
                port=REDIS_PORT
            ),
            collection=self.cache_collection,
        )

    @property
    def __docstore__(self) -> RedisDocumentStore:
        return RedisDocumentStore.from_host_and_port(
            host=REDIS_HOST,
            port=REDIS_PORT,
            namespace=self.cache_collection
        )

    @staticmethod
    def __docs_reader__(docs_path: str, **kwargs) -> SimpleDirectoryReader:
        return SimpleDirectoryReader(
            input_dir=docs_path,
            required_exts=[".md"],
            recursive=True,
            filename_as_id=True,
            **kwargs
        )

    @staticmethod
    def __confluence_reader__(access_token: str, **kwargs) -> ConfluenceReader:
        return ConfluenceReader(
            api_token=access_token,
            base_url=CONFLUENCE_BASE_URL,
            cloud=False,
            **kwargs
        )

    @staticmethod
    def __pdf_reader__(pdf_path: str, **kwargs) -> SimpleDirectoryReader:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
        from rag.embedding.grobid_pdf_reader.grobid_pdf_reader import GrobidPDFReader
        return SimpleDirectoryReader(
            input_dir=pdf_path,
            recursive=True,
            filename_as_id=True,
            required_exts=[".pdf"],
            file_extractor={
                ".pdf": GrobidPDFReader(
                    grobid_server=GROBID_BASE_URL,
                    split_sentence=GROBID_SPLIT_SENTENCE,
                    load_figures=GROBID_LOAD_FIGURES
                ),
            },
            **kwargs

        )

    @staticmethod
    def __codebase_parser__(source_dir: str, output_dir: str, debug: bool, **kwargs) -> "RepoReader":
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
        from src.rag.readers.repo import RepoReader
        return RepoReader(
            repo_path=source_dir,
            output_dir=output_dir,
            debug=debug,
            **kwargs
        )

    @staticmethod
    def __cpp_code_reader__(debug: bool, language: str = "cpp") -> "CppReader":
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
        from src.rag.readers.cpp import CppReader
        return CppReader(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            debug=debug,
            language=language
        )

    @staticmethod
    def __openapi_reader__(depth: int = 1, exclude: List = None):
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
        from src.rag.readers.openapi import OpenAPIYamlReader
        return OpenAPIYamlReader(
            depth=depth,
            exclude=exclude
        )

    @staticmethod
    def __webpage_reader__(**kwargs) -> BeautifulSoupWebReader:
        return BeautifulSoupWebReader(
            **kwargs
        )

    @staticmethod
    def __ingestion_pipeline__(transformations: List, **kwargs) -> IngestionPipeline:
        return IngestionPipeline(
            transformations=transformations,
            **kwargs
        )

    @staticmethod
    def _adjust_dynamic_metadata(documents) -> NoReturn:
        for doc_object in documents:
            doc_object.metadata.pop("creation_date", None)
            doc_object.metadata.pop("last_modified_date", None)

