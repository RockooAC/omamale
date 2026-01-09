"""
description: Script for retrieving chunks from code collections in Qdrant using filters

Usage examples:
    python test_filtering.py --collection code_coder_prod --namespace rg::coder
    python test_filtering.py --collection code_core_prod --class VideoEncoder
    python test_filtering.py --collection code_core_prod --method processFrame
"""

import argparse
import os
import sys
from qdrant_client import QdrantClient
from llama_index.embeddings.ollama import OllamaEmbedding
from tools import FilterCodeCollectionRetriever, Observer, setup_logger

QDRANT_CONFIG = {
    "BASE_URL": "http://10.255.240.18:6333",
}

EMBEDDER_CONFIG = {
    "BASE_URL": "http://10.255.240.161:11434",
    "MODEL": "jina/jina-embeddings-v2-base-en:latest",
}


def print_results(results, limit=None):
    count = len(results)
    if count == 0:
        print("\nNo results found matching the criteria.")
        return

    print(f"\nFound {count} matching results:")

    # Limit results if specified
    if limit and limit < count:
        results = results[:limit]
        print(f"Displaying first {limit} results:")

    for i, result in enumerate(results, 1):
        node = result.node
        metadata = node.metadata

        print(f"\n------- Result #{i} -------")
        print(f"ID: {node.id_}")

        if result.score is not None:
            print(f"Score: {result.score:.4f}")

        if metadata.get("class_name"):
            print(f"Class: {metadata.get('class_name')}")

        if metadata.get("method_name"):
            print(f"Method: {metadata.get('method_name')}")

        if metadata.get("file_path"):
            print(f"File: {metadata.get('file_path')}")

        if metadata.get("namespace"):
            print(f"Namespace: {metadata.get('namespace')}")

        # Print the code text
        print("\nCode:")
        print("```")
        print(node.text)
        print("```")


def main():

    logger = setup_logger("test_filtering")

    parser = argparse.ArgumentParser(description="Search code in Qdrant collection")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--class", dest="class_name")
    parser.add_argument("--method", dest="method_name")
    parser.add_argument("--namespace")
    parser.add_argument("--file", dest="file_path")
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=50)

    args = parser.parse_args()
    url = QDRANT_CONFIG["BASE_URL"]
    embedder_url = EMBEDDER_CONFIG["BASE_URL"]
    embedder_model = EMBEDDER_CONFIG["MODEL"]

    logger.info(f"Connecting to Qdrant at {url}")
    logger.info(f"Using embedder at {embedder_url} with model {embedder_model}")

    try:
        client = QdrantClient(url=url)
        embed_model = OllamaEmbedding(base_url=embedder_url, model_name=embedder_model)

        observer = Observer()
        event_key = observer.start(name="Search Code")

        code_retriever = FilterCodeCollectionRetriever(
            client=client,
            collection_name=args.collection,
            embed_model=embed_model,
            similarity_top_k=args.limit,
        )

        results = code_retriever.search(
            query=args.query,
            class_name=args.class_name,
            method_name=args.method_name,
            namespace=args.namespace,
            file_path=args.file_path,
            observer=observer,
        )

        observer.stop(key=event_key, result_count=len(results))
        print(observer.summary())
        print_results(results, args.limit)

    except Exception as e:
        logger.error(f"Error retrieving from Qdrant: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
