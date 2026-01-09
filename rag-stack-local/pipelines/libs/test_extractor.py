"""
description: Script for retrieving chunks from code collections in Qdrant using filters

Usage examples:
    python test_extractor.py
    python test_extractor.py --query "show me VideoEncoder and X265Encoder classes"
"""

import argparse
from typing import Optional, List
from qdrant_client import QdrantClient
from llama_index.core.schema import NodeWithScore
from typing import Any

from tools import (
    CodeEntityExtractor,
    setup_logger,
)

LLM_CONFIG = {
    "BASE_URL": "http://10.255.240.161:11434",
    "MODEL": "mistral",
}

DEFAULT_TEST_QUERIES = [
    "show me VideoEncoder class",
    "show me VideoEncoder and X265Encoder classes",
    "show me aftpclient class definition",
    "find methods that process frames",
    "show me contents of encoder.py file",
    "show me rg::coder namespace",
    "show implementation of processFrame method",
    "find codecs for H264",
    "what are the video encoders available",
    "list all decoders in the codebase",
    "how does the audio encoder implement buffering",
]


def test_extraction(
    query: str,
    model_name: str,
    model_base_url: str,
) -> dict:

    try:
        code_entity = CodeEntityExtractor.from_query(
            query=query, model=model_name, base_url=model_base_url, temperature=0.2
        )
        result = code_entity.to_dict(query=query)

        print("\n" + "=" * 50)
        print(f'QUERY: "{query}"')
        print("\n=== EXTRACTED INFORMATION ===")
        print(f"Class Names:      {code_entity.class_names if code_entity.class_names else 'None'}")
        print(f"Method Names:     {code_entity.method_names if code_entity.method_names else 'None'}")
        print(f"File Paths:       {code_entity.file_paths if code_entity.file_paths else 'None'}")
        print(f"Namespaces:       {code_entity.namespaces if code_entity.namespaces else 'None'}")
        print(f"Exact Match:      {code_entity.exact_match_required}")
        print(f"Search Context:   {code_entity.search_context if code_entity.search_context else 'None'}")

        return result

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"LLM Extraction failed: {e}")
        print(f"query: {query}")

        result = {"query": query}
        return result


def run_extraction_tests(
    queries: List[str],
    model_name: str,
    model_base_url: str,
) -> List[dict]:

    results = []
    for query in queries:
        result = test_extraction(query=query, model_name=model_name, model_base_url=model_base_url)
        results.append(result)

    return results


def main():

    logger = setup_logger("test_extractor")
    parser = argparse.ArgumentParser(description="Search code in Qdrant collection")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    if args.query:
        queries = [args.query]
    else:
        queries = DEFAULT_TEST_QUERIES

    model = LLM_CONFIG["MODEL"]
    model_url = LLM_CONFIG["BASE_URL"]

    logger.info("CodeEntityExtractor Test")
    logger.info("=======================")
    logger.info(f"Model: {model}")
    logger.info(f"Model URL: {model_url}")
    logger.info(f"Testing {len(queries)} queries")

    run_extraction_tests(
        queries=queries,
        model_name=model,
        model_base_url=model_url,
    )


if __name__ == "__main__":
    main()
