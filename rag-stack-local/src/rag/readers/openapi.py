"""
author: Jakub Durlik
date: 2025-02-31
version: 1.0
description: OpenAPI reader
"""

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import Optional, List, Any, Dict
import yaml
import re


class OpenAPIYamlReader(BaseReader):
    """OpenAPI reader.

    Reads OpenAPI specifications giving options to on how to parse them.

    Returns:
        List[Document]: List of documents with paths and parameters.
    """

    def __init__(self, depth: Optional[int] = 1, exclude: Optional[List[str]] = None) -> None:
        super().__init__()
        self.exclude = exclude
        self.depth = depth

    @staticmethod
    def extract_param_name(pattern: str) -> str:
        match = re.search(r'#/components/parameters/(.+)', pattern)
        if match:
            return match.group(1)
        return pattern

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "OpenAPIYamlReader"

    def _should_exclude(self, path: str) -> bool:
        """Check if the path should be excluded."""
        return self.exclude and any(
            path.endswith(exclude_path) for exclude_path in self.exclude
        )

    def _build_path_attribute(self, key: str,value: Any) -> List[Document]:
        """Build Documents from the path attributes of the YAML."""
        if "get" in value.keys() and "description" in value["get"].keys():
            description = value["get"]["description"]
            parameters = []
            if "parameters" in value["get"].keys():
                parameters = [self.extract_param_name(parameter["$ref"]) for parameter in value["get"]["parameters"]]
            return [
                Document(
                    text=f"{description}", metadata={"openAPI_key": key, "openAPI_entry": value, "parameters": parameters, "openAPI_type": "path"}
                )
            ]
        return []

    @staticmethod
    def _build_parameter_attribute(key: str, value: Any) -> List[Document]:
        """Build Documents from the parameter attributes of the YAML."""
        description = value["description"]
        return [
            Document(
                text=f"{key} parameter {description}",
                metadata={"openAPI_key": key, "openAPI_entry": value, "parameters": {}, "openAPI_type": "parameter"}
            )
        ]

    def load_data(self, input_file: str) -> (List[Document], List[Document], Dict):
        """Load data from the input file."""
        with open(input_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        paths = [
            doc
            for key, value in data["paths"].items()
            for doc in self._build_path_attribute(key, value)
        ]
        parameters = [
            doc
            for key, value in data["components"]["parameters"].items()
            for doc in self._build_parameter_attribute(key, value)
        ]

        return paths + parameters
