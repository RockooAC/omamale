"""
author: Karol Siegieda
date: 2025-02-24
version: 1.0
description: Cpp code reader
"""

import re
import tree_sitter as TS
import tree_sitter_cpp as TSCpp

from typing import Dict, List, Optional, Tuple, Optional
from pathlib import Path
from llama_index.core.schema import Document

from src.rag.readers.chunk import Chunk
from src.rag.readers.base import BaseReader
from src.rag.config import CHUNK_CONFIG


CHUNK_SIZE = CHUNK_CONFIG["SIZE"]
CHUNK_OVERLAP = CHUNK_CONFIG["OVERLAP"]

FUNCTION_NODE_TYPES = {"function_definition", "method_definition"}


class CppParser:
    """Handles C++ code parsing using tree-sitter."""

    def __init__(self):
        self.parser = TS.Parser(TS.Language(TSCpp.language()))

    def parse(self, code: str) -> TS.Tree:
        """Parse code into a Tree-Sitter syntax tree."""
        return self.parser.parse(code.encode("utf-8"))

    def get_class_name(self, node: TS.Node, source_code: str) -> str:
        for child in node.children:
            if child.type == "name":
                return source_code[child.start_byte : child.end_byte].strip()
        return ""

    def extract_namespace_name(self, node: TS.Node, source_code: str) -> str:
        """Extract namespace name, properly handling nested namespaces with :: syntax."""
        # Check all children to find namespace declaration
        namespace_text = ""
        for child in node.children:
            # Skip braces and other non-identifiers
            if child.type in ["namespace", "{", "}"]:
                continue

            # For C++17 style nested namespaces (namespace rg::core::fairplay {)
            if child.type in ["identifier", "namespace_identifier", "scoped_identifier"]:
                namespace_text = source_code[child.start_byte : child.end_byte].strip()
                break

        return namespace_text

    def get_function_name(self, node: TS.Node, source_code: str) -> str:
        function_declarator = next((child for child in node.children if child.type == "function_declarator"), None)
        if not function_declarator:
            return ""

        declarator = function_declarator.child_by_field_name("declarator")
        if declarator and declarator.type in ["identifier", "qualified_identifier", "scoped_identifier"]:
            return source_code[declarator.start_byte : declarator.end_byte].strip()
        return ""

    def has_function_definition(self, text: str) -> bool:
        return any(node_type in str(self.parse(text).root_node) for node_type in FUNCTION_NODE_TYPES)


class CppReader(BaseReader):
    """Main class for reading and chunking C++ code."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        debug: bool = True,
        language: str = "cpp",
        repo_root_path: str = "src/rg/",
        namespace_root: str = "rg",
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, debug=debug, normalize=False)
        self.parser = CppParser()
        self.method_chunk_size = chunk_size * 5
        self.current_namespaces: List[str] = []
        self.current_class: str = ""
        self.language: str = language
        self.repo_root_path: str = repo_root_path
        self.namespace_root = namespace_root

    def load_file(self, file: Path, extra_info: Optional[Dict] = None) -> List[Chunk]:
        """Load and process a single file with namespace inference."""
        self.logger.info(f"Loading file: {file}")
        try:
            # Try to infer namespace from filepath
            if "src/rg/" in str(file):
                parts = str(file).split(self.repo_root_path)[1].split("/")
                if len(parts) > 1:
                    self.current_namespaces = [self.namespace_root] + [parts[0]]
                    self.logger.debug(f"Inferred namespaces from path: {self.current_namespaces}")

            with file.open("r", encoding="utf-8") as f:
                code = f.read()
            return self.process_file(code, file.as_posix())
        except Exception as e:
            self.logger.error(f"Error processing file {file}: {e}")
            return []

    def load_data(self, digest_path: str, extra_info: Optional[Dict] = None) -> Dict[str, List[Document]]:
        """Process .digest file(s)."""
        path = Path(digest_path)
        results = {}

        if path.is_dir():
            for digest in path.glob("*.digest"):
                results.update(self.process_digest(digest, extra_info))
        elif path.suffix == ".digest":
            results.update(self.process_digest(path, extra_info))
        else:
            self.logger.error(f"Not a .digest file or directory: {path}")

        if self.debug:
            self.debug_print_summary(results)

        return results

    def process_file(self, code: str, filename: str) -> List[Chunk]:
        """Process a single C++ file."""
        self.logger.debug(f"Processing file: {filename}")

        # Search for namespace declarations in the raw code
        namespace_matches = re.findall(r"namespace\s+([^\s{;]+)", code)
        self.logger.debug(f"Regex found namespaces: {namespace_matches}")

        tree = self.parser.parse(code)
        self.logger.debug(f"Root node type: {tree.root_node.type}")

        # Reset namespaces before processing
        self.current_namespaces = []

        chunks = self.process_node(tree.root_node, code, filename)
        result = self.coalesce_chunks(chunks, code)

        return result

    def process_digest(self, file_path: Path, extra_info: Optional[Dict] = None) -> Dict[str, List[Document]]:
        self.logger.info(f"Processing file: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Digest file does not exist: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            ValueError(f"Failed to parse digest: {file_path}")


        """ Split based on separator lines, example:
                ================================================
                File: src/time.cpp
                ================================================
        """
        files = re.split(r"=+\s*\n\s*file:\s*(.*?)\s*\n\s*=+\s*\n", content, flags=re.IGNORECASE)

        if len(files) <= 1:
            raise ValueError(f"Failed to parse digest structure: no files found in {file_path}")

        file_map = {files[i]: files[i + 1] for i in range(1, len(files), 2)}

        if not file_map:
            raise ValueError(f"No valid files extracted from digest: {file_path}")

        results = {}
        for file_name, code in file_map.items():
            self.logger.debug(f"Processing {file_name} with {len(code)} bytes")
            try:
                chunks = self.process_file(code, file_name)
            except Exception as e:
                raise ValueError(f"Failed to process file '{file_name}': {e}")

            if self.debug:
                for i, chunk in enumerate(chunks):
                    self.debug_print_chunk(chunk, code, i)

            results[file_name] = {"chunks": chunks, "source": code, "extra_info": extra_info}

        return results

    def process_node(self, node: TS.Node, source_code: str, filename: str) -> List[Chunk]:
        """Process node."""

        # First check if we're at file level and need to manually detect namespaces
        if node.type == "translation_unit" and len(self.current_namespaces) == 0:
            matches = re.findall(r"namespace\s+([\w:]+)\s*{", source_code)
            for match in matches:
                if "::" in match:
                    self.logger.debug(f"Manually detected nested namespace: {match}")
                    self.current_namespaces = match.split("::")

        chunks = []
        current_chunk = Chunk(node.start_byte, node.start_byte, filename, self.language, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1)
        current_chunk.set_scope(self.current_namespaces, self.current_class)

        for child in node.children:
            child_text = source_code[child.start_byte : child.end_byte]

            if child.type == "namespace_definition":
                namespace_name = self.parser.extract_namespace_name(child, source_code)
                if self.debug:
                    self.logger.debug(f"Found namespace: '{namespace_name}'")

                if len(current_chunk) > 0:
                    chunks.append(current_chunk)

                # Save original namespace state
                old_namespaces = self.current_namespaces.copy()

                # Handle both single and nested namespaces
                if "::" in namespace_name:
                    self.current_namespaces = namespace_name.split("::")
                else:
                    self.current_namespaces.append(namespace_name)

                if self.debug:
                    self.logger.debug(f"Current namespace stack: {self.current_namespaces}")

                # Process children inside namespace
                chunks.extend(self.process_node(child, source_code, filename))

                # Restore previous namespace state
                self.current_namespaces = old_namespaces

                # Start a new chunk after namespace
                current_chunk = Chunk(child.end_byte, child.end_byte, filename, self.language, start_line=child.start_point[0] + 1, end_line=child.end_point[0] + 1)
                current_chunk.set_scope(self.current_namespaces, self.current_class)
                continue

            if child.type == "class_specifier":
                class_name = self.parser.get_class_name(child, source_code)
                if len(current_chunk) > 0:
                    chunks.append(current_chunk)

                old_class = self.current_class
                self.current_class = class_name

                class_chunk = Chunk(child.start_byte, child.end_byte, filename, self.language, start_line=child.start_point[0] + 1, end_line=child.end_point[0] + 1)
                class_chunk.set_scope(self.current_namespaces, class_name)
                chunks.append(class_chunk)

                class_content = self.process_node(child, source_code, filename)
                for chunk in class_content:
                    chunk.set_scope(self.current_namespaces, class_name)
                chunks.extend(class_content)

                self.current_class = old_class
                current_chunk = Chunk(child.end_byte, child.end_byte, filename, self.language, start_line=child.start_point[0] + 1, end_line=child.end_point[0] + 1)
                continue

            if child.type in FUNCTION_NODE_TYPES:
                if len(current_chunk) > 0:
                    chunks.append(current_chunk)

                method_name = self.parser.get_function_name(child, source_code)
                if "::" in method_name:
                    class_name = method_name.split("::")[0]
                    method_name = method_name.split("::")[-1]
                else:
                    class_name = self.current_class

                method_chunk = Chunk(child.start_byte, child.end_byte, filename, self.language, start_line=child.start_point[0] + 1, end_line=child.end_point[0] + 1)
                method_chunk.set_scope(self.current_namespaces, class_name, method_name)
                chunks.append(method_chunk)

                current_chunk = Chunk(child.end_byte, child.end_byte, filename, self.language, start_line=child.start_point[0] + 1, end_line=child.end_point[0] + 1)
                current_chunk.set_scope(self.current_namespaces, class_name)
                continue

            if len(child_text) > self.chunk_size:
                if len(current_chunk) > 0:
                    chunks.append(current_chunk)
                chunks.extend(self.process_node(child, source_code, filename))
                current_chunk = Chunk(child.end_byte, child.end_byte, filename, self.language, start_line=child.start_point[0] + 1, end_line=child.end_point[0] + 1)
                current_chunk.set_scope(self.current_namespaces, self.current_class)
            else:
                if len(current_chunk) == 0:
                    current_chunk = Chunk(child.start_byte, child.end_byte, filename, self.language, start_line=child.start_point[0] + 1, end_line=child.end_point[0] + 1)
                else:
                    current_chunk = Chunk(current_chunk.start, child.end_byte, filename, self.language, start_line=current_chunk.start_line, end_line=child.end_point[0] + 1)
                current_chunk.set_scope(self.current_namespaces, self.current_class)

        if len(current_chunk) > 0:
            current_chunk.set_scope(self.current_namespaces, self.current_class)
            chunks.append(current_chunk)

        return chunks

    def process_method_node(self, node: TS.Node, source_code: str, filename: str) -> List[Chunk]:
        """Process a method or function node."""
        node_text = source_code[node.start_byte : node.end_byte]
        if len(node_text) <= self.chunk_size:
            return [Chunk(node.start_byte, node.end_byte, filename, self.language, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1)]

        return self.split_large_method(node, source_code, filename)

    def split_large_method(self, node: TS.Node, source_code: str, filename: str) -> List[Chunk]:
        """
        Split large methods at logical boundaries. If splitting is not required, return as a single chunk.
        """
        text = source_code[node.start_byte : node.end_byte]
        if not self.debug or len(text) <= self.method_chunk_size:
            # Return as a single chunk if the method fits or splitting is not strictly required
            return [Chunk(node.start_byte, node.end_byte, filename, self.language, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1)]

        first_brace = text.find("{")
        if first_brace == -1:
            return [Chunk(node.start_byte, node.end_byte, filename, self.language, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1)]

        chunks = []
        signature = text[: first_brace + 1]
        chunks.append(Chunk(node.start_byte, node.start_byte + len(signature), filename, self.language, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1))

        current_pos = node.start_byte + len(signature)
        current_chunk = ""
        brace_count = 1

        for line in text[first_brace + 1 :].split("\n"):
            brace_count += line.count("{") - line.count("}")
            current_chunk += line + "\n"

            if len(current_chunk) > self.chunk_size and (brace_count == 1 or line.strip().endswith(";")):
                chunks.append(Chunk(current_pos, current_pos + len(current_chunk), filename, self.language, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1))
                current_pos += len(current_chunk)
                current_chunk = ""

        if current_chunk:
            chunks.append(Chunk(current_pos, node.end_byte, filename, self.language, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1))

        return chunks

    def process_regular_node(
        self, current_chunk: Chunk, child: TS.Node, child_text: str, filename: str
    ) -> Tuple[List[Chunk], Chunk]:
        """Process a regular (non-method) node."""
        chunks = []
        if len(current_chunk) + len(child_text) > self.chunk_size:
            if len(current_chunk) > 0:
                chunks.append(current_chunk)
            current_chunk = Chunk(child.start_byte, child.end_byte, filename, self.language, start_line=child.start_point[0] + 1, end_line=child.end_point[0] + 1)
        else:
            if len(current_chunk) == 0:
                current_chunk = Chunk(child.start_byte, child.end_byte, filename, self.language, start_line=child.start_point[0] + 1, end_line=child.end_point[0] + 1)
            else:
                current_chunk = Chunk(current_chunk.start, child.end_byte, filename, self.language, start_line=current_chunk.start_line, end_line=child.end_point[0] + 1)

        if self.debug:
            print(f"Regular node: {current_chunk}")
        return chunks, current_chunk

    def coalesce_chunks(self, chunks: List[Chunk], source_code: str) -> List[Chunk]:
        """Coalesce small chunks while respecting method boundaries."""
        if not chunks:
            return chunks

        result = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            current_text = current.extract(source_code)
            next_text = next_chunk.extract(source_code)

            if (
                self.parser.has_function_definition(current_text)
                or self.parser.has_function_definition(next_text)
                or len(current_text) + len(next_text) > self.chunk_overlap
            ):
                result.append(current)
                current = next_chunk
            else:
                current = Chunk(current.start, next_chunk.end, current.filename, self.language, start_line=current.start_line, end_line=next_chunk.end_line)

        result.append(current)
        return result

    @staticmethod
    def to_documents(chunks_by_file: Dict[str, List["Chunk"]]) -> List[Document]:
        """
        Convert a dictionary of chunks grouped by file into a list of Document objects.
        """
        documents = []
        for file_name, chunks in chunks_by_file.items():
            for chunk in chunks:
                doc = chunk.to_document(chunk.filename)
                if doc is not None:
                    documents.append(doc)
        return documents

    def debug_print_chunk(self, chunk: Chunk, source_code: str, index: int) -> None:
        """Log debug information for a chunk."""
        if not self.debug:
            return

        content = chunk.extract(source_code)
        print(f"\nChunk #{index}: {chunk.__repr__()}")
        print("=" * 40)
        if chunk.scope:
            print(f"Scope: {chunk.scope}")
        print(content)
        print("=" * 40)

        if self.parser.has_function_definition(content):
            print("  (Contains method/function)")

    def debug_print_summary(self, chunks_by_file: Dict[str, List[Chunk]]) -> None:
        """Log summary of processed files and chunks."""
        if not self.debug:
            return

        print("\nProcessing Summary:")
        print("=" * 40)
        for file_name, chunk_list in chunks_by_file.items():
            print(f"File: {file_name}, Chunks: {len(chunk_list)}")
