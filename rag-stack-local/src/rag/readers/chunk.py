from llama_index.core import Document
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class Chunk:
    start: int
    end: int
    filename: str
    language: str
    scope: str = ""
    start_line: int = 0
    end_line: int = 0

    def __len__(self) -> int:
        return self.end - self.start

    def extract(self, source_code: str) -> str:
        """Extract the code content of the chunk from source."""
        if not source_code or not isinstance(source_code, str):
            print(f"Invalid source_code: {type(source_code)}")  # Add debug logging
            return ""
        if self.start >= len(source_code) or self.end > len(source_code):
            print(f"Invalid bounds: start={self.start}, end={self.end}, len={len(source_code)}")  # Add debug logging
            return ""
        return source_code[self.start : self.end]

    def to_document(self, source_code: str, extra_info: Optional[Dict] = None) -> Optional[Document]:
        """Convert chunk to Document with metadata."""
        text_content = self.extract(source_code)
        if not text_content or not text_content.strip():
            return None

        metadata = {
            "file_path": self.filename,
            "start_position": self.start,
            "end_position": self.end,
            "content_length": len(self),
            "programming_language": self.language,
            "code_type": "unknown",
        }

        # Add extra_info to metadata if provided
        if extra_info:
            metadata.update(extra_info)

        # Extract scope parts for metadata fields
        if self.scope:
            scope_parts = self.scope.split("::")
            metadata.update(self._process_scope_metadata(scope_parts, text_content))
        else:
            metadata.update({"code_type": "global_code", "fully_qualified_name": "global"})

        # Add content summary
        metadata["content_summary"] = self._generate_content_summary(metadata)

        return Document(
            text=text_content,
            metadata=metadata,
            id_=f"{self.filename}::{self.start}-{self.end}",
            excluded_llm_metadata_keys=[
                "start_position",
                "end_position",
                "content_length",
                "token_count",
            ],  # LLM does not need this
        )

    def _process_scope_metadata(self, scope_parts: List[str], text_content: str) -> Dict[str, str]:
        """Process scope parts into metadata."""
        metadata = {}
        namespace_parts = []

        for part in scope_parts:
            curr_namespace = part if not namespace_parts else f"{namespace_parts[-1]}::{part}"
            namespace_parts.append(curr_namespace)

        if len(scope_parts) >= 3:
            metadata.update(
                {
                    "code_type": "method",
                    "namespace": namespace_parts[-3],
                    "class_name": scope_parts[-2],
                    "method_name": scope_parts[-1],
                    "fully_qualified_name": self.scope,
                }
            )
        elif len(scope_parts) == 2:
            if "class_" in text_content[:50].lower() or "struct" in text_content[:50].lower():
                metadata.update(
                    {
                        "code_type": "class",
                        "namespace": namespace_parts[0],
                        "class_name": scope_parts[-1],
                        "fully_qualified_name": self.scope,
                    }
                )
            else:
                metadata.update(
                    {
                        "code_type": "function",
                        "namespace": namespace_parts[0],
                        "function_name": scope_parts[-1],
                        "fully_qualified_name": self.scope,
                    }
                )
        elif len(scope_parts) == 1:
            if "namespace" in text_content[:50].lower():
                metadata.update(
                    {"code_type": "namespace", "namespace": scope_parts[0], "fully_qualified_name": scope_parts[0]}
                )
            else:
                metadata.update({"code_type": "global_code", "fully_qualified_name": scope_parts[0]})

        return metadata

    def _generate_content_summary(self, metadata: Dict[str, str]) -> str:
        """Generate content summary from metadata."""
        summary = f"{metadata['code_type'].title()} in {metadata.get('namespace', 'global scope')}"
        if "class_name" in metadata:
            summary += f", class {metadata['class_name']}"
        if "method_name" in metadata:
            summary += f", method {metadata['method_name']}"
        elif "function_name" in metadata:
            summary += f", function {metadata['function_name']}"
        return summary

    def set_scope(self, namespaces: List[str], class_name: str = "", function_name: str = "") -> None:
        """Set the fully qualified scope."""
        # Filter out empty strings to avoid empty elements
        scope_parts = (
            [part for part in namespaces if part]
            + ([class_name] if class_name else [])
            + ([function_name] if function_name else [])
        )

        # Join non-empty parts with :: separator
        self.scope = "::".join(scope_parts) if scope_parts else ""
