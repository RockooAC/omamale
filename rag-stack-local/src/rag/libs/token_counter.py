import token
from logging import Logger
from src.rag.common import setup_logger
from transformers import AutoTokenizer
class TokenCountAnnotator:
    def __init__(self, model_name: str, debug: bool = False, logger: Logger = None):
        self.logger = logger or setup_logger("TokenCountAnnotator", debug=debug)
        self.model_name = self._strip_tag(model_name)
        self.tokenizer = self._load_tokenizer()

    def _strip_tag(self, name: str) -> str:
        """Model name should have not contains ':latest' but anyway is checked"""
        return name.partition(":")[0]

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer for model '{self.model_name}': {e}")
            raise

    def annotate(self, documents):
        for doc in documents:
            try:
                meta_text = " ".join(str(v) for v in doc.metadata.values()) if doc.metadata else ""
                full_text = f"{doc.text} {meta_text}".strip()

                tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
                token_count = len(tokens)

                doc.metadata["token_count"] = token_count

                if "token_count" not in doc.metadata:
                    raise ValueError(f"token_count not added to doc {doc.id_}")

                self.logger.debug(f"Final metadata for {doc.id_}: {doc.metadata}")
                self.logger.info(f"Added {token_count} token count to doc {doc.id_}")
            except Exception as e:
                self.logger.error(f"Error processing doc {doc.id_}: {e}")
        return documents
