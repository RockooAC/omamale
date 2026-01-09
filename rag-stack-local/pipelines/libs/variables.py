from libs.tools import ObjectRepository

DEFAULT_SENTENCE_TRANSFORMER_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL = "prithivida/Splade_PP_en_v1"
DEFAULT_REQUEST_TIMEOUT = 120

# Create a global instance of ObjectRepository.
GlobalRepository = ObjectRepository()