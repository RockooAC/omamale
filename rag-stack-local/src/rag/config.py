import os

EMBEDDER_CONFIG = {
    "BASE_URL": "http://10.255.240.149:11434",  # Default embedder on c24 server
    "KS_BASE_URL": "http://10.255.240.161:11434",  # Embedder on ks server
    "MODELS": {
        "JINA": {  # Best for code embedding
            "ollama": "jina/jina-embeddings-v2-base-en:latest",
            "hf": "jinaai/jina-embeddings-v2-base-en",  # HuggingFace
        },
        "QWEN": {  # Best for textual content
            "ollama": "gte-qwen2.5-instruct-q5",
            "hf": "Alibaba-NLP/gte-Qwen2-7B-instruct",  # HuggingFace
        },
    },
    "VECTOR_SIZES": {
        768: "JINA",
        3584: "QWEN",
    },
}

QDRANT_CONFIG = {
    "BASE_URL": "http://10.255.240.18:6333",  # Qdrant server
    "BASE_URL_LOCAL": "http://localhost:6333",  # Local Qdrant server
}

LLM_CONFIG = {
    "BASE_URL": "http://10.255.240.156:11434",
    "DEFAULT_MODEL": "llama3.1:latest",
    "CODE_MODEL": "deepseek-coder-v2:latest",
    "CONTEXT_WINDOW": 96000,
}

REDIS_CONFIG = {
    "HOST": "10.255.240.18",
    "PORT": 6379,
}

CHUNK_CONFIG = {
    "SIZE": 1024,
    "OVERLAP": 128,
}

GROBID_CONFIG = {
    "ENABLED": True,
    "BASE_URL": "http://10.255.240.18:8070",
    "SPLIT_SENTENCE": True,
    "LOAD_FIGURES": True,
}

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

EXTERNAL_URLS = {
    "CONFLUENCE": "https://confluence.redge.com",
}

REPOSITORY_CONFIG = {
    "EXCLUDE_PATTERNS": {
        "*.livx",
        "*.isml",
        "*.pem",
        "*.der",
        "*.key",
        "*.inc",
        "*.pyc",
        "*.pyo",
        "*.o",
        "*.a",
        "*.so",
        "*.dll",
        "*.git",
        ".gitignore",
        ".gitmodules",
        ".svn/*",
        "*.sln",
        "*.sql",
        "*.sh",
        "*.bat",
        "*.in",
        "*.dox",
        "*.filters",
        "*.vcxproj",
        "*/.git/*",
        "*/.gitignore",
        "*/.gitmodules",
        "*/buildtools/build.sh",
        "*/buildtools/.git",
        "*/buildtools/build.py",
        "makefile_*.in",
        "*.template",
        "*.DS_Store",
        "*.log",
        "*.tmp",
        "*.xml",
        "*.m3u8",
        "*.json",
    },
    "TARGET_DIRS": {
        "src/dbir": "dbir.digest",
        "src/rg/cdn": "cdn.digest",
        "src/rg/coder": "coder.digest",
        "src/rg/core": "core.digest",
        "src/rg/cuda": "cuda.digest",
        "src/rg/opencv": "opencv.digest",
        "src/tools": "tools.digest",
    },
    "MAX_WORKERS": 8,
    "CHUNK_SIZE": 1024,
}
