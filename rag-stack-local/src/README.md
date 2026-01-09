# RAG (Retrieval-Augmented Generation) Toolkit

RAG toolkit for processing and embedding various document types including code, PDFs, and markdown files.

## Installation

```bash
# Clone the repository
git clone git@drm-gitlab.redlabs.pl:zos/chat-ai-deployment.git
cd chat-ai-deployment/src

# Install required packages
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm
```

## Development Setup

To run the code directly, you'll need to add the project's `src` directory to your PYTHONPATH:

```bash
# One-time usage
PYTHONPATH=/path/to/project/src python your_script.py

# Or add to your shell configuration (~/.bashrc):
export PYTHONPATH="${PYTHONPATH}:/path/to/project/src"
```

## Project Structure

```
src/
└── rag/
    ├── readers/        # Specialized readers (pdf, markdown, code)
    ├── query/          # Query and chat interfaces
    └── retriever/      # Retrievers
    └── samples/        # Test samples
```

## Document Types Support

### C++ Code
- Parsing source files (generated via repomix)
- Namespace, function and class extraction

### PDF Documents
- Grobid integration
- Semantic splitting
- Structure preservation
- Text normalization

### Markdown
- Section parsing
- Format preservation
- Text normalization
