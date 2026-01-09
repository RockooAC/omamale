from pathlib import Path
from rag.readers.markdown import MarkdownReader

EXPECTED_CHUNKS_LENGTH = 8


def test_file():
    reader = MarkdownReader(chunk_size=512, chunk_overlap=0, debug=False, normalize=False)

    test_content = """
# Simple Markdown Example

## Section 1: Introduction
This is a simple markdown file for testing purposes. It contains multiple sections with plain text.

## Section 2: Key Points
- Point 1: This is the first key point.
- Point 2: This is the second key point.
- Point 3: This is the third key point.

## Section 3: Conclusion
In conclusion, this markdown file helps verify if the chunking mechanism is working correctly.
    """

    expected_chunks = [
        "# Simple Markdown Example",
        "## Section 1: Introduction\nThis is a simple markdown file for testing purposes. It contains multiple sections with plain text.",
        "## Section 2: Key Points\n- Point 1: This is the first key point.\n- Point 2: This is the second key point.\n- Point 3: This is the third key point.",
        "## Section 3: Conclusion\nIn conclusion, this markdown file helps verify if the chunking mechanism is working correctly.",
    ]

    print("Processing simple test markdown content...")
    documents = reader.process_text(test_content.strip())

    for i, (doc, expected) in enumerate(zip(documents, expected_chunks), 1):
        print(f"\nVerifying Chunk {i}:")
        if doc.text.strip() == expected.strip():
            print(f"✓ Content matches for Chunk {i}")
        else:
            print(f"✗ Content mismatch for Chunk {i}")
            print(f"Expected:\n{expected}")
            print(f"Got:\n{doc.text}")


def test_directory():
    reader = MarkdownReader(chunk_size=512, chunk_overlap=0, debug=False, normalize=False)

    # Get the directory containing the current script
    script_dir = Path(__file__).parent

    # Define the relative path to the samples directory
    test_path = script_dir / "samples/markdown"

    # Resolve the absolute path
    test_path = test_path.resolve()

    if test_path.is_dir():
        docs_by_file = reader.load_data(test_path)
        total_chunks = sum(len(file_chunks) for file_chunks in docs_by_file.values())

        print(f"\nVerifying Chunks from directory processing:\n")
        if total_chunks == EXPECTED_CHUNKS_LENGTH:
            print(f"✓ Content matches expected chunks length")
        else:
            print(f"✗ Content mismatch in chunks length")
            print(f"Expected: {EXPECTED_CHUNKS_LENGTH}")
            print(f"Got: {total_chunks}")
    else:
        print(f"Test path does not exist or is invalid: {test_path}")


if __name__ == "__main__":
    test_file()
    test_directory()
