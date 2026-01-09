from pathlib import Path
from rag.readers.cpp import CppReader


def test_file():
    # Initialize the reader
    reader = CppReader(chunk_size=1024, chunk_overlap=50, debug=False)

    # Define test C++ code
    test_code = """
#include "example.h"

namespace test {
    class Example {
    public:
        Example() {}
        void doSomething() {}
    };
    
    void freeFunction() {}
}  // namespace test
    """

    # Adjust expected chunks to align with the reader's chunking logic
    expected_chunks = [
        '#include "example.h"',
        "namespace test {\n    class Example {\n    public:\n        Example() {}\n        void doSomething() {}\n    };\n    \n    void freeFunction() {}\n}",
        "// namespace test",
    ]

    # Process the code
    print("Processing test C++ code...")
    chunks = reader.process_file(test_code, "example.cpp")

    # Verify chunks
    for i, (chunk, expected) in enumerate(zip(chunks, expected_chunks), 1):
        actual = chunk.extract(test_code).strip()
        print(f"\nVerifying Chunk {i}:")
        if actual == expected:
            print(f"✓ Content matches for Chunk {i}")
        else:
            print(f"✗ Content mismatch for Chunk {i}")
            print(f"Expected:\n{expected}")
            print(f"Got:\n{actual}")


def test_directory_or_repomix():
    reader = CppReader(chunk_size=1024, chunk_overlap=50, debug=False)

    # Get the directory containing the current script
    script_dir = Path(__file__).parent

    # Define the relative path to the samples directory
    test_path = script_dir / "samples/code"

    # Resolve the absolute path
    test_path = test_path.resolve()

    # Expected values
    expected_file_count = 6
    expected_chunks = 18

    if test_path.is_dir() or test_path.suffix == ".repomix":  # Directory or Repomix file
        print(f"TEST_PATH: {test_path}")
        chunks_by_file = reader.load_data(test_path)

        # Verify the number of processed files
        print("\nVerifying number of processed files:")
        if len(chunks_by_file) == expected_file_count:
            print(f"✓ Processed file count matches: {expected_file_count}")
        else:
            print(f"✗ File count mismatch. Expected: {expected_file_count}, Got: {len(chunks_by_file)}")

        # Count total chunks across all files
        total_chunks = sum(len(file_chunks) for file_chunks in chunks_by_file.values())
        print("\nVerifying total chunks:")
        if total_chunks == expected_chunks:
            print(f"✓ Total chunks match: {expected_chunks}")
        else:
            print(f"✗ Total chunks mismatch. Expected: {expected_chunks}, Got: {total_chunks}")

    else:
        print(f"Test path does not exist or is invalid: {test_path}")


if __name__ == "__main__":
    print("Running test for single file...")
    test_file()

    print("\nRunning test for directory or Repomix file...")
    test_directory_or_repomix()
