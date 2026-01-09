from pathlib import Path
from rag.readers.pdf import GrobidPDFReader
from rag.config import GROBID_CONFIG

GROBID_BASE_URL = GROBID_CONFIG["BASE_URL"]
GROBID_SPLIT_SENTENCE = GROBID_CONFIG["SPLIT_SENTENCE"]
GROBID_LOAD_FIGURES = GROBID_CONFIG["LOAD_FIGURES"]

EXPECTED_CHUNKS_LENGTH = 46


def test_file():
    reader = GrobidPDFReader(
        grobid_server=GROBID_BASE_URL,
        split_sentence=GROBID_SPLIT_SENTENCE,
        load_figures=GROBID_LOAD_FIGURES,
        normalize=True,
        debug=False,
    )

    # Path to the single PDF file
    test_file_path = Path(__file__).parent / "samples/pdf/fairplay.pdf"
    test_file_path = test_file_path.resolve()

    if test_file_path.is_file():
        documents = reader.load_file(test_file_path, extra_info={"source": "single_test"})
        total_chunks = len(documents)

        print(f"\nVerifying chunks from single file: {test_file_path.name}\n")
        if total_chunks == EXPECTED_CHUNKS_LENGTH:
            print(f"✓ Content matches expected chunks length")
        else:
            print(f"✗ Content mismatch in chunks length")
            print(f"Expected: {EXPECTED_CHUNKS_LENGTH}")
            print(f"Got: {total_chunks}")

    else:
        print(f"Test file does not exist: {test_file_path}")


def test_directory():
    reader = GrobidPDFReader(
        grobid_server=GROBID_BASE_URL,
        split_sentence=GROBID_SPLIT_SENTENCE,
        load_figures=GROBID_LOAD_FIGURES,
        normalize=True,
        debug=False,
    )

    # Get the directory containing the current script
    script_dir = Path(__file__).parent

    # Define the relative path to the samples directory
    test_path = script_dir / "samples/pdf"

    # Resolve the absolute path
    test_path = test_path.resolve()

    if test_path.is_dir():
        docs_by_file = reader.load_data(test_path, extra_info={"source": "test_batch"})
        total_chunks = sum(len(file_chunks) for file_chunks in docs_by_file.values())

        print(f"\nVerifying chunks from directory processing:\n")
        if total_chunks == EXPECTED_CHUNKS_LENGTH:
            print(f"✓ Content matches expected chunks length")
        else:
            print(f"✗ Content mismatch in chunks length")
            print(f"Expected: {EXPECTED_CHUNKS_LENGTH}")
            print(f"Got: {total_chunks}")

    else:
        print(f"Test directory does not exist: {test_path}")


if __name__ == "__main__":
    test_file()
    test_directory()
