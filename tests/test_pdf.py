from src.ingest import extract_text_from_pdf
import os

def test_pdf():
    # Get the full path to the test.pdf file in the tests directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'test.pdf')
    assert(extract_text_from_pdf(path))
