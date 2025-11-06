from src.ingest import chunk_text


def test_chunk_text_basic():
    text = "a" * 2500
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    # Expect roughly ceil((2500 - 200) / (1000 - 200)) chunks -> 4
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)
