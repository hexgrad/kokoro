import importlib.util
from pathlib import Path


def _chunk_graphemes():
    path = Path(__file__).resolve().parents[1] / "kokoro" / "chunks.py"
    spec = importlib.util.spec_from_file_location("kokoro_chunks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.chunk_graphemes


def test_decimal_point_stays_in_one_chunk():
    chunk_graphemes = _chunk_graphemes()
    text = ("a" * 396) + "3.14" + ("b" * 20)
    parts = chunk_graphemes(text)
    assert parts == [text]
    assert "3.14" in parts[0]


def test_sentence_final_period_still_splits():
    chunk_graphemes = _chunk_graphemes()
    text = ("a" * 390) + ". " + ("b" * 30)
    parts = chunk_graphemes(text)
    assert parts[0].endswith(".")
    assert parts[1] == "b" * 30


def test_sentence_final_period_after_a_decimal_keeps_the_number():
    chunk_graphemes = _chunk_graphemes()
    text = ("a" * 390) + "3.14. " + ("b" * 30)
    parts = chunk_graphemes(text)
    assert parts[0].endswith("3.14.")
    assert not parts[0].endswith("3.")
    assert parts[1] == "b" * 30


def test_short_decimal_stays_one_chunk():
    chunk_graphemes = _chunk_graphemes()
    assert chunk_graphemes("3.14 is pi.") == ["3.14 is pi."]
