import re

# A period with a digit on both sides is a decimal point. Other ., !, and ?
# runs are sentence boundaries.
_SENTENCE_BOUNDARY = re.compile(r"((?<!\d)[.!?]+|[.!?]+(?!\d))")


def chunk_graphemes(graphemes: str, chunk_size: int = 400) -> list[str]:
    """Split text into chunks of about ``chunk_size`` characters.

    Sentence punctuation is the preferred cut. ``3.14`` stays in one chunk.
    A sentence-final period still starts a new chunk when the next sentence
    does not fit.
    """
    sentences = _SENTENCE_BOUNDARY.split(graphemes)
    current_chunk = ""
    chunks: list[str] = []

    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]

        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    if not chunks:
        chunks = [graphemes[i:i + chunk_size] for i in range(0, len(graphemes), chunk_size)]
    return chunks
