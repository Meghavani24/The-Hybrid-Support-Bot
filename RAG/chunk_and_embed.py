from sentence_transformers import SentenceTransformer
import uuid
from chapter_extractor import extract_chapter_text

# Load embedding model
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, max_tokens=350, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks


def create_chunks_with_metadata(chapters):
    all_chunks = []

    for ch in chapters:
        title = ch["chapter"]
        page_start = ch["page_start"]
        page_end = ch["page_end"]
        text = ch["text"]

        chunks = chunk_text(text)

        # FIXED: rename variable
        for idx, chunk_text_block in enumerate(chunks):
            embedding = EMBED_MODEL.encode(chunk_text_block)

            metadata = {
                "id": str(uuid.uuid4()),
                "chapter": title,
                "chunk_index": idx,
                "page_start": page_start,
                "page_end": page_end
            }

            all_chunks.append({
                "text": chunk_text_block,
                "embedding": embedding,
                "metadata": metadata
            })

    return all_chunks


if __name__ == "__main__":
    chapters = extract_chapter_text()
    chunks = create_chunks_with_metadata(chapters)

    print("Total chunks:", len(chunks))
    print("Sample metadata:", chunks[0]["metadata"])
    print("Sample text:", chunks[0]["text"][:200], "...")