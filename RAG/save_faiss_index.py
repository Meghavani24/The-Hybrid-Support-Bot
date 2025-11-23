import faiss
import numpy as np
import pickle

def save_faiss_index(chunks, index_path="amaze_index.faiss", meta_path="amaze_meta.pkl"):
    vectors = []
    metadata = []

    for item in chunks:
        vectors.append(item["embedding"])
        metadata.append({
            "chapter": item["metadata"]["chapter"],
            "chunk_index": item["metadata"]["chunk_index"],
            "page_start": item["metadata"]["page_start"],
            "page_end": item["metadata"]["page_end"],
            "embedding": item["embedding"].tolist(),
            "text": item["text"]
        })

    vectors = np.array(vectors).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print("FAISS index saved:", index_path)
    print("Full metadata saved:", meta_path)

