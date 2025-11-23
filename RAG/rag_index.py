import faiss
import pickle
import numpy as np

INDEX_PATH = "amaze_index.faiss"
META_PATH = "amaze_meta.pkl"

def load_index():
    # Load FAISS index or rebuild if needed
    index = faiss.read_index(INDEX_PATH)

    # Load metadata
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    # Extract text and embeddings
    data = [m["text"] for m in meta]
    embeddings = [m["embedding"] for m in meta]

    return data, np.array(embeddings), meta
