import time
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# -----------------------------
# Load embedding model
# -----------------------------
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# FAISS + metadata paths
# -----------------------------
INDEX_PATH = "amaze_index.faiss"
META_PATH = "amaze_meta.pkl"

# -----------------------------
# Load FAISS + metadata
# -----------------------------
def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# -----------------------------
# Vector search
# -----------------------------
SIMILARITY_THRESHOLD = 0.40   # LOWER = more tolerant, HIGHER = stricter

def vector_search(query, chapter_filter=None, top_k=5):
    index, metadata = load_index()
    q_emb = EMBED_MODEL.encode([query]).astype("float32")

    # Search full index
    distances, inds = index.search(q_emb, top_k)
    inds = inds[0]
    distances = distances[0]

    # Convert FAISS distances to cosine-like similarity
    sim = 1 / (1 + distances)

    results = []
    for i, s in zip(inds, sim):
        m = metadata[i]
        m["similarity"] = float(s)
        results.append(m)

    # chapter filtering (optional)
    if chapter_filter:
        results = [r for r in results if r["chapter"].lower() == chapter_filter.lower()]

    # apply similarity threshold
    results = [r for r in results if r["similarity"] >= SIMILARITY_THRESHOLD]

    return results


# -----------------------------
# Llama call (SHORT ANSWER ENFORCED)
# -----------------------------
def run_llama(context, query):
    prompt = f"""
You can ONLY answer using the following manual context.
If the answer is not found EXACTLY in the context, reply strictly: I DON'T KNOW.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:
"""

    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "temperature": 0.0,
        "stream": False
    }

    r = requests.post(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )

    response = r.json()
    return response.get("response", "").strip()


# -----------------------------
# Main query function
# -----------------------------
def answer_query(query, chapter=None):
    start = time.time()
    results = vector_search(query, chapter_filter=chapter)
    retrieval_time = time.time() - start

    # No chunk passes threshold → answer should be "I DON'T KNOW"
    if len(results) == 0:
        return {
            "answer": "I DON'T KNOW",
            "chunks_used": [],
            "retrieval_latency": retrieval_time,
            "generation_latency": 0
        }

    # build llama context from retrieved text only
    context = "\n\n".join(
        [f"[{r['chapter']} p.{r['page_start']}] {r['text']}" for r in results]
    )

    t0 = time.time()
    answer = run_llama(context, query)
    gen_latency = time.time() - t0

    return {
        "answer": answer,
        "chunks_used": results,
        "retrieval_latency": retrieval_time,
        "generation_latency": gen_latency
    }
