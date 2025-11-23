📌Project Title — Hybrid Support Bot (Advanced RAG)
🧾 Task Chosen

Option 1: The “Hybrid” Support Bot (Advanced RAG)
A Retrieval-Augmented Generation (RAG) system that answers user queries based on a technical PDF manual, using metadata filtering, hybrid vector search, and latency logging to improve accuracy.

⚙️ Local Setup & Running Instructions

1. Clone the Repository

git clone <your_repo_link_here>
cd <repo_folder>

2.. Install Dependencies

pip install -r requirements.txt

3.Start Ollama (for Local Llama Model)

ollama pull llama3.2

4.Run the Ingestion Pipeline (One-time only)

This step extracts chapters, chunks the data, embeds, and builds the FAISS index.

python chapter_extractor.py
python chunk_and_embed.py
python save_faiss_index.py


5. Run Query API

Open Jupyter Notebook or Python shell:

from rag_query import answer_query
answer_query("How do I adjust the seats?", chapter="Before Driving")


📌 Why These Libraries & Models Were Chosen:

| Component      | Choice                                     | Reason                                                                       |
| -------------- | ------------------------------------------ | ---------------------------------------------------------------------------- |
| Embeddings     | `sentence-transformers / all-MiniLM-L6-v2` | Fast, lightweight, high accuracy for semantic search                         |
| Vector DB      | FAISS                                      | Highly optimized similarity search with low retrieval latency                |
| LLM            | `Llama 3.2` via local Ollama               | Runs offline, no API costs, deterministic + fast, allowed for the assignment |
| PDF Parser     | PyMuPDF (`fitz`)                           | High fidelity text extraction from structured manuals                        |
| Numpy / Pickle | Array storage + metadata persistence       | Efficient & minimal overhead                                                 |


📂 Repository Structure:

project/
│
├─ data/
│   └─ honda_manual.pdf
│
├─ index/
│   └─ amaze_index.faiss
│   └─ amaze_meta.pkl
│
├─ rag_emb.py        → Embedding model
├─ rag_ingest.py     → PDF ingestion + chunking
├─ rag_index.py      → Build index (semantic + BM25)
├─ rag_query.py      → Hybrid search + LLM + threshold + logging
│
├─ requirements.txt
└─ README.md
