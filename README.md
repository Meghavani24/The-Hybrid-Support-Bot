RAG System for Vehicle Owner Manual 
📌 Task Chosen

I selected Task 1 – Retrieval-Augmented Generation (RAG) System.
The goal was to build an intelligent question-answering system over a Honda Vehicle Owner Manual PDF, using:

Smart document ingestion and chunking

Hybrid search (vector + keyword filtering)

Performance logging

Local LLM inference

Project Overview

Users can ask natural-language questions about the manual (e.g., “How do I adjust the seats?”), and the system retrieves the most relevant manual sections using FAISS vector search + chapter filtering, then generates a final answer using a local Llama model via Ollama.

The system refuses out-of-scope queries (e.g., “How do I bake a cake?”) through similarity-threshold filtering

📂 Project Structure

📦 rag-vehicle-manual-bot
 ┣ data/                          # Original manual PDF
 ┣ index/                         # FAISS index + metadata + embeddings
 ┣ chapter_extractor.py           # Extract chapter-level text from PDF
 ┣ chunk_and_embed.py             # Create chunks + generate embeddings + save FAISS index
 ┣ save_faiss_index.py            # Index persistence utilities
 ┣ rag_query.py                   # Main RAG pipeline + threshold filtering + latency logging
 ┣ requirements.txt
 ┣ environment.yml
 ┗ README.md

 ⚙️ Setup & Run Locally
1️⃣ Install Ollama and pull the model

https://ollama.com/download
ollama pull llama3.2


2️⃣ Create and activate the environment
With Conda:

conda env create -f environment.yml
conda activate ragbot

With pip:

pip install -r requirements.txt

3️⃣ Run the ingestion pipeline (first time only)

python chapter_extractor.py
python chunk_and_embed.py

4️⃣ Run a query
Open Python shell or notebook:

from rag_query import answer_query
answer_query("How do I adjust the seats?", chapter="Before Driving")

🧾 Example Output:

Query: "How do I adjust the seats?"

Answer:
To adjust the seats:
* Lever Height Adjustment (Driver’s seat only): Pull up or push down the lever...
* Lever Seat-back Angle Adjustment: Pull up the lever to change the angle...
* Bar Horizontal Adjustment: Pull up on the bar to move the seat...

Chunks Used: 2
Retrieval Latency: 0.128 s
Generation Latency: 148.4 s

Query: “How do I bake a cake?”

Answer: I DON’T KNOW
Reason: Query similarity below threshold → out-of-scope rejection

🔍 Why these libraries/models?

| Component   | Choice                  | Reason                                                |
| ----------- | ----------------------- | ----------------------------------------------------- |
| Embeddings  | `sentence-transformers` | High-quality semantic embeddings for long documents   |
| Vector DB   | `faiss-cpu`             | Fast & scalable similarity search                     |
| PDF Parsing | `pymupdf`               | Precise layout-aware extraction                       |
| Local LLM   | `llama3.2` via Ollama   | Runs offline, stable responses, avoids API dependency |
| Logging     | `time`                  | Lightweight latency tracking                          |


This combination provides:

Fast retrieval

Edge-friendly deployment (works offline)

Deterministic evaluation (no API drift)

📸 Demo Requirements (Evaluator)

A demo video/screenshots folder should show:

Query inside scope (correct result)

Low-similarity rejection ("I DON’T KNOW")

Logged latencies (retrieval + generation)

🧾 Reproducibility

This project includes:

requirements.txt

environment.yml

Allowing the evaluator to reproduce the environment using pip or Conda.
