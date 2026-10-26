# Ask PLFS - Retrieval-Augmented Generation (RAG) System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ask-plfs-by-mani.streamlit.app/)

**Ask PLFS** is an intelligent Q&A application designed to answer questions about India's Periodic Labour Force Survey (PLFS). It uses a Retrieval-Augmented Generation (RAG) pipeline to accurately extract, retrieve, and synthesize information directly from the official PLFS Annual Reports, complete with page-level citations.

**🌟 Live Application:** [https://ask-plfs-by-mani.streamlit.app/](https://ask-plfs-by-mani.streamlit.app/)

---

## 📄 Reference Material

For a detailed walkthrough of the original Google Colab implementation, pipeline testing, and code structure, please refer to the project's primary reference file: **ask plfs final version.pdf**.

---

## ✨ Key Features

*   **Hybrid Search Retrieval:** Combines Dense vector search (FAISS) and Sparse keyword search (BM25) using Reciprocal Rank Fusion (RRF) for highly accurate document retrieval.
*   **Intelligent Chunking:** Uses token-aware chunking (256 token limit, 40 token overlap) that respects table structures and markdown formatting so that context isn't lost mid-sentence or mid-table.
*   **Strict Fact-Grounding:** The generation prompt strictly limits the LLM (Gemini) to the retrieved passages. It is designed to say "I could not find this in the provided PLFS reports" if the answer isn't present, preventing hallucinations.
*   **Page Citations:** Every answer cites the specific PDF page number where the information was found.
*   **Data Cleaning:** Custom ingestion pipeline that filters out non-English (e.g., Hindi) text, removes repeated headers/footers, and handles nested table structures using `PyMuPDF4LLM`.

---

## 🛠️ Technology Stack

*   **Frontend:** Streamlit
*   **PDF Processing:** PyMuPDF, PyMuPDF4LLM
*   **Embeddings:** Sentence-Transformers (`BAAI/bge-small-en-v1.5`)
*   **Vector Database (Dense):** FAISS (`faiss-cpu`)
*   **Sparse Retrieval:** BM25 (`rank-bm25`)
*   **LLM Generator:** Google GenAI / Gemini Flash models (`gemini-3.1-flash-lite`, `gemini-3.8-flash`)
*   **Data Handling:** Pandas, NumPy

---

## 📁 Project Structure

Based on the implementation detailed in `ask plfs final version.pdf`, the core backend logic is modularized as follows:

*   `src/config.py`: Manages file paths, corpus metadata, chunking constraints, and model names.
*   `src/utils.py`: Helper functions for reading and writing `.jsonl` files.
*   `src/ingest.py`: Downloads reports, extracts text using PyMuPDF, cleans markdown, and filters out garbled or low-English-ratio pages.
*   `src/chunking.py`: Implements token-level chunking that preserves table headers for continuity.
*   `src/indexing.py`: Embeds chunks and builds the FAISS vector index.
*   `src/retrieval.py`: Handles dense, sparse, and hybrid queries, with logic to limit the number of chunks retrieved from a single page.
*   `src/generation.py`: Interfaces with the Gemini API using strict system instructions to enforce factual answers and citations.
*   `src/pipeline.py`: The main orchestrator that glues the retriever and generator together.
*   `src/evaluation.py`: Calculates retrieval metrics (Hit Rate, MRR) against a set of expected ground-truth pages.

---

## 🚀 Local Setup & Installation

If you wish to run the pipeline or Streamlit app locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd ask-plfs
   ```

2. **Install dependencies:**
   ```bash
   pip install pymupdf pymupdf4llm sentence-transformers faiss-cpu rank-bm25 google-genai pandas streamlit
   ```

3. **Set your API Key:**
   Export your Google Gemini API key as an environment variable:
   ```bash
   export GeminiAPIkey="your_api_key_here"
   ```

4. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

*(Note: Initial runs will require time to download the PLFS report, generate embeddings, and build the FAISS index as defined in the ingest and indexing scripts).*

---
*Built by Mani.*