# RAGScope: Local Agentic RAG CV–Job Matcher

RAGScope is a local AI application that compares a CV with a job description using Retrieval-Augmented Generation, embeddings, vector search, Ollama, and a lightweight LangGraph agent workflow.

## Features

- Upload CV PDF and job description TXT files
- Extract and clean document text
- Chunk documents into retrievable passages
- Create embeddings using MiniLM
- Store and search chunks using FAISS
- Generate local reports using Ollama
- Use LangGraph agents for:
  - job requirement extraction
  - CV evidence review
  - gap analysis
  - report generation
  - unsupported-claim checking
- Filter unsupported metrics and fabricated CV claims
- Generate safe future CV bullet ideas

## Tech Stack

- Python
- Streamlit
- FAISS
- sentence-transformers
- Ollama
- LangGraph
- PyPDF
- Requests

## Run Locally

Install dependencies:

```bash
pip install streamlit pypdf pandas numpy sentence-transformers faiss-cpu requests langgraph