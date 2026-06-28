import re
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="RAGScope - RAG Evidence Matcher",
    page_icon="🧠",
    layout="wide"
)


# -------------------------------------------------
# Text extraction
# -------------------------------------------------
def extract_text_from_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def extract_text_from_txt(uploaded_txt):
    return uploaded_txt.read().decode("utf-8", errors="ignore").strip()


# -------------------------------------------------
# Basic text cleaning
# -------------------------------------------------
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------------------------------
# Chunking
# -------------------------------------------------
def chunk_text(text, source_name, chunk_size=700, overlap=150):
    """
    Splits long text into overlapping chunks.
    Overlap helps preserve meaning between chunk boundaries.
    """
    text = clean_text(text)

    chunks = []
    start = 0
    chunk_id = 1

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(
                {
                    "chunk_id": f"{source_name}_{chunk_id}",
                    "source": source_name,
                    "text": chunk.strip(),
                }
            )

        start += chunk_size - overlap
        chunk_id += 1

    return chunks


# -------------------------------------------------
# Embedding model
# -------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """
    all-MiniLM-L6-v2 is lightweight and works well on normal laptops.
    First run may download the model.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model


def create_embeddings(model, texts):
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    return embeddings


# -------------------------------------------------
# FAISS vector store
# -------------------------------------------------
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def search_similar_chunks(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]

        results.append(
            {
                "score": float(score),
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            }
        )

    return results


# -------------------------------------------------
# Skill matching
# -------------------------------------------------
SKILL_ALIASES = {
    "python": ["python"],
    "sql": ["sql"],
    "git": ["git", "github"],
    "fastapi": ["fastapi"],
    "streamlit": ["streamlit"],
    "docker": ["docker"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow"],
    "transformers": ["transformers", "hugging face", "huggingface"],
    "llm": ["llm", "large language model", "large language models"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "embeddings": ["embedding", "embeddings", "sentence embeddings", "text embeddings"],
    "vector database": ["vector database", "vector db", "faiss", "chroma", "qdrant", "pgvector"],
    "langchain": ["langchain"],
    "langgraph": ["langgraph"],
    "agentic ai": ["agentic ai", "ai agent", "ai agents", "agent workflow", "tool calling"],
    "prompt engineering": ["prompt engineering", "prompting"],
    "nlp": ["nlp", "natural language processing"],
    "evaluation": ["evaluation", "model evaluation", "rag evaluation", "llm evaluation"],
    "api": ["api", "rest api", "rest apis"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural network", "neural networks"],
}


def normalize_for_skills(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def find_skills(text):
    normalized = normalize_for_skills(text)
    found = set()

    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in normalized:
                found.add(skill)
                break

    return sorted(found)


def calculate_skill_match(cv_text, jd_text):
    cv_skills = set(find_skills(cv_text))
    jd_skills = set(find_skills(jd_text))

    matched = sorted(cv_skills.intersection(jd_skills))
    missing = sorted(jd_skills.difference(cv_skills))
    extra = sorted(cv_skills.difference(jd_skills))

    score = 0
    if len(jd_skills) > 0:
        score = round((len(matched) / len(jd_skills)) * 100, 2)

    return {
        "cv_skills": sorted(cv_skills),
        "jd_skills": sorted(jd_skills),
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "score": score,
    }


# -------------------------------------------------
# Report generation without LLM
# -------------------------------------------------
def generate_simple_report(skill_results, evidence_results):
    report_lines = []

    report_lines.append("# RAGScope CV–Job Match Report\n")
    report_lines.append(f"## Overall Skill Match Score: {skill_results['score']}%\n")

    report_lines.append("## Matched Skills\n")
    if skill_results["matched"]:
        for skill in skill_results["matched"]:
            report_lines.append(f"- {skill}")
    else:
        report_lines.append("- No matched skills detected.")

    report_lines.append("\n## Missing or Weak Skills\n")
    if skill_results["missing"]:
        for skill in skill_results["missing"]:
            report_lines.append(f"- {skill}")
    else:
        report_lines.append("- No missing skills detected from the known skill list.")

    report_lines.append("\n## Retrieved Evidence Chunks\n")
    for item in evidence_results:
        report_lines.append(
            f"\n### {item['chunk_id']} | Source: {item['source']} | Similarity: {item['score']:.3f}\n"
        )
        report_lines.append(item["text"])

    report_lines.append("\n## Suggested Project Improvement\n")
    report_lines.append(
        "Build and document an Agentic RAG project with retrieval, embeddings, vector search, "
        "job-description analysis, evidence-based scoring, and evaluation. Add screenshots, "
        "sample reports, and a clear README to make it recruiter-ready."
    )

    return "\n".join(report_lines)


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("🧠 RAGScope: RAG Evidence Matcher")
st.write(
    "Upload your CV and one job description. This version uses embeddings and FAISS "
    "to retrieve evidence from your documents."
)

st.info(
    "Stage 2 goal: move beyond keyword matching and build the retrieval layer of RAG."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload CV PDF")
    cv_file = st.file_uploader("Upload your CV PDF", type=["pdf"])

with col2:
    st.subheader("2. Upload Job Description")
    jd_file = st.file_uploader("Upload job description TXT", type=["txt"])

if cv_file and jd_file:
    with st.spinner("Extracting text..."):
        cv_text = extract_text_from_pdf(cv_file)
        jd_text = extract_text_from_txt(jd_file)

    if not cv_text:
        st.error("Could not extract text from the CV PDF.")
        st.stop()

    if not jd_text:
        st.error("Could not extract text from the job description file.")
        st.stop()

    skill_results = calculate_skill_match(cv_text, jd_text)

    st.divider()
    st.subheader("📊 Basic Skill Match")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Skill Match Score", f"{skill_results['score']}%")
    metric_col2.metric("Matched Skills", len(skill_results["matched"]))
    metric_col3.metric("Missing Skills", len(skill_results["missing"]))

    skill_col1, skill_col2, skill_col3 = st.columns(3)

    with skill_col1:
        st.markdown("### ✅ Matched Skills")
        if skill_results["matched"]:
            for skill in skill_results["matched"]:
                st.success(skill)
        else:
            st.warning("No matched skills found.")

    with skill_col2:
        st.markdown("### ⚠️ Missing Skills")
        if skill_results["missing"]:
            for skill in skill_results["missing"]:
                st.error(skill)
        else:
            st.success("No missing skills detected.")

    with skill_col3:
        st.markdown("### ➕ Extra CV Skills")
        if skill_results["extra"]:
            for skill in skill_results["extra"]:
                st.info(skill)
        else:
            st.write("No extra CV skills detected.")

    st.divider()
    st.subheader("🔎 RAG Retrieval Layer")

    with st.spinner("Creating chunks, embeddings, and FAISS vector index..."):
        cv_chunks = chunk_text(cv_text, "CV")
        jd_chunks = chunk_text(jd_text, "JOB_DESCRIPTION")
        all_chunks = cv_chunks + jd_chunks

        model = load_embedding_model()
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        embeddings = create_embeddings(model, chunk_texts)
        index = build_faiss_index(embeddings)

    st.success(f"Vector index created with {len(all_chunks)} chunks.")

    default_query = (
        "Which skills, projects, and experience from the CV match this GenAI RAG job description?"
    )

    user_query = st.text_input(
        "Ask a semantic search question",
        value=default_query
    )

    top_k = st.slider("Number of evidence chunks to retrieve", min_value=3, max_value=10, value=5)

    evidence_results = search_similar_chunks(
        query=user_query,
        model=model,
        index=index,
        chunks=all_chunks,
        top_k=top_k
    )

    st.markdown("### Retrieved Evidence")

    for result in evidence_results:
        with st.expander(
            f"{result['chunk_id']} | {result['source']} | similarity: {result['score']:.3f}"
        ):
            st.write(result["text"])

    st.divider()
    st.subheader("🧾 Auto Report")

    report = generate_simple_report(skill_results, evidence_results)

    st.text_area("Generated report", report, height=350)

    st.download_button(
        label="Download Report as Markdown",
        data=report,
        file_name="ragscope_report.md",
        mime="text/markdown"
    )

    st.divider()

    with st.expander("Show extracted CV text"):
        st.text_area("CV Text", cv_text, height=250)

    with st.expander("Show extracted Job Description text"):
        st.text_area("Job Description Text", jd_text, height=250)

else:
    st.info("Upload both files to start RAG-based evidence retrieval.")