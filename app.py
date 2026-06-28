#%%

import re
import streamlit as st
from pypdf import PdfReader


# -----------------------------
# Basic text extraction
# -----------------------------
def extract_text_from_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def extract_text_from_txt(uploaded_txt):
    return uploaded_txt.read().decode("utf-8", errors="ignore")


# -----------------------------
# Simple skill matching logic
# -----------------------------
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


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def find_skills(text):
    normalized = normalize_text(text)
    found = set()

    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in normalized:
                found.add(skill)
                break

    return sorted(found)


def calculate_match(cv_text, jd_text):
    cv_skills = set(find_skills(cv_text))
    jd_skills = set(find_skills(jd_text))

    matched = sorted(cv_skills.intersection(jd_skills))
    missing = sorted(jd_skills.difference(cv_skills))
    extra = sorted(cv_skills.difference(jd_skills))

    if len(jd_skills) == 0:
        score = 0
    else:
        score = round((len(matched) / len(jd_skills)) * 100, 2)

    return {
        "cv_skills": sorted(cv_skills),
        "jd_skills": sorted(jd_skills),
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "score": score,
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="RAGScope - CV Job Matcher",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 RAGScope: CV vs Job Description Matcher")
st.write(
    "Upload your CV PDF and one job description text file. "
    "This first version extracts text and performs basic skill matching."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload CV PDF")
    cv_file = st.file_uploader("Upload your CV", type=["pdf"])

with col2:
    st.subheader("2. Upload Job Description")
    jd_file = st.file_uploader("Upload job description", type=["txt"])

if cv_file and jd_file:
    cv_text = extract_text_from_pdf(cv_file)
    jd_text = extract_text_from_txt(jd_file)

    results = calculate_match(cv_text, jd_text)

    st.divider()
    st.subheader("📊 Match Result")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric("Match Score", f"{results['score']}%")
    metric_col2.metric("Matched Skills", len(results["matched"]))
    metric_col3.metric("Missing Skills", len(results["missing"]))

    st.divider()

    left, middle, right = st.columns(3)

    with left:
        st.subheader("✅ Matched Skills")
        if results["matched"]:
            for skill in results["matched"]:
                st.success(skill)
        else:
            st.warning("No matched skills found.")

    with middle:
        st.subheader("⚠️ Missing Skills")
        if results["missing"]:
            for skill in results["missing"]:
                st.error(skill)
        else:
            st.success("No missing skills found from detected job skills.")

    with right:
        st.subheader("➕ Extra CV Skills")
        if results["extra"]:
            for skill in results["extra"]:
                st.info(skill)
        else:
            st.write("No extra skills detected.")

    st.divider()

    with st.expander("Show extracted CV text"):
        st.text_area("CV Text", cv_text, height=300)

    with st.expander("Show extracted Job Description text"):
        st.text_area("Job Description Text", jd_text, height=300)

else:
    st.info("Upload both files to start the analysis.")
# %%
