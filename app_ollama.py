#%%
import re
from typing import Any, Dict, List, TypedDict
import faiss
import requests
import streamlit as st
from langgraph.graph import END, StateGraph
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# Streamlit config
# -------------------------------------------------
st.set_page_config(
    page_title="RAGScope - Local Agentic RAG Analyzer",
    page_icon="🧠",
    layout="wide",
)


# -------------------------------------------------
# Text extraction
# -------------------------------------------------
def extract_text_from_pdf(uploaded_pdf) -> str:
    reader = PdfReader(uploaded_pdf)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def extract_text_from_txt(uploaded_txt) -> str:
    return uploaded_txt.read().decode("utf-8", errors="ignore").strip()


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_cv_noise(text: str) -> str:
    """
    Removes low-value CV header lines before chunking.
    This prevents the retriever from treating contact details as career evidence.
    """
    cleaned_lines = []

    for line in text.splitlines():
        lower = line.lower().strip()

        if not lower:
            continue

        noise_patterns = [
            "rasik rastogi",
            "passau, germany",
            "linkedin.com",
            "rasikrastogi@gmail.com",
            "+4915510191139",
        ]

        if any(pattern in lower for pattern in noise_patterns):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# -------------------------------------------------
# Chunking
# -------------------------------------------------
def chunk_text(text: str, source_name: str, chunk_size: int = 700, overlap: int = 150) -> List[Dict[str, Any]]:
    """
    Splits text into overlapping chunks.
    """
    text = clean_text(text)

    chunks = []
    start = 0
    chunk_number = 1

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                {
                    "chunk_id": f"{source_name}_{chunk_number}",
                    "source": source_name,
                    "text": chunk,
                }
            )

        start += chunk_size - overlap
        chunk_number += 1

    return chunks


def is_low_value_chunk(text: str) -> bool:
    """
    Filters chunks that are unlikely to be useful as technical evidence.
    Keeps chunks if they contain important AI/project keywords.
    """
    lower = text.lower()

    high_value_terms = [
        "rag",
        "retrieval",
        "augmented generation",
        "embedding",
        "embeddings",
        "semantic search",
        "vector",
        "qdrant",
        "pgvector",
        "faiss",
        "chroma",
        "langchain",
        "langgraph",
        "llm",
        "mistral",
        "openai",
        "fastapi",
        "streamlit",
        "evaluation",
        "prompt",
        "agent",
        "python",
        "transformers",
        "machine learning",
        "deep learning",
        "nlp",
        "semantic similarity",
        "hugging face",
        "sbert",
        "mpnet",
    ]

    low_value_terms = [
        "languages:",
        "leadership",
        "current gpa",
        "bachelors",
        "masters",
        "phone",
        "email",
        "linkedin",
    ]

    if any(term in lower for term in high_value_terms):
        return False

    if any(term in lower for term in low_value_terms):
        return True

    if len(text.split()) < 25:
        return True

    return False


# -------------------------------------------------
# Embeddings + FAISS
# -------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def create_embeddings(model, texts: List[str]):
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    return embeddings


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def search_similar_chunks(query: str, model, index, chunks: List[Dict[str, Any]], top_k: int = 6) -> List[Dict[str, Any]]:
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    safe_top_k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, safe_top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]

        results.append(
            {
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
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
    "agentic ai": ["agentic ai", "ai agent", "ai agents", "agent workflow", "tool calling", "agentic workflow"],
    "prompt engineering": ["prompt engineering", "prompting"],
    "nlp": ["nlp", "natural language processing"],
    "evaluation": ["evaluation", "model evaluation", "rag evaluation", "llm evaluation"],
    "api": ["api", "rest api", "rest apis"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural network", "neural networks"],
}


def normalize_for_skills(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def find_skills(text: str) -> List[str]:
    normalized = normalize_for_skills(text)
    found = set()

    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in normalized:
                found.add(skill)
                break

    return sorted(found)


def calculate_skill_match(cv_text: str, jd_text: str) -> Dict[str, Any]:
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


# -------------------------------------------------
# Ollama
# -------------------------------------------------
def check_ollama_connection() -> bool:
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def generate_with_ollama(model_name: str, prompt: str) -> str:
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
            "num_predict": 900,
        },
    }

    response = requests.post(url, json=payload, timeout=300)

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

    data = response.json()
    return data.get("response", "").strip()


# -------------------------------------------------
# Safety controls
# -------------------------------------------------
def remove_unsafe_fake_metrics(report: str) -> str:
    """
    Removes common fake metric patterns that small local LLMs may invent.
    This is a safety layer for CV/report generation.
    """
    fake_metric_patterns = [
        r"resulting in a \d+% [^.]*\.",
        r"improving [^.]* by \d+%[.]*",
        r"improved [^.]* by \d+%[.]*",
        r"reduced [^.]* by \d+%[.]*",
        r"reduction [^.]* by \d+%[.]*",
        r"increased [^.]* by \d+%[.]*",
        r"achieved \d+% [^.]*\.",
        r"\d+% reduction [^.]*\.",
        r"\d+% improvement [^.]*\.",
    ]

    cleaned = report

    for pattern in fake_metric_patterns:
        cleaned = re.sub(
            pattern,
            "[Removed unsupported metric: no evidence found.]",
            cleaned,
            flags=re.IGNORECASE,
        )

    return cleaned


def replace_future_cv_bullets(report: str) -> str:
    """
    Replaces the LLM-generated Future CV Bullet Ideas section
    with safe deterministic bullets. This prevents fake metrics.
    """
    safe_section = """
**CV Bullet Ideas:**

1. Built a local agentic RAG CV-job matching tool using MiniLM embeddings, FAISS vector search, Ollama, Streamlit, and LangGraph for evidence-grounded job-fit analysis.
2. Implemented document ingestion, text chunking, semantic retrieval, and source-backed reporting to identify matched skills, weak evidence, and missing GenAI requirements.
3. Added retrieval-quality checks, lightweight RAG evaluation, and unsupported-metric filtering to reduce noisy evidence and prevent fabricated CV claims.
""".strip()

    patterns = [
        r"\*\*Future CV Bullet Ideas:\*\*.*?(?=\n\*\*Interview Explanation:\*\*|\Z)",
        r"##\s*Future CV Bullet Ideas.*?(?=\n##\s*Interview Explanation|\Z)",
        r"5\.\s*Future CV Bullet Ideas.*?(?=\n6\.\s*Interview Explanation|\Z)",
    ]

    for pattern in patterns:
        if re.search(pattern, report, flags=re.DOTALL | re.IGNORECASE):
            return re.sub(
                pattern,
                safe_section + "\n",
                report,
                flags=re.DOTALL | re.IGNORECASE,
            )

    return report + "\n\n" + safe_section

def replace_interview_explanation(report: str) -> str:
    """
    Replaces the LLM-generated Interview Explanation section
    with a safe first-person explanation.
    This prevents the model from writing as an interviewer.
    """
    safe_section = """
**Interview Explanation:**

I built RAGScope as a local agentic RAG application to compare a CV against GenAI job descriptions in an evidence-grounded way. The system extracts text from documents, chunks the content, creates MiniLM embeddings, retrieves relevant evidence with FAISS, and uses Ollama for local report generation. I also added a LangGraph workflow with separate steps for requirement extraction, CV evidence review, gap analysis, report generation, and safety checking. To make the output more reliable, I added evidence-quality checks, lightweight RAG evaluation, and unsupported-metric filtering so the system does not turn listed skills into exaggerated CV claims.
""".strip()

    patterns = [
        r"\*\*Interview Explanation:\*\*.*?\Z",
        r"##\s*Interview Explanation.*?\Z",
        r"6\.\s*Interview Explanation.*?\Z",
    ]

    for pattern in patterns:
        if re.search(pattern, report, flags=re.DOTALL | re.IGNORECASE):
            return re.sub(
                pattern,
                safe_section,
                report,
                flags=re.DOTALL | re.IGNORECASE,
            )

    return report + "\n\n" + safe_section


def replace_project_improvements(report: str) -> str:
    """
    Replaces generic LLM-generated project suggestions
    with project-specific RAGScope improvements.
    """
    safe_section = """
**Project Status and Next Improvements:**

1. Completed: Added a LangGraph-based agentic workflow with separate steps for job requirement extraction, CV evidence matching, gap analysis, report generation, and unsupported-claim checking.
2. Completed: Added an evidence-quality checker that separates strong project/research evidence from weak listed skills and missing job requirements.
3. Completed: Added a lightweight RAG evaluation module to check retrieval health, CV evidence retrieval, job-description retrieval, missing-skill detection, and contact-noise filtering.
4. Next improvement: Add safe sample CV/job-description files, screenshots, sample reports, and a clear architecture diagram before final GitHub polishing.
""".strip()

    patterns = [
        r"\*\*Suggested Weekend Project Improvements:\*\*.*?(?=\n\*\*Future CV Bullet Ideas:\*\*|\Z)",
        r"##\s*Suggested Weekend Project Improvements.*?(?=\n##\s*Future CV Bullet Ideas|\Z)",
        r"4\.\s*Suggested Weekend Project Improvements.*?(?=\n5\.\s*Future CV Bullet Ideas|\Z)",
    ]

    for pattern in patterns:
        if re.search(pattern, report, flags=re.DOTALL | re.IGNORECASE):
            return re.sub(
                pattern,
                safe_section + "\n",
                report,
                flags=re.DOTALL | re.IGNORECASE,
            )

    return report + "\n\n" + safe_section


def replace_evidence_sections(
    report: str,
    skill_results: Dict[str, Any],
    evidence_chunks: List[Dict[str, Any]],
) -> str:
    """
    Replaces LLM-generated Strong Evidence and Weak/Missing Evidence sections
    with deterministic evidence-quality sections.

    The goal is to separate:
    - strong evidence: project/research/implementation chunks
    - weak evidence: skills that are only listed in a Skills/Education section
    - missing evidence: job-relevant skills not detected in the CV
    """

    def short_quote(text: str, max_chars: int = 300) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text

    def phrase_exists(text: str, phrase: str) -> bool:
        """
        Safer matching than raw substring search.
        Prevents false positives such as 'git' inside 'Digital'.
        """
        text_lower = text.lower()
        phrase_lower = phrase.lower().strip()

        if not phrase_lower:
            return False

        if re.fullmatch(r"[a-z0-9+#.-]+", phrase_lower):
            pattern = r"(?<![a-z0-9+#.-])" + re.escape(phrase_lower) + r"(?![a-z0-9+#.-])"
            return re.search(pattern, text_lower) is not None

        return phrase_lower in text_lower

    def contains_skill_alias(text: str, skill: str) -> bool:
        aliases = SKILL_ALIASES.get(skill, [skill])
        return any(phrase_exists(text, alias) for alias in aliases)

    def is_skills_or_education_chunk(text: str) -> bool:
        lower = text.lower()

        weak_section_terms = [
            "skills",
            "programming:",
            "frameworks:",
            "machine learning & deep learning:",
            "nlp & genai:",
            "visualization:",
            "languages:",
            "leadership",
            "education",
            "bachelors",
            "masters",
            "current gpa",
            "cgpa",
        ]

        return any(term in lower for term in weak_section_terms)

    def looks_like_project_evidence(text: str) -> bool:
        lower = text.lower()

        action_terms = [
            "developing",
            "building",
            "built",
            "designed",
            "implemented",
            "integrating",
            "experimenting",
            "conducted",
            "evaluated",
            "visualized",
            "compared",
            "tested",
            "analyzed",
            "created",
            "developed",
        ]

        project_context_terms = [
            "project",
            "seminar",
            "study",
            "system",
            "pipeline",
            "pipelines",
            "retrieval-augmented generation",
            "retrieval augmented generation",
            "sentence embeddings for semantic similarity",
            "semantic similarity",
            "domain classification",
            "benchmark datasets",
            "context-aware response generation",
            "vector databases",
            "semantic search",
        ]

        technical_terms = [
            "rag",
            "retrieval",
            "embedding",
            "embeddings",
            "semantic search",
            "vector database",
            "vector databases",
            "qdrant",
            "pgvector",
            "faiss",
            "langchain",
            "langgraph",
            "llm",
            "openai",
            "mistral",
            "sbert",
            "mpnet",
            "universal sentence encoder",
            "hugging face",
            "transformers",
            "evaluation",
            "metrics",
        ]

        weak_chunk = is_skills_or_education_chunk(text)
        has_action = any(term in lower for term in action_terms)
        has_project_context = any(term in lower for term in project_context_terms)
        has_technical_content = any(term in lower for term in technical_terms)

        # A skills/education chunk should not be strong evidence unless it clearly
        # contains project/research action. This prevents skill lists from being
        # treated as strong proof.
        if weak_chunk and not has_action:
            return False

        return has_technical_content and (has_action or has_project_context)

    cv_chunks = [chunk for chunk in evidence_chunks if chunk.get("source") == "CV"]

    strong_lines: List[str] = []
    weak_lines: List[str] = []

    matched_skills = skill_results.get("matched", [])
    missing_skills = skill_results.get("missing", [])

    for skill in matched_skills:
        matching_cv_chunks = [
            chunk
            for chunk in cv_chunks
            if contains_skill_alias(chunk.get("text", ""), skill)
        ]

        strong_chunk = None

        # Prefer real project/research chunks.
        for chunk in matching_cv_chunks:
            if looks_like_project_evidence(chunk.get("text", "")):
                strong_chunk = chunk
                break

        if strong_chunk:
            strong_lines.append(
                f"- **{skill}**: Supported by `{strong_chunk['chunk_id']}` "
                f"(similarity {strong_chunk['score']:.3f}). "
                f"Evidence: \"{short_quote(strong_chunk['text'])}\""
            )
        elif matching_cv_chunks:
            weak_lines.append(
                f"- **{skill}**: Detected in the CV, but mainly as a listed skill or general statement. "
                f"The retrieved evidence does not clearly show a completed project, implementation detail, result, or evaluation."
            )
        else:
            weak_lines.append(
                f"- **{skill}**: Detected by the overall skill matcher, but no supporting retrieved CV chunk was found in the current top-k evidence."
            )

    for skill in missing_skills:
        weak_lines.append(
            f"- **{skill}**: Required or useful for the job description, but not clearly detected in the CV."
        )

    if not strong_lines:
        strong_lines.append("- No strong project-level evidence was found in the retrieved CV chunks.")

    if not weak_lines:
        weak_lines.append("- No major weak or missing evidence detected from the current skill list.")

    strong_section = "**Strong Evidence:**\n\n" + "\n".join(strong_lines) + "\n"
    weak_section = "**Weak or Missing Evidence:**\n\n" + "\n".join(weak_lines) + "\n"

    strong_patterns = [
        r"\*\*Strong Evidence:?\*\*.*?(?=\n\*\*Weak or Missing Evidence:?\*\*|\n\*\*Weak/Missing Evidence:?\*\*|\Z)",
        r"##\s*Strong Evidence.*?(?=\n##\s*Weak|\Z)",
        r"2\.\s*Strong Evidence.*?(?=\n3\.\s*Weak|\Z)",
    ]

    weak_patterns = [
        r"\*\*Weak or Missing Evidence:?\*\*.*?(?=\n\*\*Suggested Weekend Project Improvements:?\*\*|\Z)",
        r"\*\*Weak/Missing Evidence:?\*\*.*?(?=\n\*\*Suggested Weekend Project Improvements:?\*\*|\Z)",
        r"##\s*Weak.*?(?=\n##\s*Suggested Weekend Project Improvements|\Z)",
        r"3\.\s*Weak.*?(?=\n4\.\s*Suggested Weekend Project Improvements|\Z)",
    ]

    replaced_strong = False
    for pattern in strong_patterns:
        if re.search(pattern, report, flags=re.DOTALL | re.IGNORECASE):
            report = re.sub(
                pattern,
                strong_section,
                report,
                flags=re.DOTALL | re.IGNORECASE,
            )
            replaced_strong = True
            break

    if not replaced_strong:
        report += "\n\n" + strong_section

    replaced_weak = False
    for pattern in weak_patterns:
        if re.search(pattern, report, flags=re.DOTALL | re.IGNORECASE):
            report = re.sub(
                pattern,
                weak_section,
                report,
                flags=re.DOTALL | re.IGNORECASE,
            )
            replaced_weak = True
            break

    if not replaced_weak:
        report += "\n\n" + weak_section

    return report


# -------------------------------------------------
# Lightweight RAG evaluation
# -------------------------------------------------
def evaluate_rag_pipeline(
    all_chunks: List[Dict[str, Any]],
    evidence_chunks: List[Dict[str, Any]],
    skill_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Lightweight evaluation for the RAG pipeline.

    This does not judge the final LLM answer.
    It checks whether retrieval and safety-related preprocessing look healthy.
    """

    checks = []

    def add_check(name: str, passed: bool, detail: str):
        checks.append(
            {
                "name": name,
                "passed": passed,
                "detail": detail,
            }
        )

    # 1. Vector index/chunk availability
    add_check(
        name="Usable chunks created",
        passed=len(all_chunks) > 0,
        detail=f"{len(all_chunks)} chunks available after filtering.",
    )

    # 2. Retrieval availability
    add_check(
        name="Evidence chunks retrieved",
        passed=len(evidence_chunks) > 0,
        detail=f"{len(evidence_chunks)} chunks retrieved for the query.",
    )

    # 3. CV evidence present
    cv_retrieved = [chunk for chunk in evidence_chunks if chunk.get("source") == "CV"]
    add_check(
        name="CV evidence retrieved",
        passed=len(cv_retrieved) > 0,
        detail=f"{len(cv_retrieved)} retrieved chunks came from the CV.",
    )

    # 4. Job description evidence present
    jd_retrieved = [
        chunk for chunk in evidence_chunks
        if chunk.get("source") == "JOB_DESCRIPTION"
    ]
    add_check(
        name="Job description evidence retrieved",
        passed=len(jd_retrieved) > 0,
        detail=f"{len(jd_retrieved)} retrieved chunks came from the job description.",
    )

    # 5. Average similarity check
    if evidence_chunks:
        avg_similarity = sum(chunk["score"] for chunk in evidence_chunks) / len(evidence_chunks)
    else:
        avg_similarity = 0.0

    add_check(
        name="Average retrieval similarity reasonable",
        passed=avg_similarity >= 0.25,
        detail=f"Average similarity score: {avg_similarity:.3f}.",
    )

    # 6. Missing-skill detection active
    missing_skills = skill_results.get("missing", [])
    add_check(
        name="Missing-skill detection active",
        passed=isinstance(missing_skills, list),
        detail=f"{len(missing_skills)} missing skills detected.",
    )

    # 7. Contact/header noise check
    combined_retrieved_text = " ".join(
        chunk.get("text", "").lower() for chunk in evidence_chunks
    )

    noise_terms = [
        "rasikrastogi@gmail.com",
        "+4915510191139",
        "linkedin.com",
        "passau, germany",
    ]

    noise_found = any(term in combined_retrieved_text for term in noise_terms)

    add_check(
        name="Contact/header noise removed",
        passed=not noise_found,
        detail="No contact/header noise found in retrieved evidence." if not noise_found else "Contact/header noise found in retrieved evidence.",
    )

    passed_count = sum(1 for check in checks if check["passed"])
    total_count = len(checks)
    evaluation_score = round((passed_count / total_count) * 100, 2) if total_count else 0

    return {
        "checks": checks,
        "passed_count": passed_count,
        "total_count": total_count,
        "evaluation_score": evaluation_score,
        "avg_similarity": avg_similarity,
        "retrieved_chunks": len(evidence_chunks),
    }


def render_evaluation_panel(evaluation_results: Dict[str, Any]) -> None:
    """
    Renders evaluation results in Streamlit.
    """

    st.divider()
    st.subheader("🧪 Lightweight RAG Evaluation")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Evaluation Score",
        f"{evaluation_results['evaluation_score']}%"
    )

    col2.metric(
        "Checks Passed",
        f"{evaluation_results['passed_count']}/{evaluation_results['total_count']}"
    )

    col3.metric(
        "Avg Similarity",
        f"{evaluation_results['avg_similarity']:.3f}"
    )

    col4.metric(
        "Retrieved Chunks",
        evaluation_results["retrieved_chunks"]
    )

    st.markdown("### Evaluation Checks")

    for check in evaluation_results["checks"]:
        if check["passed"]:
            st.success(f"{check['name']}: {check['detail']}")
        else:
            st.warning(f"{check['name']}: {check['detail']}")


# -------------------------------------------------
# Prompt building
# -------------------------------------------------
def build_evidence_text(evidence_chunks: List[Dict[str, Any]]) -> str:
    evidence_text = ""

    for item in evidence_chunks:
        evidence_text += f"\n[{item['chunk_id']} | {item['source']} | similarity={item['score']:.3f}]\n"
        evidence_text += item["text"]
        evidence_text += "\n"

    return evidence_text.strip()


def build_ollama_prompt(skill_results: Dict[str, Any], evidence_chunks: List[Dict[str, Any]]) -> str:
    evidence_text = build_evidence_text(evidence_chunks)

    prompt = f"""
You are an honest AI career analyst and RAG evaluator.

You must analyze a candidate's CV against a GenAI / RAG job description.

Important rules:
- Use ONLY the retrieved evidence chunks.
- Do not invent experience, tools, outcomes, metrics, percentages, accuracy values, reductions, or improvements.
- Do not create numbers like "10% improvement", "15% reduction", "high accuracy", or "faster performance" unless the evidence explicitly contains that number.
- Do not exaggerate listed skills into completed projects.
- Do not use contact details, name, email, phone number, location, LinkedIn, education title, or language section as strong evidence.
- Do not treat operations, coordination, documentation, or leadership as strong GenAI evidence unless directly connected to AI/ML/RAG implementation.
- Strong evidence must come from projects, technical implementation details, tools, methods, evaluation setup, or measurable outcomes.
- If the CV only lists a skill but does not show a project, result, implementation, or evaluation, mark it as weak evidence.
- Mention chunk IDs when making claims.
- Keep the report practical for a working-student or internship application in Germany.

Detected skill match:
Overall score: {skill_results['score']}%
Matched skills: {', '.join(skill_results['matched']) if skill_results['matched'] else 'None'}
Missing skills: {', '.join(skill_results['missing']) if skill_results['missing'] else 'None'}
Extra CV skills: {', '.join(skill_results['extra']) if skill_results['extra'] else 'None'}

Retrieved evidence chunks:
{evidence_text}

Write the report using exactly these sections:

1. Overall Fit
Give a realistic 3-5 sentence assessment.

2. Strong Evidence
List only skills or project evidence that are supported by the chunks.

3. Weak or Missing Evidence
List what the CV does not prove strongly enough.

4. Suggested Weekend Project Improvements
Give concrete improvements to make this project stronger for GenAI, RAG, Agentic AI, and LLM application roles.

5. Future CV Bullet Ideas
Write only this sentence:
"Future CV bullets will be generated by the application safety layer after project completion."

6. Interview Explanation
Give a short explanation the candidate can say in an interview.
"""

    return prompt


# -------------------------------------------------
# Rule-based fallback
# -------------------------------------------------
def generate_rule_based_report(skill_results: Dict[str, Any], evidence_chunks: List[Dict[str, Any]]) -> str:
    report = "# RAGScope Local Agentic RAG Job-Fit Report\n\n"

    report += "## 1. Overall Fit\n"
    report += (
        f"The detected skill match score is **{skill_results['score']}%**. "
        "This score is based on known skills found in the CV and job description. "
        "The retrieved chunks provide supporting evidence, but the final interpretation should be checked manually.\n\n"
    )

    report += "## 2. Matched Skills\n"
    if skill_results["matched"]:
        for skill in skill_results["matched"]:
            report += f"- {skill}\n"
    else:
        report += "- No matched skills detected.\n"

    report += "\n## 3. Missing or Weak Skills\n"
    if skill_results["missing"]:
        for skill in skill_results["missing"]:
            report += f"- {skill}\n"
    else:
        report += "- No missing skills detected from the known skill list.\n"

    report += "\n## 4. Retrieved Evidence\n"
    for item in evidence_chunks:
        report += f"\n### {item['chunk_id']} | {item['source']} | Similarity: {item['score']:.3f}\n"
        report += item["text"] + "\n"

    report += "\n## 5. Suggested Weekend Project Improvements\n"
    report += "- Add local LLM generation using Ollama.\n"
    report += "- Add grounded report generation from retrieved chunks.\n"
    report += "- Add a LangGraph workflow for job extraction, CV evidence matching, gap analysis, and safety checking.\n"
    report += "- Add screenshots, sample reports, and a clear README.\n"

    report += "\n## 6. Future CV Bullet Ideas\n"
    report += "- Future bullet after completion: Built a local agentic RAG application that compares CVs with GenAI job descriptions using MiniLM embeddings, FAISS, Ollama, and LangGraph.\n"
    report += "- Future bullet after completion: Implemented evidence retrieval over CV and job-description chunks to ground job-fit analysis and missing-skill detection.\n"
    report += "- Future bullet after completion: Added safety checks to remove unsupported metrics and prevent fabricated CV claims.\n"

    return report


# -------------------------------------------------
# LangGraph agent workflow
# -------------------------------------------------
class AgentState(TypedDict):
    skill_results: Dict[str, Any]
    evidence_chunks: List[Dict[str, Any]]
    model_name: str
    ollama_ready: bool
    agent_trace: List[str]
    requirement_summary: str
    evidence_summary: str
    gap_summary: str
    report: str


def add_trace(state: AgentState, message: str) -> List[str]:
    trace = list(state.get("agent_trace", []))
    trace.append(message)
    return trace


def job_requirement_agent(state: AgentState) -> AgentState:
    jd_skills = state["skill_results"].get("jd_skills", [])

    if jd_skills:
        summary = "Detected job requirements: " + ", ".join(jd_skills)
    else:
        summary = "No clear job requirements detected from the known skill list."

    return {
        **state,
        "requirement_summary": summary,
        "agent_trace": add_trace(state, "Job Requirement Agent completed requirement extraction."),
    }


def cv_evidence_agent(state: AgentState) -> AgentState:
    evidence_lines = []

    for chunk in state["evidence_chunks"]:
        source = chunk.get("source", "")
        chunk_id = chunk.get("chunk_id", "")
        score = chunk.get("score", 0.0)

        if source == "CV":
            evidence_lines.append(f"{chunk_id} with similarity {score:.3f}")

    if evidence_lines:
        summary = "Relevant CV evidence found in: " + ", ".join(evidence_lines)
    else:
        summary = "No strong CV evidence chunks found."

    return {
        **state,
        "evidence_summary": summary,
        "agent_trace": add_trace(state, "CV Evidence Agent completed evidence review."),
    }


def gap_analysis_agent(state: AgentState) -> AgentState:
    matched = state["skill_results"].get("matched", [])
    missing = state["skill_results"].get("missing", [])

    summary = ""
    summary += "Matched skills: " + (", ".join(matched) if matched else "None") + "\n"
    summary += "Missing or weak skills: " + (", ".join(missing) if missing else "None")

    return {
        **state,
        "gap_summary": summary,
        "agent_trace": add_trace(state, "Gap Analysis Agent completed skill-gap comparison."),
    }


def report_generation_agent(state: AgentState) -> AgentState:
    prompt = build_ollama_prompt(
        skill_results=state["skill_results"],
        evidence_chunks=state["evidence_chunks"],
    )

    if state["ollama_ready"]:
        try:
            report = generate_with_ollama(state["model_name"], prompt)
        except Exception as e:
            report = generate_rule_based_report(
                state["skill_results"],
                state["evidence_chunks"],
            )
            report += f"\n\nOllama failed, fallback used. Error: {e}"
    else:
        report = generate_rule_based_report(
            state["skill_results"],
            state["evidence_chunks"],
        )

    return {
        **state,
        "report": report,
        "agent_trace": add_trace(state, "Report Generation Agent completed grounded report generation."),
    }


def safety_checker_agent(state: AgentState) -> AgentState:
    report = state["report"]

    report = remove_unsafe_fake_metrics(report)
    report = replace_evidence_sections(
        report,
        state["skill_results"],
        state["evidence_chunks"],
    )
    report = replace_project_improvements(report)
    report = replace_future_cv_bullets(report)
    report = replace_interview_explanation(report)

    return {
        **state,
        "report": report,
        "agent_trace": add_trace(
            state,
            "Safety Checker Agent applied unsupported-metric filtering, evidence-quality checking, and safe CV/project sections.",
        ),
    }


@st.cache_resource
def build_agentic_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("job_requirement_agent", job_requirement_agent)
    workflow.add_node("cv_evidence_agent", cv_evidence_agent)
    workflow.add_node("gap_analysis_agent", gap_analysis_agent)
    workflow.add_node("report_generation_agent", report_generation_agent)
    workflow.add_node("safety_checker_agent", safety_checker_agent)

    workflow.set_entry_point("job_requirement_agent")

    workflow.add_edge("job_requirement_agent", "cv_evidence_agent")
    workflow.add_edge("cv_evidence_agent", "gap_analysis_agent")
    workflow.add_edge("gap_analysis_agent", "report_generation_agent")
    workflow.add_edge("report_generation_agent", "safety_checker_agent")
    workflow.add_edge("safety_checker_agent", END)

    return workflow.compile()


# -------------------------------------------------
# UI helpers
# -------------------------------------------------
def render_agentic_output(final_state: Dict[str, Any]) -> None:
    report = final_state.get("report", "")

    st.markdown("### 🧭 Agent Trace")
    for step in final_state.get("agent_trace", []):
        st.success(step)

    with st.expander("Agent 1 Output: Job Requirement Summary"):
        st.write(final_state.get("requirement_summary", ""))

    with st.expander("Agent 2 Output: CV Evidence Summary"):
        st.write(final_state.get("evidence_summary", ""))

    with st.expander("Agent 3 Output: Gap Analysis Summary"):
        st.write(final_state.get("gap_summary", ""))

    st.markdown("### 🧾 Final Agentic Report")
    st.text_area("Generated Report", report, height=500)

    st.download_button(
        label="Download Agentic Report as Markdown",
        data=report,
        file_name="ragscope_agentic_report.md",
        mime="text/markdown",
    )


# -------------------------------------------------
# Main UI
# -------------------------------------------------
st.title("🧠 RAGScope: Local Agentic RAG Analyzer")

st.write(
    "This version uses local Ollama generation instead of Groq/OpenAI. "
    "It extracts text, retrieves evidence using embeddings + FAISS, and runs a LangGraph workflow "
    "for requirement extraction, evidence review, gap analysis, report generation, and safety checking."
)

with st.sidebar:
    st.header("Local LLM Settings")

    model_name = st.text_input(
        "Ollama model name",
        value="gemma3:1b",
        help="Start with gemma3:1b on an 8 GB RAM laptop. Later try llama3.2:3b if performance is acceptable.",
    )

    ollama_ready = check_ollama_connection()

    if ollama_ready:
        st.success("Ollama is running on localhost:11434")
    else:
        st.error("Ollama is not running")
        st.caption("Open Ollama or run: ollama run gemma3:1b")

    st.markdown("---")
    st.caption("Local mode: no Groq/OpenAI API key needed.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload CV PDF")
    cv_file = st.file_uploader("Upload your CV PDF", type=["pdf"])

with col2:
    st.subheader("2. Upload Job Description TXT")
    jd_file = st.file_uploader("Upload job description", type=["txt"])

if "agentic_final_state" not in st.session_state:
    st.session_state.agentic_final_state = None

if cv_file and jd_file:
    with st.spinner("Extracting document text..."):
        cv_text = remove_cv_noise(extract_text_from_pdf(cv_file))
        jd_text = extract_text_from_txt(jd_file)

    if not cv_text:
        st.error("Could not extract text from the CV PDF.")
        st.stop()

    if not jd_text:
        st.error("Could not extract text from the job description.")
        st.stop()

    skill_results = calculate_skill_match(cv_text, jd_text)

    st.divider()
    st.subheader("📊 Skill Match Summary")

    m1, m2, m3 = st.columns(3)
    m1.metric("Skill Match Score", f"{skill_results['score']}%")
    m2.metric("Matched Skills", len(skill_results["matched"]))
    m3.metric("Missing Skills", len(skill_results["missing"]))

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### ✅ Matched")
        if skill_results["matched"]:
            for skill in skill_results["matched"]:
                st.success(skill)
        else:
            st.warning("No matched skills found.")

    with c2:
        st.markdown("### ⚠️ Missing")
        if skill_results["missing"]:
            for skill in skill_results["missing"]:
                st.error(skill)
        else:
            st.success("No missing skills detected.")

    with c3:
        st.markdown("### ➕ Extra CV Skills")
        if skill_results["extra"]:
            for skill in skill_results["extra"]:
                st.info(skill)
        else:
            st.write("No extra CV skills detected.")

    st.divider()
    st.subheader("🔎 RAG Evidence Retrieval")

    retrieval_query = st.text_area(
        "Retrieval question",
        value=(
            "Find only technical project evidence from the CV and job description related to "
            "RAG, retrieval pipelines, embeddings, vector databases, LangChain, LangGraph, "
            "LLMs, prompt engineering, evaluation, Python, FastAPI, Streamlit, and agentic AI. "
            "Ignore contact details, education header, languages, and generic personal information."
        ),
        height=90,
    )

    top_k = st.slider("Number of retrieved chunks", min_value=3, max_value=10, value=6)

    with st.spinner("Creating chunks, embeddings, and FAISS index..."):
        cv_chunks = chunk_text(cv_text, "CV")
        jd_chunks = chunk_text(jd_text, "JOB_DESCRIPTION")

        all_chunks = [
            chunk
            for chunk in (cv_chunks + jd_chunks)
            if not is_low_value_chunk(chunk["text"])
        ]

        if not all_chunks:
            st.error("No useful chunks remained after filtering. Reduce filtering or use a longer CV/job description.")
            st.stop()

        embedding_model = load_embedding_model()
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        embeddings = create_embeddings(embedding_model, chunk_texts)
        index = build_faiss_index(embeddings)

        evidence_chunks = search_similar_chunks(
            query=retrieval_query,
            model=embedding_model,
            index=index,
            chunks=all_chunks,
            top_k=top_k,
        )

    st.success(f"Created vector index with {len(all_chunks)} chunks.")

    st.markdown("### Retrieved Evidence Chunks")

    for item in evidence_chunks:
        with st.expander(
            f"{item['chunk_id']} | {item['source']} | similarity: {item['score']:.3f}"
        ):
            st.write(item["text"])

    evaluation_results = evaluate_rag_pipeline(
        all_chunks=all_chunks,
        evidence_chunks=evidence_chunks,
        skill_results=skill_results,
    )

    render_evaluation_panel(evaluation_results)

    st.divider()
    st.subheader("🧾 Local Grounded Report")

    if st.button("Generate Agentic Report"):
        with st.spinner("Running LangGraph agentic workflow locally..."):
            app_graph = build_agentic_workflow()

            initial_state = {
                "skill_results": skill_results,
                "evidence_chunks": evidence_chunks,
                "model_name": model_name,
                "ollama_ready": ollama_ready,
                "agent_trace": [],
                "requirement_summary": "",
                "evidence_summary": "",
                "gap_summary": "",
                "report": "",
            }

            final_state = app_graph.invoke(initial_state)
            st.session_state.agentic_final_state = final_state

    if st.session_state.agentic_final_state:
        render_agentic_output(st.session_state.agentic_final_state)

    st.divider()

    with st.expander("Show extracted CV text"):
        st.text_area("CV Text", cv_text, height=250)

    with st.expander("Show extracted Job Description text"):
        st.text_area("Job Description Text", jd_text, height=250)

else:
    st.info("Upload both your CV PDF and a job description TXT file to start.")
