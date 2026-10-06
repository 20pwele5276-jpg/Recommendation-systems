"""
AI Job & Career Recommendation System
========================================
Recommends jobs based on a resume, matching it against a dataset of job
roles using THREE complementary NLP techniques, with an LLM layer on top:

  1. TF-IDF + cosine similarity   - classic "bag of words" text matching
  2. Sentence embeddings          - semantic similarity (understands meaning,
                                     not just shared words)
  3. LLM structured extraction    - turns messy resume text into a clean,
                                     typed skill profile (Pydantic + Groq
                                     structured outputs)
  4. LLM explanation              - plain-English reasoning for why each
                                     job is (or isn't) a strong fit

TECH STACK
----------
Streamlit · scikit-learn (TF-IDF) · sentence-transformers (embeddings)
Groq API (LLM) · Pydantic (validation) · pypdf (optional resume PDF reading)

Run with:  streamlit run career_recommender.py
"""

import json
import logging
import os
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import groq
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# 1. THE SHAPE OF A RESUME
#    Turning messy resume text into clean, typed data so the rest of the
#    app doesn't have to guess at formats or regex its way through prose.
# =============================================================================
class SkillProfile(BaseModel):
    skills: list[str]          # normalized skill keywords, e.g. "python", "sql"
    years_experience: str      # kept as text - resumes phrase this inconsistently
    past_titles: list[str]     # job titles the person has held
    interests: list[str]       # stated or implied career interests
    summary: str                # one short paragraph, third person


SKILL_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {"type": "array", "items": {"type": "string"}},
        "years_experience": {"type": "string"},
        "past_titles": {"type": "array", "items": {"type": "string"}},
        "interests": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["skills", "years_experience", "past_titles", "interests", "summary"],
    "additionalProperties": False,
}

EXTRACTION_PROMPT = """\
You read resumes and extract a clean, structured skill profile.

Rules:
- "skills" should be concrete, lowercase keywords (e.g. "python", "sql",
  "project management"), not full sentences. Include both hard and soft
  skills that are clearly evidenced in the text.
- "years_experience" is a short phrase like "3 years" or "entry-level" -
  infer it from dates/titles if not stated directly, and say "unclear" if
  you genuinely can't tell.
- "past_titles" lists job titles actually held, most recent first.
- "interests" captures stated or clearly implied career interests/goals.
- "summary" is 2-3 sentences, third person, describing this person
  professionally - the kind of line a recruiter would skim.
"""

# =============================================================================
# 2. ERROR HANDLING
#    Technical detail -> terminal log. The user only ever sees a short,
#    friendly, actionable message.
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("career_recommender")


class RecommenderError(Exception):
    """A failure we can explain to the user in plain language."""


def explain_error(e: Exception) -> str:
    if isinstance(e, groq.AuthenticationError):
        return "Your API key was rejected. Check GROQ_API_KEY."
    if isinstance(e, groq.NotFoundError):
        return "That model isn't available for your account. Pick another one."
    if isinstance(e, groq.RateLimitError):
        return "Rate limit reached. Wait a minute and try again."
    if isinstance(e, groq.APITimeoutError):
        return "The request timed out. Please try again."
    if isinstance(e, groq.APIConnectionError):
        return "Couldn't reach Groq. Check your internet connection."
    if isinstance(e, groq.InternalServerError):
        return "Groq's servers had a problem. Please try again shortly."
    if isinstance(e, groq.GroqError):
        return f"Groq returned an error: {e}"
    return f"Unexpected error: {e}"


RETRYABLE = (groq.APIConnectionError, groq.RateLimitError, groq.InternalServerError)
MAX_ATTEMPTS = 3
STRICT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]  # support strict structured outputs


# =============================================================================
# 3. STEP 1 - EXTRACT: resume text -> structured SkillProfile
# =============================================================================
def extract_skill_profile(resume_text: str, client: Groq, model: str) -> SkillProfile:
    if not resume_text.strip():
        raise RecommenderError("Please paste or upload a resume first.")

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,  # low temperature: extraction should be consistent, not creative
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": resume_text},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "skill_profile", "strict": True, "schema": SKILL_PROFILE_SCHEMA},
                },
            )
            return SkillProfile.model_validate_json(response.choices[0].message.content)

        except RETRYABLE as e:
            last_error = e
            logger.warning("Attempt %d/%d failed: %r", attempt, MAX_ATTEMPTS, e)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
        except ValidationError as e:
            logger.error("Profile validation failed: %r", e)
            raise RecommenderError("Couldn't parse a skill profile from this resume.") from e
        except groq.GroqError as e:
            logger.error("Groq error: %r", e)
            raise RecommenderError(explain_error(e)) from e

    raise RecommenderError(explain_error(last_error)) from last_error


def explain_match(profile: SkillProfile, job: dict, matched: list, missing: list,
                   client: Groq, model: str) -> str:
    """Short, free-text explanation of fit. Degrades gracefully: if this call
    fails, the app still works - it just falls back to a simple template."""
    prompt = (
        f"Candidate summary: {profile.summary}\n"
        f"Candidate skills: {', '.join(profile.skills)}\n"
        f"Candidate interests: {', '.join(profile.interests)}\n\n"
        f"Job: {job['title']} - {job['description']}\n"
        f"Matched skills: {', '.join(matched) or 'none'}\n"
        f"Missing skills: {', '.join(missing) or 'none'}\n\n"
        f"In 2 short sentences, explain why this job is (or isn't) a good fit "
        f"for this specific candidate. Mention one concrete matched skill or "
        f"interest by name. Be honest, not salesy."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.5,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except groq.GroqError as e:
        logger.warning("Explanation call failed, using fallback: %r", e)
        if matched:
            return f"Strong overlap on {', '.join(matched[:3])}. Worth a closer look."
        return "Limited overlap with your current listed skills, but could still be a stretch fit."


# =============================================================================
# 4. JOB DATASET
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
JOBS_PATH = BASE_DIR / "career_jobs.json"


@st.cache_data
def load_jobs() -> list:
    with open(JOBS_PATH, encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 5. MATCHING ENGINE
#    Two independent scoring methods, so the UI can show both and the user
#    can see where classic word-overlap and semantic similarity agree or
#    disagree.
# =============================================================================
def tfidf_scores(resume_text: str, job_texts: list) -> np.ndarray:
    """Classic NLP baseline: term-frequency overlap, no semantic understanding."""
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([resume_text] + job_texts)
    return cosine_similarity(matrix[0:1], matrix[1:]).flatten()


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Loaded once per session. Returns None if sentence-transformers isn't
    installed, so the app can fall back to TF-IDF-only instead of crashing."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:  # not installed, or model download failed (no internet, etc.)
        logger.warning("Embedding model unavailable, falling back to TF-IDF only: %r", e)
        return None


def embedding_scores(resume_text: str, job_texts: list, embed_model) -> np.ndarray:
    """Semantic similarity: catches meaning even when wording differs."""
    vectors = embed_model.encode([resume_text] + job_texts, normalize_embeddings=True)
    return cosine_similarity(vectors[0:1], vectors[1:]).flatten()


def normalize_skill(s: str) -> str:
    return s.strip().lower()


# =============================================================================
# 6. FILE READING (resume as pasted text or uploaded .txt / .pdf)
# =============================================================================
def read_uploaded_resume(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RecommenderError("PDF support isn't installed. Run: pip install pypdf")
        reader = PdfReader(BytesIO(uploaded_file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


# =============================================================================
# 7. GROQ CLIENT
# =============================================================================
load_dotenv(BASE_DIR / ".env.txt")
load_dotenv(BASE_DIR / ".env")


def get_client() -> Groq | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Groq API Key", type="password")
        st.sidebar.caption("Or set GROQ_API_KEY in a .env file next to this script.")
    if not api_key:
        return None
    return Groq(api_key=api_key.strip(), timeout=60.0, max_retries=0)


# =============================================================================
# 8. PAGE CONFIG + THEME
#    "Signal / radar" concept: the match score IS the visual - a filled bar
#    whose height encodes strength, like a reading on a meter, rather than
#    a decorative card.
# =============================================================================
st.set_page_config(page_title="AI Career Recommender", page_icon="🧭", layout="centered")

GRAPHITE = "#14171C"
PANEL = "#1E232B"
BORDER = "#2C333D"
TEXT = "#E8EAED"
MUTED = "#8A93A1"
INDIGO = "#6366F1"
AMBER = "#F59E0B"
SAGE = "#34D399"
CLAY = "#C97B63"

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background: {GRAPHITE}; color: {TEXT}; }}
        .block-container {{ padding-top: 2.2rem; max-width: 780px; }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; color: {TEXT}; }}

        .hero-title {{
            font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.2rem;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, {INDIGO}, {AMBER});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .hero-sub {{ color: {MUTED}; margin-bottom: 1.6rem; }}

        section[data-testid="stSidebar"] {{ background: #0F1216; border-right: 1px solid {BORDER}; }}

        div[data-testid="stForm"] {{
            background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px;
            padding: 1.4rem 1.4rem 0.6rem 1.4rem;
        }}
        .stButton > button, .stDownloadButton > button {{
            background: {INDIGO}; color: #fff; border: none; border-radius: 10px; font-weight: 600;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{ background: #4F46E5; }}

        /* Match row: a signal bar whose fill encodes the score */
        .match-row {{ display: flex; gap: 0.9rem; background: {PANEL}; border: 1px solid {BORDER};
                      border-radius: 10px; padding: 0.9rem 1rem; margin-bottom: 0.7rem; }}
        .signal {{ width: 34px; height: 54px; border-radius: 6px; flex-shrink: 0;
                   background: linear-gradient(to top, {AMBER} var(--fill), {BORDER} var(--fill)); }}
        .match-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; }}
        .match-meta {{ color: {MUTED}; font-size: 0.82rem; margin-top: 0.1rem; }}
        .chip {{ display: inline-block; font-size: 0.74rem; padding: 0.08rem 0.5rem;
                 border-radius: 6px; margin: 0.15rem 0.3rem 0 0; }}
        .chip-match {{ background: rgba(52,211,153,0.15); color: {SAGE}; }}
        .chip-missing {{ background: rgba(201,123,99,0.18); color: {CLAY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 9. SIDEBAR - settings, method choice, about
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    client = get_client()
    model = st.selectbox("LLM model", STRICT_MODELS, help="Used for resume parsing and match explanations.")
    top_n = st.slider("Jobs to show", 3, 10, 5)

    st.divider()
    st.markdown("### 🧪 Matching method")
    method = st.radio(
        "How should resumes be matched to jobs?",
        ["Blend (recommended)", "TF-IDF only", "Embeddings only"],
        help="TF-IDF matches shared words. Embeddings match meaning, even with different wording.",
    )
    blend_weight = 0.5
    if method == "Blend (recommended)":
        blend_weight = st.slider("Blend: TF-IDF \u2194 Embeddings", 0.0, 1.0, 0.5,
                                  help="0 = pure TF-IDF, 1 = pure embeddings")

    st.divider()
    with st.expander("ℹ️ About this project"):
        st.markdown(
            "Matches a resume against a job dataset using **TF-IDF** "
            "(classic term-overlap) and **sentence embeddings** (semantic "
            "similarity), blended together. An **LLM** (Groq, structured "
            "outputs) extracts a clean skill profile from the raw resume "
            "text and writes a plain-English explanation for each match."
        )

if client is None:
    st.markdown('<p class="hero-title">🧭 AI Career Recommender</p>', unsafe_allow_html=True)
    st.warning("Enter your Groq API key in the sidebar to get started.")
    st.stop()

embed_model = load_embedding_model()
if method != "TF-IDF only" and embed_model is None:
    st.sidebar.warning("Embedding model unavailable - falling back to TF-IDF only. "
                        "Run `pip install sentence-transformers` to enable it.")

# =============================================================================
# 10. HERO + RESUME INPUT
# =============================================================================
st.markdown('<p class="hero-title">🧭 AI Career Recommender</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Paste or upload a resume - get back ranked job matches, '
    'skill gaps, and a plain-English reason for each one.</p>',
    unsafe_allow_html=True,
)

with st.form("resume_form"):
    resume_text = st.text_area("Paste resume text", height=160,
                                placeholder="Paste your resume, or upload a file below...")
    uploaded = st.file_uploader("...or upload a .txt or .pdf resume", type=["txt", "pdf"])
    submitted = st.form_submit_button("🧭 Find my best-fit jobs")

# =============================================================================
# 11. RUN THE PIPELINE
# =============================================================================
if submitted:
    try:
        if uploaded is not None:
            resume_text = read_uploaded_resume(uploaded)

        with st.spinner("Reading your resume..."):
            profile = extract_skill_profile(resume_text, client, model)

        jobs = load_jobs()
        job_texts = [f"{j['title']}. {j['description']} Skills: {', '.join(j['required_skills'])}" for j in jobs]
        resume_for_matching = resume_text + " " + " ".join(profile.skills)

        with st.spinner("Scoring jobs against your profile..."):
            tfidf = tfidf_scores(resume_for_matching, job_texts)

            if method == "TF-IDF only" or embed_model is None:
                combined = tfidf
                embed = np.zeros_like(tfidf)
            else:
                embed = embedding_scores(resume_for_matching, job_texts, embed_model)
                if method == "Embeddings only":
                    combined = embed
                else:
                    combined = blend_weight * embed + (1 - blend_weight) * tfidf

        ranked = sorted(
            zip(jobs, tfidf, embed, combined), key=lambda x: x[3], reverse=True
        )[:top_n]

        user_skills = {normalize_skill(s) for s in profile.skills}
        results = []
        with st.spinner("Writing explanations..."):
            for job, t_score, e_score, c_score in ranked:
                required = {normalize_skill(s) for s in job["required_skills"]}
                matched = sorted(user_skills & required)
                missing = sorted(required - user_skills)
                reason = explain_match(profile, job, matched, missing, client, model)
                results.append({
                    "job": job, "tfidf": t_score, "embedding": e_score, "combined": c_score,
                    "matched": matched, "missing": missing, "reason": reason,
                })

        st.session_state["profile"] = profile
        st.session_state["results"] = results

    except RecommenderError as e:
        st.error(f"❌ {e}")
    except Exception as e:
        logger.exception("Unexpected error during matching")
        st.error("😕 Something unexpected went wrong. Check the terminal for details.")
        with st.expander("Technical details"):
            st.code(repr(e))

# =============================================================================
# 12. DISPLAY RESULTS
# =============================================================================
if "results" in st.session_state:
    profile: SkillProfile = st.session_state["profile"]
    results: list = st.session_state["results"]

    st.subheader("Your profile")
    st.write(profile.summary)
    st.caption(
        f"{profile.years_experience} experience · "
        f"Past titles: {', '.join(profile.past_titles) or 'none listed'}"
    )
    with st.expander(f"All {len(profile.skills)} extracted skills"):
        st.write(", ".join(profile.skills))

    st.subheader("Best-fit jobs")
    for r in results:
        job, pct = r["job"], round(r["combined"] * 100)
        st.markdown(
            f"""
            <div class="match-row">
                <div class="signal" style="--fill:{max(pct,4)}%;"></div>
                <div>
                    <div class="match-title">{job['title']} &nbsp;
                        <span style="color:{AMBER};font-weight:600;">{pct}% match</span></div>
                    <div class="match-meta">{job['category']} ·
                        TF-IDF {round(r['tfidf']*100)}% · Embeddings {round(r['embedding']*100)}%</div>
                    <div style="margin-top:0.4rem;">
                        {''.join(f'<span class="chip chip-match">{s}</span>' for s in r['matched']) or
                         '<span class="chip chip-missing">no direct skill overlap</span>'}
                        {''.join(f'<span class="chip chip-missing">missing: {s}</span>' for s in r['missing'][:5])}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Why this match?"):
            st.write(r["reason"])

    # --- Export -------------------------------------------------------------
    st.divider()
    report = {
        "profile": profile.model_dump(),
        "matches": [
            {"title": r["job"]["title"], "match_pct": round(r["combined"] * 100),
             "matched_skills": r["matched"], "missing_skills": r["missing"], "reason": r["reason"]}
            for r in results
        ],
    }
    st.download_button(
        "💾 Download full report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"career_matches_{datetime.now():%Y%m%d}.json",
        use_container_width=True,
    )
