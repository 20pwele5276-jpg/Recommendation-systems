"""
Study Resource Recommender
===========================
Portfolio project: turns a learning goal into a structured, ordered list of
study resources (courses, books, videos, articles, podcasts) — beginner to
advanced — using an LLM constrained to a strict JSON schema, validated with
Pydantic.
 
TECH STACK
----------
- Streamlit        - UI
- Groq API         - LLM inference (open-weight gpt-oss models)
- Pydantic         - data validation / the "shape" of a recommendation
- Structured Outputs (JSON Schema, strict mode) - forces the model to
  reply with clean typed data instead of free-form prose
 
Run with:  streamlit run study_recommender.py
"""
 
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
 
import groq
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError
 
# =============================================================================
# 1. THE SHAPE OF A RECOMMENDATION
#    One Pydantic model = one resource. A StudyPlan is a goal summary plus
#    an ordered list of resources (foundational -> advanced).
# =============================================================================
RESOURCE_TYPES = ["Course", "Book", "Video", "Article", "Tutorial", "Podcast"]
LEVELS = ["Beginner", "Intermediate", "Advanced"]
 
 
class Resource(BaseModel):
    title: str
    resource_type: str      # one of RESOURCE_TYPES
    provider: str            # e.g. "Coursera", "O'Reilly", "freeCodeCamp"
    level: str                # one of LEVELS
    estimated_time: str      # e.g. "6 hours", "250 pages"
    why_recommended: str     # tailored to the user's stated goal
    how_to_find: str         # a search hint, e.g. "Search the title on YouTube"
    # NOTE: we deliberately don't ask the model for a clickable URL - LLMs
    # often invent plausible-looking but broken links. A search hint is
    # honest about what the model actually knows.
 
 
class StudyPlan(BaseModel):
    goal_summary: str
    resources: list[Resource]
 
 
def search_link(resource: Resource) -> tuple[str, str]:
    """
    Build a REAL, always-working link in code - never from the model.
    A search-results page can't 404 the way a guessed direct link can, so
    this is how the "how_to_find" hint becomes something actually clickable.
    Returns (label, url).
    """
    query = quote_plus(f"{resource.title} {resource.provider}".strip())
    if resource.resource_type == "Video":
        return "▶️ Search on YouTube", f"https://www.youtube.com/results?search_query={query}"
    if resource.resource_type == "Podcast":
        return "🎧 Search on YouTube", f"https://www.youtube.com/results?search_query={query}"
    return "🔍 Search on Google", f"https://www.google.com/search?q={query}"
 
 
# The hand-written JSON Schema sent to Groq. "enum" keeps resource_type and
# level consistent so the UI can reliably color-code them; "strict" mode
# guarantees the reply always matches this shape exactly.
STUDY_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "goal_summary": {"type": "string"},
        "resources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "resource_type": {"type": "string", "enum": RESOURCE_TYPES},
                    "provider": {"type": "string"},
                    "level": {"type": "string", "enum": LEVELS},
                    "estimated_time": {"type": "string"},
                    "why_recommended": {"type": "string"},
                    "how_to_find": {"type": "string"},
                },
                "required": [
                    "title", "resource_type", "provider", "level",
                    "estimated_time", "why_recommended", "how_to_find",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["goal_summary", "resources"],
    "additionalProperties": False,
}
 
RECOMMEND_PROMPT = """\
You are a thoughtful learning advisor who recommends real, well-known study \
resources (courses, books, videos, articles, tutorials, podcasts).
 
Rules:
- Tailor every recommendation to the learner's stated goal and current level.
- Only recommend resources that genuinely exist and are reasonably well known
  in their field. Do not invent titles.
- Order the list from foundational to advanced, so it reads as a learning path.
- Mix resource types rather than recommending five of the same type, unless
  the learner asked for a specific format only.
- "why_recommended" must reference the learner's specific goal, not generic
  praise like "this is a great resource".
- "how_to_find" should be a short, honest search hint (e.g. "Search the
  title on Coursera" or "Available on O'Reilly and most libraries"), never
  a fabricated URL.
"""
 
# =============================================================================
# 2. ERROR HANDLING
#    Technical detail goes to the terminal log; the user only ever sees a
#    short, friendly message.
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("study_recommender")
 
 
class RecommendationError(Exception):
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
 
# Only these models currently support Groq's strict structured outputs.
STRICT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
 
 
# =============================================================================
# 3. THE CORE FUNCTION: goal + preferences in, a validated StudyPlan out
# =============================================================================
def recommend_resources(
    client: Groq,
    model: str,
    goal: str,
    level: str,
    formats: list[str],
    count: int,
) -> StudyPlan:
    """Ask the LLM for a structured study plan, with retries on transient errors."""
    if not goal.strip():
        raise RecommendationError("Please describe what you want to learn.")
 
    format_line = (
        f"Preferred formats: {', '.join(formats)}."
        if formats else "No format preference - use a sensible mix."
    )
    user_prompt = (
        f"Learning goal: {goal.strip()}\n"
        f"Current level: {level}\n"
        f"{format_line}\n"
        f"Recommend exactly {count} resources."
    )
 
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": RECOMMEND_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "study_plan",
                        "strict": True,
                        "schema": STUDY_PLAN_SCHEMA,
                    },
                },
            )
            raw_json = response.choices[0].message.content
            return StudyPlan.model_validate_json(raw_json)
 
        except RETRYABLE as e:
            last_error = e
            logger.warning("Attempt %d/%d failed: %r", attempt, MAX_ATTEMPTS, e)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
 
        except ValidationError as e:
            logger.error("Model reply failed validation: %r", e)
            raise RecommendationError("The model's answer didn't match the expected format.") from e
 
        except groq.GroqError as e:
            logger.error("Groq error: %r", e)
            raise RecommendationError(explain_error(e)) from e
 
    raise RecommendationError(explain_error(last_error)) from last_error
 
 
# =============================================================================
# 4. EXPORT HELPERS (JSON for machines, Markdown for humans / sharing)
# =============================================================================
def plan_to_markdown(plan: StudyPlan) -> str:
    lines = [f"# Study Plan: {plan.goal_summary}", ""]
    for i, r in enumerate(plan.resources, start=1):
        lines += [
            f"## {i}. {r.title}",
            f"- **Type:** {r.resource_type}  ·  **Provider:** {r.provider}  ·  **Level:** {r.level}",
            f"- **Estimated time:** {r.estimated_time}",
            f"- **Why:** {r.why_recommended}",
            f"- **How to find it:** {r.how_to_find}",
            "",
        ]
    return "\n".join(lines)
 
 
# =============================================================================
# 5. GROQ CLIENT
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
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
# 6. PAGE CONFIG + THEME
#    "Illuminated manuscript" palette: deep plum background, amethyst +
#    emerald as the lead colors, gold / teal / rose as connecting jewel
#    tones used only to color-code resource types (like stained glass).
# =============================================================================
st.set_page_config(page_title="Study Resource Recommender", page_icon="📚", layout="centered")
 
INK = "#1A1025"
CARD = "#241534"
BORDER = "#3A2550"
PARCHMENT = "#F3EFE8"
MUTED = "#B8AFC9"
AMETHYST = "#8B5CF6"
EMERALD = "#10B981"
GOLD = "#D4AF37"
 
TYPE_COLORS = {
    "Course": AMETHYST,
    "Book": EMERALD,
    "Video": GOLD,
    "Article": "#2DD4BF",   # jade teal
    "Tutorial": "#38BDF8",  # azure
    "Podcast": "#F472B6",   # rose
}
 
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
 
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background: {INK}; color: {PARCHMENT}; }}
        .block-container {{ padding-top: 2.2rem; max-width: 760px; }}
        header[data-testid="stHeader"] {{ background: transparent; }}
 
        h1, h2, h3 {{ font-family: 'Lora', serif; color: {PARCHMENT}; }}
 
        .hero-title {{
            font-family: 'Lora', serif; font-weight: 700; font-size: 2.3rem;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, {AMETHYST}, {EMERALD});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .hero-sub {{ color: {MUTED}; margin-bottom: 1.6rem; }}
 
        section[data-testid="stSidebar"] {{ background: #130B1C; border-right: 1px solid {BORDER}; }}
 
        /* Input "desk" card */
        div[data-testid="stForm"] {{
            background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
            padding: 1.4rem 1.4rem 0.6rem 1.4rem;
        }}
 
        .stButton > button, .stDownloadButton > button {{
            background: {AMETHYST}; color: #fff; border: none; border-radius: 10px;
            font-weight: 600;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{ background: #7C3AED; }}
 
        /* Bookshelf row */
        .book-row {{
            background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
            padding: 0.9rem 1rem; margin-bottom: 0.7rem; display: flex; gap: 0.9rem;
        }}
        .book-spine {{ width: 6px; border-radius: 4px; flex-shrink: 0; }}
        .book-title {{ font-family: 'Lora', serif; font-weight: 600; font-size: 1.05rem; color: {PARCHMENT}; }}
        .book-meta {{ color: {MUTED}; font-size: 0.85rem; margin-top: 0.1rem; }}
        .book-badge {{
            display: inline-block; font-size: 0.72rem; font-weight: 600;
            padding: 0.1rem 0.5rem; border-radius: 6px; margin-right: 0.4rem;
            background: rgba(255,255,255,0.08); color: {PARCHMENT};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)
 
# =============================================================================
# 7. SIDEBAR - settings + a short "about" blurb (handy for a portfolio demo)
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    client = get_client()
    model = st.selectbox("Model", STRICT_MODELS, help="Only these support strict structured output.")
    resource_count = st.slider("Number of resources", 3, 10, 5)
 
    st.divider()
    with st.expander("ℹ️ About this project"):
        st.markdown(
            "A learning-path generator that turns a goal into a structured, "
            "typed recommendation list — not free-text the app has to guess "
            "at. Built with **Streamlit**, **Groq** (open-weight gpt-oss "
            "models) and **Pydantic**, using JSON-schema *strict mode* "
            "structured outputs so every response is validated, reliable "
            "data."
        )
 
if client is None:
    st.markdown('<p class="hero-title">📚 Study Resource Recommender</p>', unsafe_allow_html=True)
    st.warning("Enter your Groq API key in the sidebar to get started.")
    st.stop()
 
# =============================================================================
# 8. HERO + INPUT FORM
# =============================================================================
st.markdown('<p class="hero-title">📚 Study Resource Recommender</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Tell it what you want to learn - get back a structured, '
    'ordered reading-and-watching list.</p>',
    unsafe_allow_html=True,
)
 
EXAMPLE_GOALS = [
    "Go from zero to building REST APIs with Python",
    "Understand machine learning well enough to read research papers",
    "Learn enough UX design to redesign my own side project",
]
 
with st.form("goal_form"):
    goal = st.text_area(
        "What do you want to learn?",
        placeholder=EXAMPLE_GOALS[0],
        height=90,
    )
    col1, col2 = st.columns(2)
    with col1:
        level = st.selectbox("Current level", LEVELS)
    with col2:
        formats = st.multiselect("Preferred formats (optional)", RESOURCE_TYPES)
 
    submitted = st.form_submit_button("📖 Build my study plan")
 
if not goal.strip() and not submitted:
    st.caption("Need inspiration? Try one of these:")
    cols = st.columns(1)
    for example in EXAMPLE_GOALS:
        st.caption(f"\u2022 {example}")
 
# =============================================================================
# 9. GENERATE + DISPLAY
# =============================================================================
if submitted:
    try:
        with st.spinner("Pulling resources off the shelf..."):
            plan = recommend_resources(client, model, goal, level, formats, resource_count)
        st.session_state["plan"] = plan
    except RecommendationError as e:
        st.error(f"❌ {e}")
    except Exception as e:
        logger.exception("Unexpected error during recommendation")
        st.error("😕 Something unexpected went wrong. Check the terminal for details.")
        with st.expander("Technical details"):
            st.code(repr(e))
 
if "plan" in st.session_state:
    plan: StudyPlan = st.session_state["plan"]
 
    st.subheader("Your learning path")
    st.write(plan.goal_summary)
 
    for i, r in enumerate(plan.resources, start=1):
        color = TYPE_COLORS.get(r.resource_type, MUTED)
        st.markdown(
            f"""
            <div class="book-row">
                <div class="book-spine" style="background:{color};"></div>
                <div>
                    <div class="book-title">{i}. {r.title}</div>
                    <div class="book-meta">
                        <span class="book-badge" style="color:{color};">{r.resource_type}</span>
                        <span class="book-badge">{r.level}</span>
                        {r.provider} · {r.estimated_time}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Why this, and how to find it"):
            st.write(f"**Why:** {r.why_recommended}")
            st.write(f"**How to find it:** {r.how_to_find}")
            label, url = search_link(r)
            st.link_button(label, url)
 
    # --- Export -------------------------------------------------------------
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "💾 Download as JSON",
            data=plan.model_dump_json(indent=2),
            file_name=f"study_plan_{datetime.now():%Y%m%d}.json",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "💾 Download as Markdown",
            data=plan_to_markdown(plan),
            file_name=f"study_plan_{datetime.now():%Y%m%d}.md",
            use_container_width=True,
        )
 
