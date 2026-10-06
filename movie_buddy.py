"""
🎬 CineBuddy - a movie recommendation chatbot that remembers your taste.

How the memory works
--------------------
Language models have no memory of their own. Every request must include the
whole conversation so far. We keep that conversation in a list
(`st.session_state.conversation`) and send it with each call, so when you ask
"Why did you pick that one?" the model can see what you said earlier.

Run with:  streamlit run movie_buddy.py
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

import groq
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# =============================================================================
# 1. PAGE SETUP & RED / BLACK THEME
# =============================================================================
st.set_page_config(page_title="CineBuddy", page_icon="🎬", layout="centered")

RED = "#E50914"
DARK_RED = "#8B0000"
BLACK = "#0A0A0A"
CARD = "#151515"
BORDER = "#2B2B2B"

st.markdown(
    f"""
    <style>
        /* ---- Base ---- */
        .stApp {{ background: {BLACK}; color: #F5F5F5; }}
        .block-container {{ padding-top: 2rem; max-width: 820px; }}
        header[data-testid="stHeader"] {{ background: transparent; }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background: #050505;
            border-right: 2px solid {DARK_RED};
        }}

        /* ---- Title ---- */
        .cine-title {{
            font-size: 2.4rem; font-weight: 800; letter-spacing: 1px; margin: 0;
            background: linear-gradient(90deg, {RED}, #ff6b6b);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .cine-sub {{ color: #9a9a9a; margin: 0 0 1.2rem 0; }}

        /* ---- Chat bubbles ---- */
        div[data-testid="stChatMessage"] {{
            background: {CARD}; border: 1px solid {BORDER};
            border-radius: 14px; padding: 0.7rem 0.9rem;
        }}
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
            background: #2a0508; border: 1px solid {DARK_RED};
        }}

        /* ---- Buttons: outlined red, primary = filled red ---- */
        .stButton > button, .stDownloadButton > button {{
            background: transparent; color: #F5F5F5;
            border: 1px solid {RED}; border-radius: 10px;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background: {DARK_RED}; border-color: {RED}; color: #fff;
        }}
        button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
            background: {RED} !important; color: #fff !important; border: none !important;
        }}
        button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {{
            background: #b20710 !important;
        }}

        /* ---- Inputs ---- */
        div[data-testid="stChatInput"] {{ border: 1px solid {RED}; border-radius: 12px; }}
        div[role="slider"] {{ background-color: {RED} !important; }}
        hr {{ border-color: {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 2. LOGGING & CUSTOM ERROR
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("cinebuddy")


class ChatError(Exception):
    """A failure we can explain to the user in plain language."""


def explain_error(e: Exception) -> str:
    """Translate Groq/network exceptions into friendly, actionable messages."""
    if isinstance(e, groq.AuthenticationError):
        return "Your API key was rejected. Check GROQ_API_KEY."
    if isinstance(e, groq.PermissionDeniedError):
        return "Your API key can't use this model."
    if isinstance(e, groq.NotFoundError):
        return "That model isn't available for your account. Pick another one in the sidebar."
    if isinstance(e, groq.RateLimitError):
        return "Rate limit reached. Wait a minute or choose a smaller model."
    if isinstance(e, groq.BadRequestError):
        return "The request was rejected. Try a shorter message."
    if isinstance(e, groq.APITimeoutError):
        return "The request timed out. Please try again."
    if isinstance(e, groq.APIConnectionError):
        return "Couldn't reach Groq. Check your internet connection."
    if isinstance(e, groq.InternalServerError):
        return "Groq's servers had a problem. Try again shortly."
    if isinstance(e, groq.APIStatusError):
        return f"Groq returned an error (HTTP {e.status_code})."
    return f"Unexpected error: {e}"


# =============================================================================
# 3. BOT CONFIG (bonus 2: personality)
#    Groq's equivalent of google-genai's `types.GenerateContentConfig`:
#    a system instruction (the personality) plus generation settings.
# =============================================================================
BASE_INSTRUCTIONS = (
    "You are CineBuddy, a movie recommendation assistant.\n"
    "Rules:\n"
    "1. Remember every preference the user states: genres they love, genres "
    "they hate, favourite films, and mood.\n"
    "2. NEVER recommend a film from a genre the user said they dislike.\n"
    "3. Recommend ONE specific film at a time unless asked for more. Give the "
    "title, year, and one or two sentences on why it fits.\n"
    "4. When asked why you chose something, explain by explicitly referring to "
    "what the user told you earlier in the conversation.\n"
    "5. Keep replies under 120 words unless asked for more."
)

PERSONALITIES = {
    "🎞️ Cinephile Buddy": "Personality: warm, enthusiastic film nerd who loves sharing trivia.",
    "😏 Sassy Critic": "Personality: witty, playfully sarcastic critic with strong opinions, but always kind.",
    "🛋️ Cozy Friend": "Personality: relaxed, gentle friend who makes movie night feel comfy.",
    "🕵️ Noir Detective": "Personality: hard-boiled 1940s detective narrating recommendations like a case file.",
}


@dataclass
class GenConfig:
    """Everything that shapes how the bot responds."""
    system_instruction: str
    temperature: float = 0.7
    max_tokens: int = 2048


def build_config(personality: str, temperature: float, max_tokens: int) -> GenConfig:
    """Combine the base rules with the chosen personality into one config."""
    instruction = f"{BASE_INSTRUCTIONS}\n\n{PERSONALITIES[personality]}"
    return GenConfig(system_instruction=instruction, temperature=temperature, max_tokens=max_tokens)


# =============================================================================
# 4. API KEY, CLIENT & MODEL LIST
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env.txt"))
load_dotenv(os.path.join(base_dir, ".env"))

api_key = os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.markdown("## 🎬 CineBuddy")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password")
        st.caption("Or put GROQ_API_KEY in a .env file next to this script.")
    else:
        st.success("API key loaded", icon="🔑")

if not api_key or not api_key.strip():
    st.markdown('<p class="cine-title">🎬 CineBuddy</p>', unsafe_allow_html=True)
    st.warning("Enter your Groq API key in the sidebar to get started.")
    st.stop()

api_key = api_key.strip()
client = Groq(api_key=api_key, timeout=60.0, max_retries=0)  # we retry ourselves

NON_CHAT = ("whisper", "tts", "guard", "orpheus", "playai", "embed")
PREFERRED = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")


@st.cache_data(ttl=600, show_spinner=False)
def get_chat_models(key: str) -> list:
    """Ask Groq which chat models this key can use (names change often)."""
    ids = [m.id for m in Groq(api_key=key, timeout=15.0, max_retries=0).models.list().data]
    ids = [i for i in ids if not any(x in i.lower() for x in NON_CHAT)]
    return sorted(ids, key=lambda i: (i not in PREFERRED, i))


try:
    available_models = get_chat_models(api_key)
except groq.GroqError as e:
    logger.error("Model list failed: %r", e)
    st.sidebar.error(explain_error(e))
    available_models = []
except Exception as e:
    logger.exception("Unexpected error listing models")
    st.sidebar.error(f"Unexpected error: {e}")
    available_models = []

if not available_models:
    st.error("No models available. Check your API key and connection, then reload.")
    st.stop()

# =============================================================================
# 5. SESSION STATE (Streamlit reruns the script on every click, so anything
#    that must survive lives in st.session_state)
# =============================================================================
st.session_state.setdefault("conversation", [])  # the bot's memory
st.session_state.setdefault("queue", [])         # prompts waiting to be sent


def queue_prompt(*prompts: str) -> None:
    """Add one or more prompts to be sent, one chat call at a time."""
    st.session_state.queue.extend(prompts)


def reset_chat() -> None:
    """Bonus 1: wipe the memory and start a brand new conversation."""
    st.session_state.conversation = []
    st.session_state.queue = []


# =============================================================================
# 6. THE chat() FUNCTION (reused from the previous exercise, now with
#    config, retries and error handling)
# =============================================================================
RETRYABLE = (groq.APIConnectionError, groq.RateLimitError, groq.InternalServerError)
MAX_ATTEMPTS = 3
MAX_HISTORY = 40  # only the most recent messages are sent, to stay within token limits


def _ask_groq(messages: list, model: str, config: GenConfig) -> str:
    """One API call with retries. Returns reply text or raises ChatError."""
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            if not response.choices:
                raise ChatError("The model returned no answer. Please try again.")
            text = response.choices[0].message.content
            if not text or not text.strip():
                raise ChatError("The reply was empty. Try raising 'Max reply length'.")
            return text
        except RETRYABLE as e:
            last_error = e
            logger.warning("Attempt %d/%d failed: %r", attempt, MAX_ATTEMPTS, e)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))  # back off: 1s, then 2s
        except ChatError:
            raise
        except groq.GroqError as e:
            logger.error("Groq error: %r", e)
            raise ChatError(explain_error(e)) from e
    raise ChatError(explain_error(last_error)) from last_error


def chat(user_message: str, model: str, config: GenConfig) -> str:
    """Add the user's message to memory, get a reply, store it, and return it."""
    conversation = st.session_state.conversation

    # 1) Add user input to the conversation
    conversation.append({"role": "user", "content": user_message})

    # 2) Build the payload: system instruction + recent history (extra keys removed)
    history = [{"role": m["role"], "content": m["content"]} for m in conversation[-MAX_HISTORY:]]
    messages = [{"role": "system", "content": config.system_instruction}] + history

    try:
        # 3) Call the API
        assistant_reply = _ask_groq(messages, model, config)
    except Exception:
        conversation.pop()  # undo, so a failed turn doesn't pollute the memory
        raise

    # 4) Add the assistant's reply to the conversation
    conversation.append({"role": "assistant", "content": assistant_reply})
    return assistant_reply


# =============================================================================
# 7. SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.divider()
    model = st.selectbox("Model", available_models)
    personality = st.selectbox("Buddy personality", list(PERSONALITIES.keys()))
    temperature = st.slider("Creativity", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider(
        "Max reply length (tokens)", 256, 8192, 2048, 256,
        help="Reasoning models spend some tokens thinking, so don't set this too low.",
    )
    config = build_config(personality, temperature, max_tokens)

    st.divider()
    n = len(st.session_state.conversation)
    st.caption(f"🧠 Memory: {n} message{'s' if n != 1 else ''}")

    st.button("🔄 New chat", type="primary", use_container_width=True,
              on_click=reset_chat, help="Empties the memory and starts fresh")

    transcript = "\n\n".join(
        f"{'You' if m['role'] == 'user' else 'CineBuddy'}: {m['content']}"
        for m in st.session_state.conversation
    )
    st.download_button(
        "💾 Export chat", data=transcript or "No messages yet.",
        file_name=f"cinebuddy_{datetime.now():%Y%m%d_%H%M%S}.txt",
        use_container_width=True, disabled=n == 0,
    )

    with st.expander("🔍 Memory inspector"):
        st.caption("This is exactly what the model receives on the next call:")
        st.json(
            [{"role": "system", "content": config.system_instruction[:80] + "..."}]
            + [{"role": m["role"], "content": m["content"]} for m in st.session_state.conversation]
        )

# =============================================================================
# 8. MAIN PAGE
# =============================================================================
st.markdown('<p class="cine-title">🎬 CineBuddy</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="cine-sub">Your movie-night buddy who remembers your taste · {personality}</p>',
    unsafe_allow_html=True,
)

# The three-step memory test from the exercise
DEMO_STEPS = [
    "I love science fiction and thrillers, but I can't stand horror movies.",
    "Suggest me a movie to watch tonight.",
    "Why did you pick that one?",
]

# Empty state: guide the user through the memory test
if not st.session_state.conversation and not st.session_state.queue:
    st.info("👋 Tell me what you like and I'll remember it. Try the 3-step memory test below, one message at a time:")
    for i, step in enumerate(DEMO_STEPS, start=1):
        st.button(f"{i}. {step}", key=f"step_{i}", use_container_width=True,
                  on_click=queue_prompt, args=(step,))
    st.button("▶️ Run all 3 steps automatically", type="primary", use_container_width=True,
              on_click=queue_prompt, args=tuple(DEMO_STEPS))

# Replay the conversation
for m in st.session_state.conversation:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🎬"):
        st.markdown(m["content"])
        if m.get("caption"):
            st.caption(m["caption"])

# =============================================================================
# 9. HANDLE INPUT (typed messages and queued messages, one call at a time)
# =============================================================================
typed = st.chat_input("Tell me what you like to watch...")
if typed:
    queue_prompt(typed)

if st.session_state.queue:
    prompt = st.session_state.queue.pop(0).strip()

    if not prompt:
        st.warning("Please type a message first.")
    elif len(prompt) > 4000:
        st.warning("That message is too long. Keep it under 4,000 characters.")
    else:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🎬"):
            started = time.time()
            try:
                with st.spinner("CineBuddy is picking something good..."):
                    reply = chat(prompt, model, config)
            except ChatError as e:
                st.session_state.queue.clear()
                st.error(f"❌ {e}")
            except Exception as e:
                st.session_state.queue.clear()
                logger.exception("Unhandled error during chat")
                st.error("😕 Something unexpected went wrong. Check the terminal for details.")
                with st.expander("Technical details"):
                    st.code(repr(e))
            else:
                # Attach a small caption, then rerun so the sidebar and the next
                # queued step (if any) are handled cleanly.
                st.session_state.conversation[-1]["caption"] = f"⏱ {time.time() - started:.1f}s · {model}"
                st.rerun()
