import logging
import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
import groq
from groq import Groq

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("chat_app")


class ChatError(Exception):
    """Raised when a chat request fails for a reason we can explain to the user."""


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Groq Chat", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; max-width: 820px;}
        .app-title {font-size: 2rem; font-weight: 700; margin-bottom: 0;}
        .app-sub {color: #8a8f98; margin-top: 0; margin-bottom: 1.2rem;}
        div[data-testid="stChatMessage"] {border-radius: 14px; padding: 0.6rem 0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env.txt"))
load_dotenv(os.path.join(base_dir, ".env"))

api_key = os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.header("⚙️ Settings")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password")
        st.caption("Or set GROQ_API_KEY in a .env file next to this script.")
    else:
        st.success("API key loaded from .env")

if not api_key or not api_key.strip():
    st.markdown('<p class="app-title">💬 Groq Chat</p>', unsafe_allow_html=True)
    st.warning("Enter your Groq API key in the sidebar to get started.")
    st.stop()

api_key = api_key.strip()
client = Groq(api_key=api_key, timeout=60.0, max_retries=0)

# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------
RETRYABLE = (groq.APIConnectionError, groq.RateLimitError, groq.InternalServerError)
MAX_ATTEMPTS = 3


def explain_error(e: Exception) -> str:
    if isinstance(e, groq.AuthenticationError):
        return "Your API key was rejected. Check that GROQ_API_KEY is correct and still active."
    if isinstance(e, groq.PermissionDeniedError):
        return "Your API key doesn't have permission to use this model."
    if isinstance(e, groq.NotFoundError):
        return "That model isn't available for your account. Pick a different model in the sidebar."
    if isinstance(e, groq.RateLimitError):
        return "Rate limit reached. Wait a minute and try again, or choose a smaller model."
    if isinstance(e, groq.BadRequestError):
        return "The request was rejected. Try a shorter message or reduce the memory length."
    if isinstance(e, groq.APITimeoutError):
        return "The request timed out. Please try again."
    if isinstance(e, groq.APIConnectionError):
        return "Couldn't reach Groq. Check your internet connection and try again."
    if isinstance(e, groq.InternalServerError):
        return "Groq's servers had a problem. Please try again in a moment."
    if isinstance(e, groq.APIStatusError):
        return f"Groq returned an error (HTTP {e.status_code}). Please try again."
    return f"Unexpected error: {e}"


# ---------------------------------------------------------------------------
# Model list (asked from Groq, since model names change often)
# ---------------------------------------------------------------------------
NON_CHAT = ("whisper", "tts", "guard", "orpheus", "playai", "embed")
PREFERRED = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")


@st.cache_data(ttl=600, show_spinner=False)
def get_chat_models(key: str):
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
    logger.exception("Unexpected error while listing models")
    st.sidebar.error(f"Unexpected error: {e}")
    available_models = []

if not available_models:
    st.error("No models available. Check your API key and internet connection, then reload.")
    st.stop()

# ---------------------------------------------------------------------------
# Session state (Streamlit reruns the script on every interaction, so a global
# list would be wiped each time. session_state is what keeps the memory.)
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role", "content", "caption"?}]
if "pending" not in st.session_state:
    st.session_state.pending = None  # prompt queued by a suggestion button

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
PERSONAS = {
    "Helpful assistant": "You are a helpful, friendly assistant. Answer clearly and accurately.",
    "Concise expert": "You are a concise expert. Give short, precise answers with no filler.",
    "Patient tutor": "You are a patient tutor. Explain step by step, use simple examples, and check understanding.",
    "Creative writer": "You are a creative writing partner with vivid, imaginative language.",
    "Custom": "",
}

with st.sidebar:
    model = st.selectbox("Model", available_models)
    persona = st.selectbox("Personality", list(PERSONAS.keys()))
    if persona == "Custom":
        system_prompt = st.text_area("System prompt", value="You are a helpful assistant.", height=100)
    else:
        system_prompt = PERSONAS[persona]

    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider(
        "Max reply length (tokens)", 256, 8192, 2048, 256,
        help="Reasoning models spend some of these tokens thinking, so don't set this too low.",
    )
    memory_len = st.slider(
        "Memory (recent messages sent)", 2, 40, 20, 2,
        help="How many recent messages the model can 'remember'. Older ones are dropped to save tokens.",
    )

    st.divider()
    n_msgs = len(st.session_state.messages)
    st.caption(f"💭 {n_msgs} message{'s' if n_msgs != 1 else ''} in this chat")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear", use_container_width=True, disabled=n_msgs == 0):
            st.session_state.messages = []
            st.session_state.pending = None
            st.rerun()
    with c2:
        transcript = "\n\n".join(
            f"{'You' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in st.session_state.messages
        )
        st.download_button(
            "💾 Export",
            data=transcript or "No messages yet.",
            file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.txt",
            use_container_width=True,
            disabled=n_msgs == 0,
        )


# ---------------------------------------------------------------------------
# Streaming reply (with retries before the stream starts)
# ---------------------------------------------------------------------------
def build_api_messages() -> list:
    """System prompt + the most recent N messages, stripped to what the API accepts."""
    history = st.session_state.messages[-memory_len:]
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    if system_prompt.strip():
        msgs.insert(0, {"role": "system", "content": system_prompt.strip()})
    return msgs


def stream_reply(api_messages: list):
    """Yield text chunks. Raises ChatError with a friendly message on failure."""
    stream, last_error = None, None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            break
        except RETRYABLE as e:
            last_error = e
            logger.warning("Attempt %d/%d failed: %r", attempt, MAX_ATTEMPTS, e)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
        except groq.GroqError as e:
            logger.error("Groq error: %r", e)
            raise ChatError(explain_error(e)) from e

    if stream is None:
        raise ChatError(explain_error(last_error)) from last_error

    got_text = False
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                got_text = True
                yield delta
    except groq.GroqError as e:
        logger.error("Stream interrupted: %r", e)
        raise ChatError(explain_error(e)) from e

    if not got_text:
        raise ChatError(
            "The model returned an empty reply. Try raising 'Max reply length' in the sidebar."
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="app-title">💬 Groq Chat</p>', unsafe_allow_html=True)
st.markdown(f'<p class="app-sub">Model: <code>{model}</code> · Personality: {persona}</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Empty state: suggestion buttons (the last two show off conversation memory)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.info("👋 Ask me anything. I remember what we've said earlier in this chat.")
    suggestions = [
        "Explain how a neural network learns, simply",
        "Give me 3 ideas for a weekend project",
        "Hi, my name is Wajiha. Please remember my name.",
        "What is my name?",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state.pending = s
            st.rerun()

# ---------------------------------------------------------------------------
# Replay history
# ---------------------------------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])
        if m.get("caption"):
            st.caption(m["caption"])

# ---------------------------------------------------------------------------
# Handle new input
# ---------------------------------------------------------------------------
user_input = st.chat_input("Type your message...")
if not user_input and st.session_state.pending:
    user_input, st.session_state.pending = st.session_state.pending, None

if user_input:
    user_input = user_input.strip()

    if not user_input:
        st.warning("Please type a message first.")
    elif len(user_input) > 8000:
        st.warning("That message is too long. Please keep it under 8,000 characters.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        started = time.time()
        full_text, failed = "", False

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking..._")
            try:
                for piece in stream_reply(build_api_messages()):
                    full_text += piece
                    placeholder.markdown(full_text + "▌")   # typing cursor
                placeholder.markdown(full_text)

            except ChatError as e:
                failed = True
                if full_text:   # keep whatever arrived before the failure
                    placeholder.markdown(full_text)
                    st.warning(f"The reply was cut off: {e}")
                else:
                    placeholder.empty()
                    st.error(f"❌ {e}")

            except Exception as e:
                failed = True
                logger.exception("Unhandled error during chat")
                placeholder.empty()
                st.error("😕 Something unexpected went wrong. Check the terminal for details.")
                with st.expander("Technical details"):
                    st.code(repr(e))

            else:
                elapsed = time.time() - started
                caption = f"⏱ {elapsed:.1f}s · {model}"
                st.caption(caption)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_text, "caption": caption}
                )

            finally:
                logger.info("Chat request finished in %.2fs (failed=%s)", time.time() - started, failed)

        # If nothing usable came back, drop the unanswered user message so the
        # history stays clean and the user can simply try again.
        if failed and not full_text:
            st.session_state.messages.pop()
        elif failed and full_text:
            st.session_state.messages.append(
                {"role": "assistant", "content": full_text, "caption": "⚠️ partial reply"}
            )
        else:
            st.rerun()  # refresh sidebar counters / export button
