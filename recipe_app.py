import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv
import groq
from groq import Groq

# ---------------------------------------------------------------------------
# Logging (errors go to your terminal, friendly messages go to the UI)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("recipe_app")


# ---------------------------------------------------------------------------
# Custom exception: carries a message that is safe to show to the user
# ---------------------------------------------------------------------------
class RecipeError(Exception):
    """Raised when recipe generation fails for a reason we can explain."""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env.txt"))
load_dotenv(os.path.join(base_dir, ".env"))

st.set_page_config(page_title="Recipe Generator", page_icon="🍲", layout="centered")

st.title("🍲 AI Recipe Generator")
st.caption("Powered by Groq")

# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------
api_key = os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.header("Settings")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password")
        st.caption("Or set GROQ_API_KEY in a .env file next to this script.")
    else:
        st.success("API key loaded from .env")

if not api_key or not api_key.strip():
    st.warning("Enter your Groq API key in the sidebar to get started.")
    st.stop()

api_key = api_key.strip()

# timeout: don't hang forever. max_retries=0: we do our own retries below.
client = Groq(api_key=api_key, timeout=30.0, max_retries=0)

# ---------------------------------------------------------------------------
# Ask Groq which models this key can actually use (names change often)
# ---------------------------------------------------------------------------
NON_CHAT = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "distil-whisper")
PREFERRED = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")


def explain_error(e: Exception) -> str:
    """Turn any Groq/network exception into a friendly, actionable message."""
    if isinstance(e, groq.AuthenticationError):
        return "Your API key was rejected. Check that GROQ_API_KEY is correct and still active."
    if isinstance(e, groq.PermissionDeniedError):
        return "Your API key doesn't have permission to use this model or feature."
    if isinstance(e, groq.NotFoundError):
        return "That model isn't available for your account. Pick a different model in the sidebar."
    if isinstance(e, groq.RateLimitError):
        return "Rate limit reached. Wait a minute and try again, or choose a smaller model."
    if isinstance(e, groq.BadRequestError):
        return "The request was rejected as invalid. Try shorter or simpler ingredients."
    if isinstance(e, groq.APITimeoutError):
        return "The request timed out. Please try again."
    if isinstance(e, groq.APIConnectionError):
        return "Couldn't reach Groq. Check your internet connection and try again."
    if isinstance(e, groq.InternalServerError):
        return "Groq's servers had a problem. Please try again in a moment."
    if isinstance(e, groq.APIStatusError):
        return f"Groq returned an error (HTTP {e.status_code}). Please try again."
    return f"Unexpected error: {e}"


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
except Exception as e:  # anything we didn't anticipate
    logger.exception("Unexpected error while listing models")
    st.sidebar.error(f"Unexpected error: {e}")
    available_models = []

if not available_models:
    st.error("No models available. Check your API key and internet connection, then reload.")
    st.stop()

with st.sidebar:
    MODEL = st.selectbox("Model", available_models)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
MAX_INGREDIENT_CHARS = 500


def validate_inputs(ingredients: str, cuisine: str, diet: str):
    """Return cleaned (ingredients, cuisine, diet) or raise ValueError."""
    ingredients = (ingredients or "").strip()
    if not ingredients:
        raise ValueError("Please enter at least one ingredient.")
    if len(ingredients) > MAX_INGREDIENT_CHARS:
        raise ValueError(
            f"Ingredient list is too long ({len(ingredients)} chars). "
            f"Keep it under {MAX_INGREDIENT_CHARS}."
        )
    if not any(ch.isalpha() for ch in ingredients):
        raise ValueError("Ingredients should contain some words, not just numbers/symbols.")

    cuisine = (cuisine or "").strip() or "Any"
    diet = (diet or "").strip() or "Any"
    return ingredients, cuisine, diet


# ---------------------------------------------------------------------------
# Recipe generation with retries + error handling
# ---------------------------------------------------------------------------
RETRYABLE = (
    groq.APIConnectionError,   # includes timeouts
    groq.RateLimitError,
    groq.InternalServerError,
)
MAX_ATTEMPTS = 3


def generate_recipe(ingredients: str, cuisine: str, diet: str, max_words: int = 150) -> str:
    prompt = (
        f"Generate a food recipe using these ingredients: {ingredients}.\n"
        f"Cuisine: {cuisine}.\n"
        f"Diet: {diet}.\n"
        f"Keep the recipe to no more than {max_words} words. "
        f"Include a short ingredient list and numbered steps."
    )

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful chef who writes clear, concise recipes."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            # The call worked, but make sure we actually got text back.
            if not completion.choices:
                raise RecipeError("The model returned no answer. Please try again.")
            text = completion.choices[0].message.content
            if not text or not text.strip():
                raise RecipeError("The model returned an empty recipe. Please try again.")
            return text

        except RETRYABLE as e:
            last_error = e
            logger.warning("Attempt %d/%d failed: %r", attempt, MAX_ATTEMPTS, e)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))  # wait 1s, then 2s
            continue

        except RecipeError:
            raise  # already friendly, pass straight through

        except groq.GroqError as e:
            # Non-retryable API errors (bad key, model not found, bad request...)
            logger.error("Groq error: %r", e)
            raise RecipeError(explain_error(e)) from e

    # All retries used up
    raise RecipeError(explain_error(last_error)) from last_error


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.subheader("Tell me what you've got")

ingredients = st.text_area(
    "Ingredients (comma-separated)",
    value="chicken, rice, onions, garlic, ginger, tomatoes, yogurt, spices",
    height=80,
)

col1, col2 = st.columns(2)
with col1:
    cuisine = st.selectbox(
        "Cuisine",
        ["Any", "Mexican", "Indian", "Pakistani", "Italian", "Chinese", "Thai", "Mediterranean", "American", "Other"],
    )
    if cuisine == "Other":
        cuisine = st.text_input("Enter cuisine", value="")
with col2:
    diet = st.selectbox(
        "Diet",
        ["Any", "Vegetarian", "Vegan", "Non-vegetarian", "Gluten-free", "Keto", "Other"],
    )
    if diet == "Other":
        diet = st.text_input("Enter diet", value="")

max_words = st.slider("Max recipe length (words)", 50, 400, 150, step=25)

if st.button("🍳 Generate Recipe", type="primary"):
    started = time.time()
    try:
        clean_ing, clean_cuisine, clean_diet = validate_inputs(ingredients, cuisine, diet)

        with st.spinner("Cooking up your recipe..."):
            recipe = generate_recipe(clean_ing, clean_cuisine, clean_diet, max_words)

    except ValueError as e:
        # Bad user input
        st.warning(str(e))

    except RecipeError as e:
        # Known failure with a friendly message
        st.error(f"❌ {e}")

    except Exception as e:
        # Safety net for anything unexpected: log details, show a generic message
        logger.exception("Unhandled error while generating recipe")
        st.error("😕 Something unexpected went wrong. Check the terminal for details.")
        with st.expander("Technical details"):
            st.code(repr(e))

    else:
        # Runs only if no exception happened
        st.subheader("Your Recipe")
        st.markdown(recipe)
        st.download_button("💾 Download recipe", recipe, file_name="recipe.txt")

    finally:
        # Runs no matter what
        logger.info("Request finished in %.2fs", time.time() - started)
