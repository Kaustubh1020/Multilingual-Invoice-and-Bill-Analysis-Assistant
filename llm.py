from __future__ import annotations

import os
import time
import logging

from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # loads .env file if present

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not _API_KEY:
    logger.warning(
        "GROQ_API_KEY not found in environment variables. "
        "Set it via `export GROQ_API_KEY=your_key` or a .env file."
    )

client = Groq(api_key=_API_KEY)

# Model choices — fast, free, and multilingual
MODEL_FAST = "llama-3.1-8b-instant"
MODEL_QUALITY = "llama-3.3-70b-versatile"

DEFAULT_MODEL = MODEL_FAST

# ---------------------------------------------------------------------------
# Core LLM helper
# ---------------------------------------------------------------------------
def ask_llm(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    retries: int = 3,
) -> str:
    """Send a prompt to the Groq API and return the response text.

    Features:
    - Optional system message for role-setting
    - Configurable temperature (lower = more deterministic)
    - Exponential-backoff retries on transient failures
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            wait = 2 ** attempt  # 2s, 4s, 8s
            logger.warning("Groq API attempt %d/%d failed: %s — retrying in %ds", attempt, retries, exc, wait)
            time.sleep(wait)

    logger.error("All %d Groq API attempts failed.", retries)
    return f"⚠️ LLM request failed after {retries} attempts: {last_error}"
