

import os
import logging
from openai import OpenAI, APIError
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()
# Setup basic logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration from environment
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY  = os.getenv("NIM_API_KEY", "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")
NIM_CHAT_MODEL  = os.getenv("NIM_CHAT_MODEL","qwen/qwen3-next-80b-a3b-instruct")
NIM_EMBED_MODEL = os.getenv("NIM_EMBED_MODEL", "nv-embedqa-e5-v5")


if not NIM_API_KEY:
    logger.warning(
        "NIM_API_KEY is not set. Calls to chat_completion/embedding will fail "
        "until you provide a valid key."
    )

# Single shared client instance
_client = OpenAI(
    base_url=NIM_BASE_URL,
    api_key=NIM_API_KEY or "DUMMY-KEY",  # avoids immediate constructor error
)


def chat_completion(
    messages: List[Dict[str, Any]],
    model_id:  None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Call the chat-completion endpoint and return a plain dict.
    """
    model = model_id or NIM_CHAT_MODEL
    logger.info("Calling NIM chat model '%s' with %d messages", model, len(messages))

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **extra,
    }

    try:
        resp = _client.chat.completions.create(**payload)
        return resp.to_dict()
    except APIError as e:
        try:
            err_body = e.response.json()
        except Exception:
            err_body = None
        logger.error(
            "Error calling NIM chat model '%s': %s. Body: %s",
            model,
            e,
            err_body,
        )
        raise RuntimeError(f"NIM chat failed: {e} – {err_body}") from e


def embedding(
    texts: List[str],
    model_id: None,
    **extra: Any,
) -> List[List[float]]:
    """
    Call the embedding endpoint and return a list of vectors.
    """
    model = model_id or NIM_EMBED_MODEL
    if not model:
        raise RuntimeError(
            "No embedding model configured. Set NIM_EMBED_MODEL in your environment."
        )

    logger.info(
        "Calling NIM embedding model '%s' for %d text(s)", model, len(texts)
    )

    payload: Dict[str, Any] = {
        "model": model,
        "input": texts,
        **extra,
    }

    try:
        response = _client.embeddings.create(**payload)
        resp_dict = response.to_dict()
        data = resp_dict.get("data", [])
        return [item["embedding"] for item in data]
    except APIError as e:
        try:
            err_body = e.response.json()
        except Exception:
            err_body = None
        logger.error(
            "Error calling NIM embedding model '%s': %s. Body: %s",
            model,
            e,
            err_body,
        )
        raise RuntimeError(f"NIM embedding failed: {e} – {err_body}") from e
