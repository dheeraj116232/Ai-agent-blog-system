from __future__ import annotations

import base64
import json
import operator
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field, ValidationError

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import logging
import database

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Backend")

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


ALLOWED_BLOG_KINDS = {"explainer", "tutorial", "news_roundup", "comparison", "system_design"}


def _normalize_task_dict(task: dict, index: int) -> dict:
    if not isinstance(task, dict):
        task = {}

    if task.get("id") is None:
        if "task_number" in task:
            try:
                task["id"] = int(task["task_number"])
            except Exception:
                task["id"] = index + 1
        else:
            task["id"] = index + 1

    if not task.get("title"):
        task["title"] = str(task.get("goal") or task.get("task_title") or f"Task {task['id']}").strip()

    if not task.get("goal"):
        task["goal"] = task["title"]

    bullets = task.get("bullets")
    if isinstance(bullets, str):
        bullets = [line.strip(" -*") for line in bullets.splitlines() if line.strip()]
    if not isinstance(bullets, list):
        bullets = []
    bullets = [str(item).strip() for item in bullets if str(item).strip()]
    while len(bullets) < 3:
        bullets.append("Detail missing.")
    task["bullets"] = bullets[:6]

    target_words = task.get("target_words")
    if not isinstance(target_words, int):
        try:
            target_words = int(target_words)
        except Exception:
            target_words = 250
    task["target_words"] = max(120, min(550, target_words))

    task.setdefault("tags", [])
    task.setdefault("requires_research", False)
    task.setdefault("requires_citations", False)
    task.setdefault("requires_code", False)

    return task


def _normalize_plan_dict(plan: dict, topic: str, mode: str) -> dict:
    if not isinstance(plan, dict):
        plan = {}

    blog_kind = plan.get("blog_kind")
    if blog_kind not in ALLOWED_BLOG_KINDS:
        plan["blog_kind"] = "news_roundup" if mode == "open_book" else "explainer"

    plan["blog_title"] = str(plan.get("blog_title") or f"{topic} explained").strip()
    plan["audience"] = str(plan.get("audience") or "technical readers").strip()
    plan["tone"] = str(plan.get("tone") or "informative").strip()
    plan.setdefault("constraints", [])

    raw_tasks = plan.get("tasks")
    if not isinstance(raw_tasks, list):
        raw_tasks = []

    normalized_tasks = [_normalize_task_dict(task, idx) for idx, task in enumerate(raw_tasks)]
    if not normalized_tasks:
        normalized_tasks = [
            {
                "id": 1,
                "title": f"Introduction to {topic}",
                "goal": f"Introduce {topic} and explain why it matters.",
                "bullets": [
                    "Define the topic.",
                    "Summarize the main idea.",
                    "Describe why it is important for the reader.",
                ],
                "target_words": 250,
                "tags": [],
                "requires_research": False,
                "requires_citations": False,
                "requires_code": False,
            }
        ]

    plan["tasks"] = normalized_tasks
    return plan


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer
    merged_md: str
    final: str


# -----------------------------
# 2) LLM
# -----------------------------
TEXT_MODEL_PROVIDER = os.getenv("TEXT_MODEL_PROVIDER", "auto").strip().lower()
IMAGE_MODEL_PROVIDER = os.getenv("IMAGE_MODEL_PROVIDER", "auto").strip().lower()

# Bedrock / Eden
AWS_BEARER_TOKEN_BEDROCK = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "").strip()
BEDROCK_TEXT_MODEL_ID = os.getenv("BEDROCK_TEXT_MODEL_ID", "anthropic.claude-opus-4-7").strip()
AWS_REGION_BEDROCK = os.getenv("AWS_REGION_BEDROCK", os.getenv("AWS_REGION", "")).strip()

EDEN_API_KEY = os.getenv("EDEN_API_KEY", "").strip()
EDEN_IMAGE_ENDPOINT = os.getenv("EDEN_IMAGE_ENDPOINT", "").strip()  # optional if Eden SDK is used

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")

OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "AI Blog Agent")
OWL_ALPHA_MODEL = os.getenv("OWL_ALPHA_MODEL", "openrouter/owl-alpha")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")).strip()
GROK_MODEL = os.getenv("GROK_MODEL", os.getenv("XAI_MODEL", "grok-4.3"))
GROK_BASE_URL = os.getenv("GROK_BASE_URL", os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")).rstrip("/")
GROK_IMAGE_MODEL = os.getenv("GROK_IMAGE_MODEL", os.getenv("XAI_GROK_IMAGE_MODEL", "x-ai/grok-imagine-image-quality"))
XAI_IMAGE_MODEL = os.getenv("XAI_IMAGE_MODEL", "grok-imagine-image-quality")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj_start = text.find("{")
        obj_end = text.rfind("}")
        if obj_start != -1 and obj_end != -1:
            try:
                return json.loads(text[obj_start : obj_end + 1])
            except json.JSONDecodeError:
                pass
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        if arr_start != -1 and arr_end != -1:
            try:
                return json.loads(text[arr_start : arr_end + 1])
            except json.JSONDecodeError:
                pass
        raise


def _gemini_text_model_candidates() -> List[str]:
    configured_model = GEMINI_MODEL
    if configured_model in {"gemini-1.5-pro", "gemini-1.5-flash"}:
        configured_model = "gemini-2.5-flash"

    candidates = [
        configured_model,
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash-lite",
    ]
    deduped: List[str] = []
    for model in candidates:
        model = (model or "").strip()
        if model and model not in deduped:
            deduped.append(model)
    return deduped


def _extract_gemini_text(data: dict) -> str:
    texts: List[str] = []
    for candidate in data.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if text:
                texts.append(text)
    if texts:
        return "\n".join(texts).strip()
    raise RuntimeError(f"Gemini response did not include text: {data}")


def _gemini_error_message(model: str, status_code: int, response_text: str) -> str:
    try:
        data = json.loads(response_text)
        error = data.get("error") or {}
        message = error.get("message") or response_text
        if status_code == 429:
            return (
                f"{model}: quota/rate limit exceeded (HTTP 429). "
                f"{message.splitlines()[0]}"
            )
        return f"{model}: HTTP {status_code} from Gemini API. {message}"
    except json.JSONDecodeError:
        return f"{model}: HTTP {status_code} from Gemini API. {response_text}"


def _gemini_text_rest(prompt: str, model: str, api_key: str) -> str:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Gemini REST fallback requires the requests package. Install it with pip install requests."
        ) from exc

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7},
    }
    headers = {
        "Content-Type": "application/json",
        "Connection": "close",
        "User-Agent": "Mozilla/5.0",
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            response = requests.post(
                url,
                params={"key": api_key},
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            return _extract_gemini_text(response.json())
        except requests.HTTPError as exc:
            status_code = response.status_code
            if status_code < 500:
                raise RuntimeError(_gemini_error_message(model, status_code, response.text)) from exc
            last_error = exc
        except requests.RequestException as exc:
            last_error = exc

        if attempt < 4:
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"{model}: Gemini API connection failed after retries. {last_error}")


def _gemini_text_openrouter(prompt: str, model: str, api_key: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenRouter support requires the openai package. Install it with pip install openai."
        ) from exc

    if "/" not in model:
        model = f"google/{model}"

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=_openrouter_extra_headers(),
        timeout=90,
        max_retries=2,
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1200,
            )
            text = response.choices[0].message.content
            if not text:
                raise RuntimeError(f"OpenRouter Gemini response did not include text: {response}")
            return text.strip()
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code and status_code not in {408, 409, 429} and status_code < 500:
                raise
            last_error = exc
            if attempt < 4:
                time.sleep(1.5 * attempt)

    raise RuntimeError(f"OpenRouter Gemini request failed after retries: {last_error}")


def _gemini_text(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("Google_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY or Google_API_KEY is not set. Set it in the environment or Streamlit secrets before running the app."
        )

    errors: List[str] = []
    for model in _gemini_text_model_candidates():
        try:
            if api_key.startswith("sk-or-"):
                return _gemini_text_openrouter(prompt, model, api_key)
            else:
                return _gemini_text_rest(prompt, model, api_key)
        except Exception as exc:
            errors.append(str(exc))

    raise RuntimeError("Gemini API request failed. " + " | ".join(errors))


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _looks_like_openrouter_key(api_key: str) -> bool:
    return api_key.startswith("sk-or-")


def _openrouter_extra_headers() -> dict:
    headers = {"X-Title": OPENROUTER_APP_TITLE}
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    return headers


def _openrouter_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **_openrouter_extra_headers(),
    }


def _owl_alpha_api_key() -> Optional[str]:
    # Backward compatible keys + new mapping request.
    # - NVIDIA_API_KEY is treated as the Owl Alpha/OpenRouter text key.
    return _env_first(
        "NVIDIA_API_KEY",
        "OWL_ALPHA_API_KEY",
        "OPENROUTER_API_KEY",
        "Owl Alpha_API_KEY",
    )



def _owl_alpha_text(prompt: str) -> str:
    api_key = _owl_alpha_api_key()
    if not api_key:
        raise RuntimeError("OWL_ALPHA_API_KEY or OPENROUTER_API_KEY is not set for Owl Alpha text generation.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Owl Alpha/OpenRouter support requires the openai package. Install it with pip install openai."
        ) from exc

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=_openrouter_extra_headers(),
        timeout=90,
        max_retries=2,
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            response = client.chat.completions.create(
                model=OWL_ALPHA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1200,
            )
            text = response.choices[0].message.content
            if not text:
                raise RuntimeError(f"Owl Alpha/OpenRouter response did not include text: {response}")
            return text.strip()
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code and status_code not in {408, 409, 429} and status_code < 500:
                raise
            last_error = exc
            if attempt < 4:
                time.sleep(1.5 * attempt)

    raise RuntimeError(f"Owl Alpha/OpenRouter request failed after retries: {last_error}")


def _grok_api_key() -> Optional[str]:
    return os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")


def _groq_api_key() -> Optional[str]:
    return os.getenv("GROQ_API_KEY")


def _extract_openai_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    texts: List[str] = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                texts.append(str(text))
    if texts:
        return "\n".join(texts).strip()

    raise RuntimeError(f"Model response did not include text: {response}")


def _grok_text(prompt: str) -> str:
    api_key = _grok_api_key()
    if not api_key:
        raise RuntimeError("GROK_API_KEY or XAI_API_KEY is not set for Grok text generation.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Grok support requires the openai package. Install it with pip install openai."
        ) from exc

    client = OpenAI(api_key=api_key, base_url=GROK_BASE_URL)
    errors: List[str] = []

    try:
        response = client.responses.create(
            model=GROK_MODEL,
            input=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_output_tokens=1000,
            store=False,
        )
        return _extract_openai_response_text(response)
    except Exception as exc:
        errors.append(f"responses API: {exc}")

    try:
        response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError(f"Model response did not include text: {response}")
        return text.strip()
    except Exception as exc:
        errors.append(f"chat completions API: {exc}")

    raise RuntimeError("Grok API request failed. " + " | ".join(errors))


def _groq_text(prompt: str) -> str:
    api_key = _groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set for Groq text generation.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Groq support requires the openai package. Install it with pip install openai."
        ) from exc

    client = OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        timeout=90,
        max_retries=2,
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            response = client.chat.completions.create(
                model=GROQ_TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1200,
            )
            text = response.choices[0].message.content
            if not text:
                raise RuntimeError(f"Groq response did not include text: {response}")
            return text.strip()
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code and status_code not in {408, 409, 429} and status_code < 500:
                raise
            last_error = exc
            if attempt < 4:
                time.sleep(1.5 * attempt)

    raise RuntimeError(f"Groq request failed after retries: {last_error}")


def _openai_text(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set for OpenAI text generation.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI support requires the openai package. Install it with pip install openai."
        ) from exc

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
    )
    text = response.choices[0].message.content
    return text.strip()


def _bedrock_text(prompt: str) -> str:
    """Invoke AWS Bedrock text using AWS_BEARER_TOKEN_BEDROCK."""
    if not AWS_BEARER_TOKEN_BEDROCK:
        raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is not set for Bedrock text generation.")
    if not BEDROCK_TEXT_MODEL_ID:
        raise RuntimeError("BEDROCK_TEXT_MODEL_ID is not set for Bedrock text generation.")
    if not AWS_REGION_BEDROCK:
        raise RuntimeError("AWS_REGION_BEDROCK is not set for Bedrock text generation.")

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Bedrock support requires boto3. Install it with pip install boto3.") from exc

    runtime = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION_BEDROCK,
        aws_access_key_id="dummy",
        aws_secret_access_key="dummy",
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1200,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
    }

    # boto3 bedrock-runtime expects auth via the client configuration headers.
    # Do not pass extra fields that can be interpreted as part of the request body.
    response = runtime.invoke_model(
        modelId=BEDROCK_TEXT_MODEL_ID,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )


    raw = response.get("body")
    if raw is None:
        raise RuntimeError(f"Bedrock response missing body: {response}")

    payload_bytes = raw.read() if hasattr(raw, "read") else raw
    payload_text = payload_bytes.decode("utf-8") if isinstance(payload_bytes, (bytes, bytearray)) else str(payload_bytes)
    data = json.loads(payload_text)

    content = data.get("content")
    if isinstance(content, list) and content:
        item0 = content[0]
        if isinstance(item0, dict) and item0.get("text"):
            return str(item0["text"]).strip()

    if isinstance(data.get("output"), dict) and data["output"].get("text"):
        return str(data["output"]["text"]).strip()

    if data.get("completion"):
        return str(data["completion"]).strip()

    raise RuntimeError(f"Could not extract text from Bedrock response: {data}")


def _local_text(prompt: str) -> str:
    if not LOCAL_MODEL_NAME:
        raise RuntimeError("LOCAL_MODEL_NAME is not configured for local text generation.")

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Local model support requires the transformers package. Install it or use OPENAI_API_KEY instead."
        ) from exc

    text_pipe = pipeline("text-generation", model=LOCAL_MODEL_NAME, device=-1)
    result = text_pipe(prompt, max_length=512, do_sample=False, num_return_sequences=1)
    if not result or not isinstance(result, list):
        raise RuntimeError("Local text generation did not return a valid result.")
    return result[0].get("generated_text", "").strip()



def _get_text_provider_order() -> List[str]:
    # Check explicit provider overrides first
    if TEXT_MODEL_PROVIDER in ("bedrock", "aws_bedrock"):
        return ["bedrock"]
    if TEXT_MODEL_PROVIDER in ("owl", "owl-alpha", "owl_alpha", "openrouter", "openrouter_owl"):
        return ["owl_alpha"]
    if TEXT_MODEL_PROVIDER in ("groq", "groqcloud"):
        return ["groq"]
    if TEXT_MODEL_PROVIDER in ("grok", "xai"):
        return ["grok"]
    if TEXT_MODEL_PROVIDER in ("gemini", "openai", "local"):
        return [TEXT_MODEL_PROVIDER]

    # Auto-mode or default ordering
    order: List[str] = []
    if _groq_api_key():
        order.append("groq")
    if _owl_alpha_api_key():
        order.append("owl_alpha")
    if os.getenv("GOOGLE_API_KEY") or os.getenv("Google_API_KEY"):
        order.append("gemini")
    if AWS_BEARER_TOKEN_BEDROCK:
        order.append("bedrock")
    if _grok_api_key():
        order.append("grok")
    if os.getenv("OPENAI_API_KEY"):
        order.append("openai")
    if LOCAL_MODEL_NAME:
        order.append("local")
    return order or ["owl_alpha"]


def run_text_model(prompt: str) -> str:
    errors: List[str] = []
    for provider in _get_text_provider_order():
        try:
            if provider == "owl_alpha":
                return _owl_alpha_text(prompt)
            if provider == "bedrock":
                return _bedrock_text(prompt)
            if provider == "grok":
                return _grok_text(prompt)
            if provider == "groq":
                return _groq_text(prompt)
            if provider == "gemini":
                return _gemini_text(prompt)
            if provider == "openai":
                return _openai_text(prompt)
            if provider == "local":
                return _local_text(prompt)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
            logger.warning("Provider %s failed during text generation: %s", provider, exc)

    logger.error("All text providers failed. Errors: %s", errors)
    raise RuntimeError(
        "All text providers failed.\n" + "\n".join(errors)
    )


def _is_groq_text_mode() -> bool:
    return _get_text_provider_order()[0] == "groq"

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    prompt = (
        f"{ROUTER_SYSTEM}\n\n"
        f"Topic: {state['topic']}\n"
        f"As-of date: {state['as_of']}\n"
        "Return only valid JSON with keys: needs_research, mode, reason, queries, max_results_per_query."
    )
    raw = run_text_model(prompt)
    decision_data = _parse_json(raw)
    decision = RouterDecision(**decision_data)

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception as exc:
        logger.exception("Tavily search failed for query: '%s'", query)
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _dedupe_evidence(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    dedup = {}
    for item in evidence:
        if item.url:
            dedup[item.url] = item
    return list(dedup.values())


def _raw_results_to_evidence(raw: List[dict]) -> List[EvidenceItem]:
    evidence: List[EvidenceItem] = []
    for result in raw:
        url = (result.get("url") or "").strip()
        if not url:
            continue

        published_at = result.get("published_at") or result.get("published_date")
        published_date = _iso_to_date(published_at)
        evidence.append(
            EvidenceItem(
                title=(result.get("title") or url).strip(),
                url=url,
                published_at=published_date.isoformat() if published_date else None,
                snippet=(result.get("snippet") or result.get("content") or "")[:500],
                source=result.get("source"),
            )
        )
    return _dedupe_evidence(evidence)


def _compact_raw_results_for_prompt(raw: List[dict], limit: int, snippet_chars: int) -> List[dict]:
    compact: List[dict] = []
    for result in raw[:limit]:
        compact.append(
            {
                "title": (result.get("title") or "").strip(),
                "url": (result.get("url") or "").strip(),
                "published_at": result.get("published_at") or result.get("published_date"),
                "source": result.get("source"),
                "snippet": (result.get("snippet") or result.get("content") or "")[:snippet_chars],
            }
        )
    return compact


def _compact_evidence_for_prompt(evidence: List[EvidenceItem], limit: int, snippet_chars: int) -> List[dict]:
    return [
        {
            "title": item.title,
            "url": item.url,
            "published_at": item.published_at,
            "source": item.source,
            "snippet": (item.snippet or "")[:snippet_chars],
        }
        for item in evidence[:limit]
    ]


def _filter_evidence_for_mode(state: State, evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    if state.get("mode") != "open_book":
        return evidence

    as_of = date.fromisoformat(state["as_of"])
    cutoff = as_of - timedelta(days=int(state["recency_days"]))
    return [item for item in evidence if (d := _iso_to_date(item.published_at)) and d >= cutoff]

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    groq_mode = _is_groq_text_mode()
    max_queries = _env_positive_int("GROQ_RESEARCH_MAX_QUERIES", 4) if groq_mode else 10
    max_results = _env_positive_int("GROQ_RESEARCH_MAX_RESULTS", 3) if groq_mode else 6
    queries = (state.get("queries") or [])[:max_queries]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=max_results))

    if not raw:
        return {"evidence": []}

    fallback_evidence = _raw_results_to_evidence(raw)
    prompt_raw = (
        _compact_raw_results_for_prompt(raw, max_queries * max_results, 220)
        if groq_mode
        else raw
    )
    prompt = (
        f"{RESEARCH_SYSTEM}\n\n"
        f"As-of date: {state['as_of']}\n"
        f"Recency days: {state['recency_days']}\n\n"
        f"Raw results:\n{prompt_raw}\n"
        "Return only valid JSON with a top-level key named evidence, where evidence is a list of objects with title, url, published_at, snippet, and source."
    )
    try:
        raw_response = run_text_model(prompt)
        pack = EvidencePack(**_parse_json(raw_response))
        evidence = _dedupe_evidence(pack.evidence)
    except Exception:
        # Quota/network failures should not kill the whole run just because
        # the search results could not be LLM-cleaned.
        evidence = fallback_evidence

    return {"evidence": _filter_evidence_for_mode(state, evidence)}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None
    groq_mode = _is_groq_text_mode()
    evidence_for_prompt = _compact_evidence_for_prompt(
        evidence,
        limit=_env_positive_int("GROQ_PLAN_EVIDENCE_ITEMS", 8) if groq_mode else 16,
        snippet_chars=220 if groq_mode else 500,
    )

    prompt = (
        f"{ORCH_SYSTEM}\n\n"
        f"Topic: {state['topic']}\n"
        f"Mode: {mode}\n"
        f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
        f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
        f"Evidence:\n{evidence_for_prompt}\n"
        "Return only valid JSON matching the Plan schema."
    )
    raw_response = run_text_model(prompt)
    parsed = _parse_json(raw_response)
    normalized_plan = _normalize_plan_dict(parsed, state['topic'], mode)
    try:
        plan = Plan(**normalized_plan)
    except ValidationError:
        plan = Plan(**_normalize_plan_dict({}, state['topic'], mode))

    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.

Diagrams:
- If this section explains a workflow, architecture, roadmap, comparison, or step-by-step process, include one concise Mermaid diagram block.
- Use only simple Mermaid syntax that renders reliably: graph TD, graph LR, flowchart TD, or sequenceDiagram.
- Output Mermaid directly as a fenced code block:
  ```mermaid
  graph TD
      A[Start] --> B(Process)
  ```
- Do not describe a diagram as a prompt. Actually write the Mermaid code block.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    raw_plan = payload["plan"]
    if isinstance(raw_plan, dict):
        raw_plan = _normalize_plan_dict(raw_plan, payload.get("topic", ""), payload.get("mode", ""))
    plan = Plan(**raw_plan)
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_limit = _env_positive_int("GROQ_WORKER_EVIDENCE_ITEMS", 8) if _is_groq_text_mode() else 20
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:evidence_limit]
    )
    diagram_hint = (
        "This is the opening section. Include one compact Mermaid roadmap diagram for the blog."
        if task.id == 1
        else "Include a Mermaid diagram only if it helps this section explain a flow, architecture, or process."
    )

    prompt = (
        f"{WORKER_SYSTEM}\n\n"
        f"Blog title: {plan.blog_title}\n"
        f"Audience: {plan.audience}\n"
        f"Tone: {plan.tone}\n"
        f"Blog kind: {plan.blog_kind}\n"
        f"Constraints: {plan.constraints}\n"
        f"Topic: {payload['topic']}\n"
        f"Mode: {payload.get('mode')}\n"
        f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
        f"Section title: {task.title}\n"
        f"Goal: {task.goal}\n"
        f"Target words: {task.target_words}\n"
        f"Tags: {task.tags}\n"
        f"requires_research: {task.requires_research}\n"
        f"requires_citations: {task.requires_citations}\n"
        f"requires_code: {task.requires_code}\n"
        f"Diagram guidance: {diagram_hint}\n"
        f"Bullets:{bullets_text}\n\n"
        f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
    )
    section_md = run_text_model(prompt).strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) Reducer
#    merge_content and save markdown output
# ============================================================
def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def _mermaid_image_url(code: str) -> str:
    encoded = base64.urlsafe_b64encode(code.encode("utf-8")).decode("ascii")
    return f"https://mermaid.ink/svg/{encoded}"


def _mermaid_image_markdown(code: str, title: str = "Mermaid Diagram") -> str:
    return f"![{title}]({_mermaid_image_url(code)})\n\n*(Generated diagram for the blog structure/flow)*"


def _convert_mermaid_to_images(md_content: str) -> tuple[str, int]:
    # Match fenced Mermaid blocks with Windows or Unix line endings.
    pattern = r"```\s*mermaid\s*\r?\n(.*?)\r?\n```"
    converted_count = 0

    def replacer(match):
        nonlocal converted_count
        code = match.group(1).strip()
        try:
            converted_count += 1
            return _mermaid_image_markdown(code)
        except Exception as exc:
            logger.warning("Failed to encode Mermaid code block: %s", exc)
            return match.group(0)

    converted_md = re.sub(pattern, replacer, md_content, flags=re.DOTALL | re.IGNORECASE)
    return converted_md, converted_count


def _mermaid_label(text: str, max_chars: int = 38) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.replace("\\", "").replace('"', "'").replace("[", "(").replace("]", ")")
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip() + "..."
    return cleaned or "Section"


def _fallback_roadmap_diagram(plan: Plan) -> str:
    tasks = plan.tasks[:7]
    if not tasks:
        return ""

    lines = ["flowchart TD", f'    T["{_mermaid_label(plan.blog_title, 48)}"]']
    previous = "T"
    for index, task in enumerate(tasks, start=1):
        node_id = f"S{index}"
        lines.append(f'    {node_id}["{_mermaid_label(task.title)}"]')
        lines.append(f"    {previous} --> {node_id}")
        previous = node_id
    return "\n".join(lines)


def _insert_fallback_diagram(md_content: str, plan: Plan) -> str:
    if "mermaid.ink/" in md_content or "```mermaid" in md_content.lower():
        return md_content

    diagram_code = _fallback_roadmap_diagram(plan)
    if not diagram_code:
        return md_content

    diagram_md = f"## Blog Roadmap\n\n{_mermaid_image_markdown(diagram_code, 'Blog Roadmap Diagram')}\n\n"
    return re.sub(r"^(# .+?\r?\n\r?\n)", rf"\1{diagram_md}", md_content, count=1, flags=re.DOTALL)


def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    # Convert Mermaid code blocks to inline images. If the model skipped diagrams,
    # insert a deterministic roadmap so every generated blog still has a visual.
    merged_md, converted_diagrams = _convert_mermaid_to_images(merged_md)
    if converted_diagrams == 0:
        merged_md = _insert_fallback_diagram(merged_md, plan)

    blog_slug = _safe_slug(plan.blog_title)
    
    # 1. Export local markdown file (fallback/export)
    filename = f"{blog_slug}.md"
    try:
        Path(filename).write_text(merged_md, encoding="utf-8")
        logger.info("Saved local markdown export: %s", filename)
    except Exception as exc:
        logger.error("Failed to write local markdown export: %s", exc)

    # 2. Save to SQLite database
    try:
        evidence_list = [e.model_dump() for e in state.get("evidence", [])]
        plan_dict = plan.model_dump()
        database.save_blog(blog_slug, plan.blog_title, merged_md, evidence_list, plan_dict)
        logger.info("Saved blog to SQLite database: slug=%s", blog_slug)
    except Exception as exc:
        logger.exception("Failed to save blog to SQLite database")

    return {"merged_md": merged_md, "final": merged_md}


# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", merge_content)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
