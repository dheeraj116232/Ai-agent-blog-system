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


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

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

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
TEXT_MODEL_PROVIDER = os.getenv("TEXT_MODEL_PROVIDER", "auto").strip().lower()
IMAGE_MODEL_PROVIDER = os.getenv("IMAGE_MODEL_PROVIDER", "auto").strip().lower()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "AI Blog Agent")
OWL_ALPHA_MODEL = os.getenv("OWL_ALPHA_MODEL", "openrouter/owl-alpha")
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


def _gemini_text(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Set it in the environment or Streamlit secrets before running the app."
        )

    errors: List[str] = []
    for model in _gemini_text_model_candidates():
        try:
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
    return _env_first("OWL_ALPHA_API_KEY", "OPENROUTER_API_KEY", "Owl Alpha_API_KEY")


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
    if TEXT_MODEL_PROVIDER in ("owl", "owl-alpha", "owl_alpha", "openrouter", "openrouter_owl"):
        return ["owl_alpha"]
    if TEXT_MODEL_PROVIDER in ("grok", "xai"):
        return ["grok"]
    if TEXT_MODEL_PROVIDER in ("gemini", "openai", "local"):
        return [TEXT_MODEL_PROVIDER]

    order: List[str] = []
    if _owl_alpha_api_key():
        order.append("owl_alpha")
    if _grok_api_key():
        order.append("grok")
    if os.getenv("GOOGLE_API_KEY"):
        order.append("gemini")
    if os.getenv("OPENAI_API_KEY"):
        order.append("openai")
    if LOCAL_MODEL_NAME:
        order.append("local")
    return order or ["gemini"]


def run_text_model(prompt: str) -> str:
    errors: List[str] = []
    for provider in _get_text_provider_order():
        try:
            if provider == "owl_alpha":
                return _owl_alpha_text(prompt)
            if provider == "grok":
                return _grok_text(prompt)
            if provider == "gemini":
                return _gemini_text(prompt)
            if provider == "openai":
                return _openai_text(prompt)
            if provider == "local":
                return _local_text(prompt)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")

    raise RuntimeError(
        "All text providers failed.\n" + "\n".join(errors)
    )

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
    except Exception:
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
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    fallback_evidence = _raw_results_to_evidence(raw)
    prompt = (
        f"{RESEARCH_SYSTEM}\n\n"
        f"As-of date: {state['as_of']}\n"
        f"Recency days: {state['recency_days']}\n\n"
        f"Raw results:\n{raw}\n"
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

    prompt = (
        f"{ORCH_SYSTEM}\n\n"
        f"Topic: {state['topic']}\n"
        f"Mode: {mode}\n"
        f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
        f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
        f"Evidence:\n{[e.model_dump() for e in evidence][:16]}\n"
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
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    raw_plan = payload["plan"]
    if isinstance(raw_plan, dict):
        raw_plan = _normalize_plan_dict(raw_plan, payload.get("topic", ""), payload.get("mode", ""))
    plan = Plan(**raw_plan)
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
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
        f"Bullets:{bullets_text}\n\n"
        f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
    )
    section_md = run_text_model(prompt).strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
- Every image object must include placeholder, filename, alt, caption, prompt, size, and quality.
- Use size values only from: 1024x1024, 1024x1536, 1536x1024.
- Use quality values only from: low, medium, high.
Return strictly this JSON shape:
{"md_with_placeholders":"...","images":[{"placeholder":"[[IMAGE_1]]","filename":"diagram_name.png","alt":"Short alt text","caption":"Short caption","prompt":"Detailed image prompt","size":"1024x1024","quality":"medium"}]}
"""


def _normalize_image_plan_dict(raw: object, merged_md: str) -> dict:
    if not isinstance(raw, dict):
        return {"md_with_placeholders": merged_md, "images": []}

    md = raw.get("md_with_placeholders") or raw.get("markdown") or raw.get("content") or merged_md
    md = str(md)

    raw_images = raw.get("images") or []
    if not isinstance(raw_images, list):
        raw_images = []

    images: List[dict] = []
    for index, item in enumerate(raw_images[:3], start=1):
        if not isinstance(item, dict):
            continue

        prompt = str(item.get("prompt") or item.get("description") or item.get("title") or "").strip()
        if not prompt:
            continue

        title = str(
            item.get("title")
            or item.get("caption")
            or item.get("alt")
            or f"Image {index}"
        ).strip()
        placeholder = str(item.get("placeholder") or f"[[IMAGE_{index}]]").strip()
        if not re.fullmatch(r"\[\[IMAGE_[1-3]\]\]", placeholder):
            placeholder = f"[[IMAGE_{index}]]"

        filename = str(item.get("filename") or f"{_safe_slug(title)}.png").strip()
        if not re.search(r"\.(png|jpg|jpeg|webp)$", filename, flags=re.IGNORECASE):
            filename = f"{_safe_slug(filename)}.png"

        size = str(item.get("size") or "1024x1024")
        if size not in {"1024x1024", "1024x1536", "1536x1024"}:
            size = "1024x1024"

        quality = str(item.get("quality") or "medium").lower()
        if quality not in {"low", "medium", "high"}:
            quality = "medium"

        if placeholder not in md:
            md = f"{md.rstrip()}\n\n{placeholder}\n"

        images.append(
            {
                "placeholder": placeholder,
                "filename": filename,
                "alt": str(item.get("alt") or title).strip(),
                "caption": str(item.get("caption") or title).strip(),
                "prompt": prompt,
                "size": size,
                "quality": quality,
            }
        )

    return {"md_with_placeholders": md, "images": images}


def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    prompt = (
        f"{DECIDE_IMAGES_SYSTEM}\n\n"
        f"Blog kind: {plan.blog_kind}\n"
        f"Topic: {state['topic']}\n\n"
        "Insert placeholders + propose image prompts.\n\n"
        f"{merged_md}"
        "\nReturn only valid JSON matching the GlobalImagePlan schema."
    )
    image_plan_data = _normalize_image_plan_dict(_parse_json(run_text_model(prompt)), merged_md)
    image_plan = GlobalImagePlan(**image_plan_data)

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _grok_generate_image_bytes(prompt: str) -> bytes:
    api_key = _grok_api_key()
    if not api_key:
        raise RuntimeError("GROK_API_KEY or XAI_API_KEY is not set for Grok image generation.")

    try:
        from openai import OpenAI
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Grok image generation requires the openai and requests packages."
        ) from exc

    client = OpenAI(api_key=api_key, base_url=GROK_BASE_URL)
    image_response = client.images.generate(
        model=XAI_IMAGE_MODEL,
        prompt=prompt,
    )

    data = getattr(image_response, "data", None)
    if not data:
        raise RuntimeError("Grok image generation returned no data.")

    image_data = data[0]
    if isinstance(image_data, dict):
        b64_json = image_data.get("b64_json")
        image_url = image_data.get("url")
    else:
        b64_json = getattr(image_data, "b64_json", None)
        image_url = getattr(image_data, "url", None)

    if b64_json:
        return base64.b64decode(b64_json)

    if image_url:
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        return response.content

    raise RuntimeError("Grok image generation returned neither base64 data nor a URL.")


def _openrouter_grok_image_api_key() -> Optional[str]:
    explicit_key = _env_first("XAI_GROK_API_KEY", "xAI_GROK_API_KEY", "GROK_IMAGE_API_KEY")
    if explicit_key:
        return explicit_key

    grok_key = _grok_api_key()
    if grok_key and _looks_like_openrouter_key(grok_key):
        return grok_key
    return None


def _decode_image_url_to_bytes(url: str) -> bytes:
    if url.startswith("data:image/") and "," in url:
        _, encoded = url.split(",", 1)
        return base64.b64decode(encoded)

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Downloading remote image URLs requires the requests package."
        ) from exc

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _extract_openrouter_image_bytes(data: dict) -> bytes:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter image response did not include choices: {data}")

    message = choices[0].get("message") or {}
    images = message.get("images") or []
    for image in images:
        if not isinstance(image, dict):
            continue
        image_url = image.get("image_url") or {}
        if isinstance(image_url, dict) and image_url.get("url"):
            return _decode_image_url_to_bytes(str(image_url["url"]))

    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            image_url = part.get("image_url") or {}
            if isinstance(image_url, dict) and image_url.get("url"):
                return _decode_image_url_to_bytes(str(image_url["url"]))

    raise RuntimeError(f"OpenRouter image response did not include generated image data: {data}")


def _openrouter_grok_generate_image_bytes(prompt: str) -> bytes:
    api_key = _openrouter_grok_image_api_key()
    if not api_key:
        raise RuntimeError("XAI_GROK_API_KEY or GROK_IMAGE_API_KEY is not set for OpenRouter Grok image generation.")

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "OpenRouter Grok image generation requires the requests package."
        ) from exc

    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    errors: List[str] = []
    for modalities in (["image"], ["image", "text"]):
        payload = {
            "model": GROK_IMAGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": modalities,
            "stream": False,
        }
        try:
            response = requests.post(
                url,
                headers=_openrouter_headers(api_key),
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            return _extract_openrouter_image_bytes(response.json())
        except requests.HTTPError as exc:
            response_text = getattr(exc.response, "text", "") if exc.response is not None else ""
            errors.append(f"modalities={modalities}: HTTP {response.status_code} from OpenRouter image API. {response_text[:500]}")
        except Exception as exc:
            errors.append(f"modalities={modalities}: {exc}")

    raise RuntimeError("OpenRouter Grok image generation failed. " + " | ".join(errors))


def _get_image_provider_order() -> List[str]:
    if IMAGE_MODEL_PROVIDER in ("grok", "grok_openrouter", "openrouter_grok", "xai_grok"):
        return ["grok_openrouter"]
    if IMAGE_MODEL_PROVIDER in ("xai", "xai_official"):
        return ["xai"]
    if IMAGE_MODEL_PROVIDER in ("gemini", "openai"):
        return [IMAGE_MODEL_PROVIDER]

    order: List[str] = []
    if _openrouter_grok_image_api_key():
        order.append("grok_openrouter")
    if _grok_api_key() and not _looks_like_openrouter_key(_grok_api_key() or ""):
        order.append("xai")
    if os.getenv("GOOGLE_API_KEY"):
        order.append("gemini")
    if os.getenv("OPENAI_API_KEY"):
        order.append("openai")
    return order


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _openai_generate_image_bytes(prompt: str) -> bytes:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set for OpenAI image generation.")

    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI image fallback requires the openai package. Install it with pip install openai."
        ) from exc

    openai.api_key = api_key
    image_response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024",
    )
    if not image_response or not image_response.data:
        raise RuntimeError("OpenAI image generation returned no data.")

    b64_json = image_response.data[0].b64_json
    if not b64_json:
        raise RuntimeError("OpenAI image generation returned empty image data.")

    return base64.b64decode(b64_json)


def _generate_image_bytes(prompt: str) -> bytes:
    errors: List[str] = []

    for provider in _get_image_provider_order():
        try:
            if provider == "grok_openrouter":
                return _openrouter_grok_generate_image_bytes(prompt)
            if provider == "xai":
                return _grok_generate_image_bytes(prompt)
            if provider == "gemini":
                return _gemini_generate_image_bytes(prompt)
            if provider == "openai":
                return _openai_generate_image_bytes(prompt)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")

    if not errors:
        errors.append("no image provider key is configured")

    raise RuntimeError("Image generation failed. " + " | ".join(errors))


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    blog_slug = _safe_slug(plan.blog_title)
    images_dir = Path("images") / blog_slug
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = Path(spec["filename"]).name
        out_path = images_dir / filename

        try:
            img_bytes = _generate_image_bytes(spec["prompt"])
            out_path.write_bytes(img_bytes)
        except Exception as e:
            # graceful fallback: keep doc usable
            prompt_block = (
                f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                f"> **Alt:** {spec.get('alt','')}\n>\n"
                f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                f"> **Error:** {e}\n"
            )
            md = md.replace(placeholder, prompt_block)
            continue

        img_md = f"![{spec['alt']}](images/{blog_slug}/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{blog_slug}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
