from __future__ import annotations

import json
import os
import re
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import database


# -----------------------------
# Import your compiled LangGraph app lazily (avoid import-time side effects)
# -----------------------------
def get_graph_app():
    """Lazily import and return the compiled LangGraph app.

    Importing the compiled backend at module import time can trigger
    heavy work (or runtime calls) before Streamlit has initialized
    session internals. Import on first use instead.
    """
    from bwa_backend import app as _app

    return _app


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def _load_font(size: int, bold: bool = False, mono: bool = False):
    candidates = []
    if mono:
        candidates.extend(["consola.ttf", "DejaVuSansMono.ttf", "C:/Windows/Fonts/consola.ttf"])
    elif bold:
        candidates.extend(["arialbd.ttf", "DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"])
    else:
        candidates.extend(["arial.ttf", "DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"])

    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _clean_markdown_text(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"[*_`]+", "", text)
    return text.strip()


def _wrap_text_px(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word

        while draw.textlength(current, font=font) > max_width and len(current) > 1:
            split_at = max(1, len(current) - 1)
            while split_at > 1 and draw.textlength(current[:split_at], font=font) > max_width:
                split_at -= 1
            lines.append(current[:split_at])
            current = current[split_at:]

    if current:
        lines.append(current)
    return lines


def markdown_to_pdf_bytes(md: str) -> bytes:
    page_width, page_height = 1240, 1754  # A4-ish at 150 DPI
    margin = 90
    max_width = page_width - (margin * 2)
    bg = "white"
    text_color = "#111827"
    muted_color = "#4b5563"
    code_bg = "#f3f4f6"

    body_font = _load_font(28)
    bold_font = _load_font(30, bold=True)
    h1_font = _load_font(48, bold=True)
    h2_font = _load_font(38, bold=True)
    h3_font = _load_font(32, bold=True)
    code_font = _load_font(24, mono=True)

    pages: List[Image.Image] = []

    def new_page():
        page = Image.new("RGB", (page_width, page_height), bg)
        pages.append(page)
        return page, ImageDraw.Draw(page), margin

    page, draw, y = new_page()

    def ensure_space(required: int):
        nonlocal page, draw, y
        if y + required <= page_height - margin:
            return
        page, draw, y = new_page()

    def draw_wrapped(text: str, font, color=text_color, indent: int = 0, spacing: int = 10):
        nonlocal y
        text = _clean_markdown_text(text)
        if not text:
            y += 18
            return
        lines = _wrap_text_px(draw, text, font, max_width - indent)
        line_height = int(font.size * 1.35) if hasattr(font, "size") else 36
        ensure_space((line_height * len(lines)) + spacing)
        for line in lines:
            draw.text((margin + indent, y), line, font=font, fill=color)
            y += line_height
        y += spacing

    in_code = False
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            y += 12
            i += 1
            continue

        if not stripped:
            y += 18
            i += 1
            continue

        if in_code:
            ensure_space(42)
            draw.rectangle((margin - 12, y - 6, page_width - margin + 12, y + 36), fill=code_bg)
            draw.text((margin, y), line[:110], font=code_font, fill=text_color)
            y += 42
        elif stripped.startswith("![") and stripped.endswith(")"):
            match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if match:
                alt = match.group(1)
                url = match.group(2)
                try:
                    import requests
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        img_data = Image.open(BytesIO(resp.content))
                        w, h = img_data.size
                        scale = min(max_width / w, 450 / h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        try:
                            resample_filter = Image.Resampling.LANCZOS
                        except AttributeError:
                            resample_filter = Image.ANTIALIAS
                        img_scaled = img_data.resize((new_w, new_h), resample_filter)
                        ensure_space(new_h + 20)
                        x_offset = margin + int((max_width - new_w) / 2)
                        page.paste(img_scaled, (x_offset, y))
                        y += new_h + 20
                except Exception:
                    draw_wrapped(f"[Image: {alt or 'Diagram'} (could not load)]", body_font, color=muted_color)
            else:
                draw_wrapped(stripped, body_font)
        elif stripped.startswith("# "):
            draw_wrapped(stripped[2:], h1_font, spacing=28)
        elif stripped.startswith("## "):
            draw_wrapped(stripped[3:], h2_font, spacing=22)
        elif stripped.startswith("### "):
            draw_wrapped(stripped[4:], h3_font, spacing=18)
        elif stripped.startswith(("- ", "* ")):
            draw_wrapped("• " + stripped[2:], body_font, indent=24)
        elif re.match(r"^\d+\.\s+", stripped):
            draw_wrapped(stripped, body_font, indent=24)
        else:
            draw_wrapped(stripped, body_font)
        i += 1

    buf = BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:], resolution=150)
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress.
    Yields ("updates"/"values"/"final"/"error", payload).
    """
    latest_state: Dict[str, Any] = {}
    try:
        provider = os.getenv("TEXT_MODEL_PROVIDER", "").strip().lower()
        config = {"max_concurrency": 1} if provider in {"groq", "groqcloud"} else None
        for step in graph_app.stream(inputs, config=config, stream_mode="values"):
            if isinstance(step, dict):
                latest_state = step
            yield ("values", step)
    except Exception as exc:
        yield ("error", exc)
        return

    yield ("final", latest_state)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# -----------------------------
# Past blogs helpers
# -----------------------------
def get_past_blogs() -> List[Dict[str, Any]]:
    return database.list_blogs()


def get_blog_by_id(blog_id: int) -> Optional[Dict[str, Any]]:
    return database.get_blog(blog_id)


def delete_blog_by_id(blog_id: int) -> Tuple[bool, str]:
    try:
        ok = database.delete_blog(blog_id)
        if ok:
            return True, "Blog deleted successfully from database."
        else:
            return False, "Blog not found in database."
    except Exception as exc:
        return False, f"Could not delete selected blog: {exc}"


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

def inject_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #f5f7fb;
                --panel: #ffffff;
                --ink: #18212f;
                --muted: #667085;
                --line: #dfe5ed;
                --accent: #176b63;
            }

            .stApp {
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.84), rgba(245, 247, 251, 0.96)),
                    var(--bg);
                color: var(--ink);
            }

            .block-container {
                max-width: 1280px;
                padding-top: 1.4rem;
                padding-bottom: 3rem;
            }

            [data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] > div:first-child {
                padding-top: 1.5rem;
            }

            h1, h2, h3 {
                color: var(--ink);
                letter-spacing: 0;
            }

            h1 {
                font-size: clamp(2rem, 4vw, 3.1rem);
                line-height: 1.05;
                margin-bottom: 0.35rem;
            }

            .app-header {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 1rem;
                padding: 0.35rem 0 1.15rem 0;
                border-bottom: 1px solid var(--line);
                margin-bottom: 1.1rem;
            }

            .app-kicker {
                color: var(--accent);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                margin: 0 0 0.2rem 0;
                text-transform: uppercase;
            }

            .app-subtitle {
                color: var(--muted);
                font-size: 1rem;
                margin: 0;
            }

            .status-chip {
                border: 1px solid rgba(23, 107, 99, 0.22);
                background: #eef8f6;
                color: var(--accent);
                border-radius: 999px;
                padding: 0.42rem 0.75rem;
                font-weight: 800;
                white-space: nowrap;
            }

            .stButton > button,
            .stDownloadButton > button {
                border-radius: 8px;
                border: 1px solid var(--line);
                min-height: 2.65rem;
                font-weight: 800;
                transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover {
                border-color: rgba(23, 107, 99, 0.55);
                box-shadow: 0 8px 20px rgba(22, 35, 54, 0.08);
                transform: translateY(-1px);
            }

            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #176b63, #2b6f4e);
                border: 0;
                color: #ffffff;
            }

            textarea,
            input,
            [data-baseweb="select"] > div {
                border-radius: 8px !important;
            }

            div[data-baseweb="tab-list"] {
                gap: 0.45rem;
                border-bottom: 1px solid var(--line);
            }

            button[data-baseweb="tab"] {
                background: rgba(255, 255, 255, 0.74);
                border: 1px solid var(--line);
                border-bottom: 0;
                border-radius: 8px 8px 0 0;
                color: var(--muted);
                padding: 0.55rem 0.95rem;
            }

            button[data-baseweb="tab"][aria-selected="true"] {
                background: #ffffff;
                border-top: 3px solid var(--accent);
                color: var(--accent);
                font-weight: 800;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: 8px;
                overflow: hidden;
            }

            [data-testid="stExpander"] {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 8px;
            }

            [data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.85rem 1rem;
            }

            .empty-state {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1.1rem 1.25rem;
                color: var(--muted);
            }

            .stAlert {
                border-radius: 8px;
            }

            @media (max-width: 760px) {
                .app-header {
                    align-items: flex-start;
                    flex-direction: column;
                }
                .status-chip {
                    white-space: normal;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

st.markdown(
    """
    <div class="app-header">
        <div>
            <p class="app-kicker">LangGraph workspace</p>
            <h1>Blog Writing Agent</h1>
            <p class="app-subtitle">Focused draft console</p>
        </div>
        <div class="status-chip">Local</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "topic_input" not in st.session_state:
    st.session_state["topic_input"] = ""

with st.sidebar:
    st.header("New Blog")
    topic = st.text_area(
        "Topic",
        value=st.session_state["topic_input"],
        height=120,
        placeholder="Enter a topic, angle, or question...",
    )
    st.session_state["topic_input"] = topic
    as_of = st.date_input("As-of date", value=date.today(), key="as_of_date")
    run_btn = st.button("Generate Blog", type="primary", use_container_width=True, key="generate_blog_btn")

    # ✅ NEW: Past blogs list (keeps everything else intact)
    st.divider()
    st.subheader("Past Blogs")

    past_blogs = get_past_blogs()
    if not past_blogs:
        st.caption("No saved blogs found in database.")
        selected_blog_id = None
    else:
        # Build labels from db records
        options: List[str] = []
        blog_by_label: Dict[str, int] = {}
        for b in past_blogs[:50]:
            label = f"{b['title']} (ID: {b['id']})"
            options.append(label)
            blog_by_label[label] = b['id']

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
            key="select_past_blog",
        )
        selected_blog_id = blog_by_label.get(selected_label)

        if st.button("Load Selected Blog", use_container_width=True, key="load_blog_btn"):
            if selected_blog_id is not None:
                blog = get_blog_by_id(selected_blog_id)
                if blog:
                    # Load into session_state, restoring plan and evidence metadata!
                    st.session_state["last_out"] = {
                        "plan": blog["plan"],
                        "evidence": blog["evidence"],
                        "final": blog["final_markdown"],
                        "sections": [],  # we can infer count from header regex
                    }
                    st.session_state["topic_input"] = blog["title"]
                    st.session_state["loaded_blog_id"] = blog["id"]
                    rerun_app()

        confirm_delete = st.checkbox("Confirm delete", key="confirm_delete_blog")
        if st.button("Delete Selected Blog", type="secondary", use_container_width=True, disabled=not confirm_delete, key="delete_blog_btn"):
            if selected_blog_id is not None:
                is_currently_loaded = st.session_state.get("loaded_blog_id") == selected_blog_id
                ok, message = delete_blog_by_id(selected_blog_id)
                if ok:
                    if is_currently_loaded:
                        st.session_state["last_out"] = None
                        st.session_state.pop("loaded_blog_id", None)
                    st.success(message)
                    rerun_app()
                else:
                    st.error(message)


# Keep your topic input as-is; optionally prefill for next run after loading a blog

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


def friendly_error_message(exc: Exception) -> str:
    msg = str(exc)
    lowered = msg.lower()
    if "owl alpha" in lowered or "owl_alpha" in lowered or "openrouter" in lowered:
        if "api key expired" in lowered or "api key not valid" in lowered or "invalid api key" in lowered or "401" in msg:
            return (
                "OpenRouter/Owl Alpha API key is invalid or unauthorized. Check OWL_ALPHA_API_KEY "
                "in .env, then restart Streamlit."
            )
        if "connection error" in lowered or "connection" in lowered or "timed out" in lowered:
            return (
                "OpenRouter/Owl Alpha connection failed while generating text. Your key/model are configured, "
                "so this is usually a temporary network/provider issue. Retry the generation in a moment."
            )
        if "429" in msg or "quota" in lowered or "rate limit" in lowered:
            return (
                "OpenRouter/Owl Alpha quota or rate limit was reached. Wait for quota reset "
                "or use another OpenRouter key."
            )
        if "402" in msg or "payment required" in lowered or "credits" in lowered:
            return (
                "OpenRouter says credits or billing are required for this request. Check the "
                "OpenRouter account tied to the configured key."
            )
    if "groq" in lowered:
        if "api key expired" in lowered or "api key not valid" in lowered or "invalid api key" in lowered or "401" in msg:
            return (
                "Groq API key is invalid or unauthorized. Check GROQ_API_KEY in .env, "
                "then restart Streamlit."
            )
        if "429" in msg or "quota" in lowered or "rate limit" in lowered:
            return (
                "Groq quota or rate limit was reached. Wait for quota reset, enable billing, "
                "or use another provider key."
            )
        if "connection error" in lowered or "connection" in lowered or "timed out" in lowered:
            return (
                "Groq connection failed while generating text. This is usually temporary; "
                "retry the generation in a moment."
            )
    if ("grok" in lowered or "xai" in lowered) and (
        "api key expired" in lowered or "api key not valid" in lowered or "invalid api key" in lowered
    ):
        return (
            "Grok/xAI API key is expired or invalid. The app is running, but Grok cannot "
            "generate text until you renew the key or add another valid provider key."
        )
    if ("grok" in lowered or "xai" in lowered) and (
        "429" in msg or "quota" in lowered or "rate limit" in lowered
    ):
        return (
            "Grok/xAI quota or rate limit was reached. The app is running, but Grok cannot "
            "generate more text right now. Wait for quota reset, enable billing, or add another provider key."
        )
    if "api key expired" in lowered or "api key not valid" in lowered or "invalid api key" in lowered:
        return (
            "Google API key is expired or invalid. The app is running, but Gemini cannot "
            "generate text until you renew the key or add a valid provider key."
        )
    if "429" in msg or "quota" in lowered or "rate limit" in lowered:
        return (
            "Gemini API quota/rate limit exceeded. The app is running, but the current "
            "Google API key cannot generate more text right now. Wait for quota reset, "
            "enable billing, or add another text provider key."
        )
    if "connectionreseterror" in lowered or "forcibly closed" in lowered:
        return (
            "Gemini connection was reset while generating text. Retry in a moment, or "
            "use another text provider if this keeps happening."
        )
    return f"Generation failed: {msg}"


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "final": "",
    }

    status = st.status("Running graph...", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(get_graph_app(), inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            summary = {
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                "evidence_count": len(current_state.get("evidence", []) or []),
                "tasks": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                "sections_done": len(current_state.get("sections", []) or []),
            }
            progress_area.json(summary)

            log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            status.update(label="Done", state="complete", expanded=False)
            log("[final] received final state")

        elif kind == "error":
            message = friendly_error_message(payload)
            status.update(label="Generation stopped", state="error", expanded=True)
            st.error(message)
            log(f"[error] {message}")

# Layout
tab_plan, tab_evidence, tab_preview, tab_logs = st.tabs(
    ["Plan", "Evidence", "Preview", "Logs"]
)

# Render last result (if any)
out = st.session_state.get("last_out")
if out:
    final_md_stats = out.get("final") or ""
    plan_stats = out.get("plan")
    if hasattr(plan_stats, "tasks"):
        task_count = len(plan_stats.tasks)
    elif isinstance(plan_stats, dict):
        task_count = len(plan_stats.get("tasks") or [])
    else:
        task_count = 0

    stats_cols = st.columns(4)
    word_count = len(re.findall(r'\b\w+\b', final_md_stats))
    stats_cols[0].metric("Words", f"{word_count:,}")
    
    sections_count = len(out.get("sections") or [])
    if sections_count == 0 and final_md_stats:
        sections_count = len(re.findall(r"^##\s+", final_md_stats, re.MULTILINE))
    stats_cols[1].metric("Sections", sections_count)
    
    stats_cols[2].metric("Tasks", task_count)
    stats_cols[3].metric("Evidence", len(out.get("evidence") or []))

    # --- Plan tab ---
    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Preview tab ---
    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            st.markdown(final_md, unsafe_allow_html=False)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                # fallback: parse from markdown title
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

            pdf = markdown_to_pdf_bytes(final_md)
            st.download_button(
                "Download PDF",
                data=pdf,
                file_name=f"{safe_slug(blog_title)}.pdf",
                mime="application/pdf",
            )

    # --- Logs tab ---
    with tab_logs:
        st.subheader("Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.markdown(
        """
        <div class="empty-state">
            Enter a topic in the sidebar and generate your first draft.
        </div>
        """,
        unsafe_allow_html=True,
    )
