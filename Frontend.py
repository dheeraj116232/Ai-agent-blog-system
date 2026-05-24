from __future__ import annotations

import json
import os
import re
import textwrap
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

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


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


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

    def draw_local_image(src: str):
        nonlocal y
        img_path = _resolve_image_path(src)
        if not img_path.exists():
            draw_wrapped(f"[Image not found: {src}]", body_font, muted_color)
            return
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                scale = min(max_width / img.width, 520 / img.height, 1)
                size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                ensure_space(size[1] + 30)
                img = img.resize(size)
                x = margin + ((max_width - size[0]) // 2)
                page.paste(img, (x, y))
                y += size[1] + 24
        except Exception as exc:
            draw_wrapped(f"[Image could not be rendered: {src} ({exc})]", body_font, muted_color)

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

        image_match = _MD_IMG_RE.fullmatch(stripped)
        if image_match:
            draw_local_image(image_match.group("src").strip())
            i += 1
            continue

        caption_match = _CAPTION_LINE_RE.match(stripped)
        if caption_match:
            draw_wrapped(caption_match.group("cap"), body_font, muted_color, spacing=18)
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
        for step in graph_app.stream(inputs, stream_mode="values"):
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
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def image_paths_from_markdown(md: str) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for match in _MD_IMG_RE.finditer(md):
        src = (match.group("src") or "").strip()
        if src.startswith("http://") or src.startswith("https://") or src.startswith("data:"):
            continue
        path = _resolve_image_path(src)
        key = str(path)
        if path.exists() and key not in seen:
            paths.append(path)
            seen.add(key)
    return paths


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def show_image(image: Any, caption: Optional[str] = None):
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError as exc:
        if "use_container_width" not in str(exc):
            raise
        st.image(image, caption=caption, use_column_width=True)


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            show_image(src, caption=caption or (alt or None))
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                show_image(str(img_path), caption=caption or (alt or None))
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# ✅ NEW: Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """
    Returns .md files in current working directory, newest first.
    Filters out obvious non-blog markdown files if needed.
    """
    cwd = Path(".")
    excluded = {"readme.md"}
    files = [p for p in cwd.glob("*.md") if p.is_file() and p.name.lower() not in excluded]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _is_safe_blog_path(p: Path) -> bool:
    try:
        cwd = Path(".").resolve()
        target = p.resolve()
        return target.parent == cwd and target.suffix.lower() == ".md" and target.name.lower() != "readme.md"
    except Exception:
        return False


def _remove_empty_image_parents(paths: List[Path]) -> None:
    images_root = (Path(".") / "images").resolve()
    for path in paths:
        parent = path.parent
        while parent != images_root and images_root in parent.parents:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def delete_blog_file(blog_file: Path) -> Tuple[bool, str]:
    if not _is_safe_blog_path(blog_file):
        return False, "Selected file is not a safe generated blog file."

    try:
        md_text = read_md_file(blog_file)
    except FileNotFoundError:
        return False, "Selected blog file no longer exists."
    except Exception as exc:
        return False, f"Could not read selected blog: {exc}"

    blog_slug = blog_file.stem
    images_root = (Path(".") / "images").resolve()
    related_image_dir = (Path("images") / blog_slug).resolve()
    deleted_images = 0

    try:
        if related_image_dir.exists() and related_image_dir.is_dir() and images_root in related_image_dir.parents:
            for p in sorted(related_image_dir.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                    deleted_images += 1
                elif p.is_dir():
                    p.rmdir()
            related_image_dir.rmdir()
        else:
            image_paths = [
                p for p in image_paths_from_markdown(md_text)
                if images_root in p.parents and p.parent.name == blog_slug
            ]
            for image_path in image_paths:
                if image_path.exists():
                    image_path.unlink()
                    deleted_images += 1
            _remove_empty_image_parents(image_paths)

        blog_file.unlink()
    except Exception as exc:
        return False, f"Could not delete selected blog: {exc}"

    detail = f"Deleted {blog_file.name}"
    if deleted_images:
        detail += f" and {deleted_images} related image file(s)"
    return True, detail + "."


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

with st.sidebar:
    st.header("New Blog")
    topic = st.text_area(
        "Topic",
        height=120,
        placeholder="Enter a topic, angle, or question...",
    )
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("Generate Blog", type="primary", use_container_width=True)

    # ✅ NEW: Past blogs list (keeps everything else intact)
    st.divider()
    st.subheader("Past Blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        # Build labels from file name + (optional) parsed title
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title} - {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("Load Selected Blog", use_container_width=True):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                # Load into session_state as if it were a run output
                st.session_state["last_out"] = {
                    "plan": None,          # old files don't include plan
                    "evidence": [],        # old files don't include evidence
                    "image_specs": [],     # optional (not persisted)
                    "final": md_text,      # markdown body
                }
                # also update the topic input to the title (best-effort) without changing UI
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)
                st.session_state["loaded_blog_path"] = str(selected_md_file.resolve())

        confirm_delete = st.checkbox("Confirm delete", key="confirm_delete_blog")
        if st.button("Delete Selected Blog", type="secondary", use_container_width=True, disabled=not confirm_delete):
            if selected_md_file:
                deleted_loaded_blog = st.session_state.get("loaded_blog_path") == str(selected_md_file.resolve())
                ok, message = delete_blog_file(selected_md_file)
                if ok:
                    if deleted_loaded_blog:
                        st.session_state["last_out"] = None
                        st.session_state.pop("loaded_blog_path", None)
                    st.success(message)
                    rerun_app()
                else:
                    st.error(message)

    

# Keep your topic input as-is; optionally prefill for next run after loading a blog
if "topic_prefill" in st.session_state and isinstance(st.session_state["topic_prefill"], str):
    # Do not mutate widgets; just keep as a hint.
    pass

# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Layout
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["Plan", "Evidence", "Preview", "Images", "Logs"]
)

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
        "md_with_placeholders": "",
        "image_specs": [],
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
                "images": len(current_state.get("image_specs", []) or []),
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
    stats_cols[0].metric("Words", f"{len(re.findall(r'\\b\\w+\\b', final_md_stats)):,}")
    stats_cols[1].metric("Sections", len(out.get("sections") or []))
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
            render_markdown_with_local_images(final_md)

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

            image_paths = image_paths_from_markdown(final_md)
            bundle_root = image_paths[0].parent if image_paths else Path("images")
            bundle = bundle_zip(final_md, md_filename, bundle_root)
            st.download_button(
                "Download Bundle (MD + images)",
                data=bundle,
                file_name=f"{safe_slug(blog_title)}_bundle.zip",
                mime="application/zip",
            )

    # --- Images tab ---
    with tab_images:
        st.subheader("Images")
        specs = out.get("image_specs") or []
        final_md_for_images = out.get("final") or ""
        current_images = image_paths_from_markdown(final_md_for_images)

        if not specs and not current_images:
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)

            if current_images:
                for p in current_images:
                    show_image(str(p), caption=p.name)

                z = images_zip(current_images[0].parent)
                if z:
                    st.download_button(
                        "Download Images (zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
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
