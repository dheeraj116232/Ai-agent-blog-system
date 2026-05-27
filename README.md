# AI Blog Agent

AI Blog Agent is a Streamlit application for generating structured technical blog posts with optional research, image generation, Markdown export, PDF export, and saved blog management.

The app uses LangGraph to route each generation through planning, optional web research, section writing, image planning, image generation, and final Markdown assembly.

## Features

- Generate complete blog drafts from a topic prompt.
- Automatically decide whether web research is needed.
- Use Tavily search for evidence-backed, recent topics.
- Generate technical diagrams or illustrative images when useful.
- Preview generated Markdown with local images.
- Download output as Markdown, PDF, or a ZIP bundle.
- View, load, and delete previous generated blogs.
- Keep generated images organized per blog topic.

## Tech Stack

- Streamlit for the web interface
- LangGraph for the generation workflow
- Pydantic for structured data validation
- AWS Bedrock (Claude) for text generation (via `AWS_BEARER_TOKEN_BEDROCK`)
- Image generation: existing providers (Grok/Gemini/OpenAI). Eden can be wired later via `EDEN_API_KEY`.
- Tavily for optional web research
- Pillow for PDF export


## Project Structure

```text
.
|-- Backend.py          # LangGraph workflow, model routing, research, image generation
|-- Frontend.py         # Streamlit UI, downloads, saved blog management
|-- bwa_backend.py      # Thin import wrapper for the compiled graph app
|-- requirements.txt    # Python dependencies
|-- .gitignore          # Ignored local files, secrets, logs, generated outputs
`-- README.md
```

Generated blog files and generated images are local runtime outputs and are ignored by Git.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
TEXT_MODEL_PROVIDER=owl_alpha
IMAGE_MODEL_PROVIDER=grok_openrouter

OWL_ALPHA_API_KEY=sk-or-v1-...
OWL_ALPHA_MODEL=openrouter/owl-alpha

XAI_GROK_API_KEY=sk-or-v1-...
GROK_IMAGE_MODEL=x-ai/grok-imagine-image-quality

TAVILY_API_KEY=...
```

`TAVILY_API_KEY` is optional. Without it, the app can still generate closed-book drafts, but research-backed blogs will not receive search evidence.

## Run Locally

```powershell
python -m streamlit run Frontend.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Configuration

### Text Generation

Recommended text provider (Bedrock):

```text
TEXT_MODEL_PROVIDER=bedrock
AWS_BEARER_TOKEN_BEDROCK=...
AWS_REGION_BEDROCK=ap-south-1
BEDROCK_TEXT_MODEL_ID=anthropic.claude-opus-4-7
```


`OPENROUTER_API_KEY` is also supported as a fallback for Owl Alpha.

Supported text provider values include:

```text
bedrock
owl_alpha
grok
gemini
openai
local
auto
```

When `TEXT_MODEL_PROVIDER=auto`, the backend tries configured providers. If `AWS_BEARER_TOKEN_BEDROCK` is set, Bedrock is preferred.


### Image Generation

Recommended image provider:

```text
IMAGE_MODEL_PROVIDER=grok_openrouter
XAI_GROK_API_KEY=sk-or-v1-...
GROK_IMAGE_MODEL=x-ai/grok-imagine-image-quality
```

Supported image provider values include:

```text
grok_openrouter
xai
gemini
openai
auto
```

Official xAI image keys are also supported through `GROK_API_KEY` or `XAI_API_KEY`, but this project is configured for OpenRouter-style keys by default.

## Outputs

After generating a blog, the Preview tab provides:

- Markdown download
- PDF download
- ZIP bundle containing Markdown and generated images

Generated files are saved locally as:

```text
<blog_slug>.md
images/<blog_slug>/
```

The Past Blogs panel lets you load or delete previous generated blogs. Deleting a blog also removes its matching per-blog image folder when available.

## Deploy on Streamlit Cloud

1. Push the repository to GitHub.
2. Create a new Streamlit app from the repository.
3. Set `Frontend.py` as the main file.
4. Add secrets in Streamlit Cloud.

Recommended Streamlit secrets:

```toml
TEXT_MODEL_PROVIDER = "owl_alpha"
IMAGE_MODEL_PROVIDER = "grok_openrouter"

OWL_ALPHA_API_KEY = "sk-or-v1-..."
OWL_ALPHA_MODEL = "openrouter/owl-alpha"

XAI_GROK_API_KEY = "sk-or-v1-..."
GROK_IMAGE_MODEL = "x-ai/grok-imagine-image-quality"

TAVILY_API_KEY = "..."
```

Do not commit `.env` to GitHub.

## Git Hygiene

The repository ignores:

- `.env`
- virtual environments
- Python cache files
- Streamlit logs
- generated blog Markdown files
- generated images

Before pushing, check:

```powershell
git status --short
```

## Troubleshooting

If localhost refuses to connect, start the app again:

```powershell
python -m streamlit run Frontend.py
```

If text generation shows an OpenRouter connection error, retry after a moment. The backend includes retries for transient provider/network failures.

If images look unrelated to the current blog, regenerate the blog. New outputs are stored in a per-blog image folder to avoid reusing images from previous topics.

## Security Notes

- Keep API keys in `.env` locally or in Streamlit Cloud secrets.
- Rotate any API key that was accidentally shared publicly.
- Do not commit generated logs or local output files unless intentionally publishing samples.
