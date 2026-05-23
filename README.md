# AI Blog Agent

A Streamlit-based blog writing agent using LangGraph and OpenAI.

## Setup

1. Create a Python environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your keys:
   ```text
   OPENAI_API_KEY=sk-...
   TAVILY_API_KEY=...
   GOOGLE_API_KEY=...
   ```

If you want to use Gemini for text generation instead of OpenAI, add:
   ```text
   TEXT_MODEL_PROVIDER=gemini
   GEMINI_MODEL=gemini-2.5-flash
   ```


## Run locally

```powershell
python -m streamlit run Frontend.py
```

## Deploy on Streamlit

1. Push this repository to GitHub.
2. Create a new Streamlit app from the GitHub repo.
3. Set the following secrets in the Streamlit dashboard:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
   - `GOOGLE_API_KEY`
4. Use `Frontend.py` as the main app file.

## Notes

- The backend now uses lazy OpenAI client creation so the app imports cleanly even before secrets are configured.
- Do not commit `.env` to source control.
