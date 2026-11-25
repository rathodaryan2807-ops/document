# DocMind AI — Advanced Gemini Edition

A Streamlit document research assistant using Google Gemini and TF-IDF retrieval.

## Features
- Upload multiple PDFs and search across them
- Ask questions with source filename and page citations
- Session chat history
- Smart summaries
- MCQ quiz generation
- Compare two PDFs
- Export answers, summaries, quizzes, comparisons, and chat history to PDF or Word
- OCR for scanned/image-based PDFs through Gemini vision (up to 30 pages per scanned PDF)
- Hidden API key configuration via `.streamlit/secrets.toml` / Streamlit Secrets (key is never shown in the UI)
- Sidebar **Refresh models & check API** button and automatic model discovery
- Automatic exponential-backoff retries and fallback model handling for temporary `503 UNAVAILABLE` spikes

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and this README to the root of your GitHub repository (keep `.streamlit/secrets.toml` local only — it is ignored by `.gitignore`).
2. In Streamlit Cloud, deploy/reboot the app.
3. Set your API key in **App settings → Secrets**:
   ```toml
   GEMINI_API_KEY = "paste-your-key-here"
   ```
4. Create/manage keys at https://aistudio.google.com/apikey.

## Important
- Never commit API keys to GitHub.
- Gemini calls require internet access and may consume API quota. OCR sends rendered scanned page images to Google.
- OCR is limited to 30 pages per scanned PDF to help control latency and API usage.
- Chat history is session-only; it is not persisted to a database and may reset when the session restarts.
- The default model is `gemini-3.8-flash`. If Google returns `503 UNAVAILABLE` (temporary high demand) or changes model availability, click **Check API key & load models** in the sidebar to select another model enabled for your key or set a fallback model.
- Run locally with: `streamlit run app.py`
