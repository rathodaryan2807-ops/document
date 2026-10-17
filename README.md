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

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and this README to the root of your GitHub repository.
2. In Streamlit Cloud, deploy/reboot the app.
3. Set your API key in **App settings → Secrets**:
   ```toml
   GEMINI_API_KEY = "paste-your-key-here"
   ```
   Or enter the key in the app sidebar.
4. Create/manage keys at https://aistudio.google.com/apikey.

## Important
- Never commit API keys to GitHub.
- Gemini calls require internet access and may consume API quota. OCR sends rendered scanned page images to Google.
- OCR is limited to 30 pages per scanned PDF to help control latency and API usage.
- Chat history is session-only; it is not persisted to a database and may reset when the session restarts.
- The code uses `gemini-3.8-flash`, based on the model name returned by the user's current Gemini API error. If Google changes model availability, update `MODEL_NAME` near the top of `app.py` to a model enabled for your API key.
- Run locally with: `streamlit run app.py`
