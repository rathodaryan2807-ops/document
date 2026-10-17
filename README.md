# DocMind AI — Google Gemini version

1. Get a Gemini API key from https://aistudio.google.com/apikey.
2. Run the app and paste the key into the sidebar, or configure `GEMINI_API_KEY` as an environment variable / Streamlit secret.
3. Never commit API keys to GitHub.

## Deploy
Upload `app.py` and `requirements.txt` to the root of a GitHub repository, then deploy at https://share.streamlit.io/.

For Streamlit Cloud Secrets, use:
```toml
GEMINI_API_KEY = "your-key-here"
```

## Run locally on Windows
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This app uses TF-IDF to retrieve passages and Gemini to compose answers. Scanned PDFs need OCR.
