# DocMind AI — Document Research Assistant

A starter M.Sc. CA project for uploading PDFs, searching their contents, asking questions, and inspecting page-linked evidence.

## Features
- Upload and index multiple PDFs
- Extract text with page references
- Split long pages into overlapping passages
- Search passages with TF-IDF ranking
- Optional grounded answers using the OpenAI API
- Evidence panel with filename and page number
- Research chat history for the current Streamlit session
- Clear workspace control

## Requirements
- Python 3.10 or newer
- Internet connection only if installing packages or using the optional OpenAI API

## Run on Windows
Open PowerShell or the VS Code terminal in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this command for the current terminal session, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Alternatively, install packages and run Streamlit without activating the environment, using `.venv\Scripts\python.exe`.

## Enable AI answers (optional)
1. Create an API key with your chosen provider.
2. In the app sidebar, paste it into **OpenAI API key**.
3. Enable **Use AI to compose answers**.

You can use the app's local passage retrieval without an API key. Do not commit API keys to GitHub. AI mode sends retrieved document passages to the configured provider; avoid confidential material unless you have permission.

## How it works
1. `pypdf` extracts selectable text page by page.
2. Text is split into overlapping passages.
3. `TfidfVectorizer` ranks passages relevant to a question.
4. Optional LLM generation answers using the retrieved context and asks for page citations.
5. Each passage retains the original filename and page number.

## Known limitations
- Scanned/image-only PDFs need OCR; this starter version reports when no selectable text is found.
- TF-IDF is a lexical baseline, not semantic vector search.
- Uploaded files and conversation history are held in Streamlit session memory; there is no persistent database or multi-user authentication yet.
- AI-generated answers can still be wrong. Check the cited passages before relying on a result.

## Suggested next steps for a senior submission
1. Add semantic embeddings and a persistent vector database (FAISS or Chroma).
2. Add OCR for scanned documents.
3. Add document collections, user authentication, and database persistence.
4. Add export to Markdown/PDF, summary generation, and source highlighting.
5. Add automated tests, threat modelling, and a documented evaluation set.
6. Deploy with Docker and document the architecture, API, test cases, limitations, and results.

## Suggested report structure
- Abstract and problem statement
- Literature review and existing-system limitations
- Requirements and feasibility study
- System architecture and data-flow diagrams
- Database and module design
- Algorithms: chunking, TF-IDF retrieval, optional RAG answer generation
- Implementation screenshots
- Test plan, test cases, and evaluation
- Security, privacy, limitations, and future scope
- Conclusion and references
