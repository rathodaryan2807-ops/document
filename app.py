import re
from typing import List, Dict

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="DocMind AI — Document Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DocMind AI")
st.caption("Upload PDFs, ask questions, and find answers with page references.")

with st.sidebar:
    st.header("About DocMind")
    st.write(
        "DocMind searches the text of your uploaded PDFs and shows the most "
        "relevant passages with page numbers. This starter version runs without "
        "an external AI API key."
    )
    st.info("Tip: Text-based PDFs work best. Scanned/image-only PDFs need OCR.")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

def extract_pdf_pages(uploaded_file) -> List[Dict]:
    """Extract non-empty text from each PDF page."""
    pages = []
    try:
        reader = PdfReader(uploaded_file)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append({
                    "document": uploaded_file.name,
                    "page": page_number,
                    "text": text,
                })
    except Exception as exc:
        st.error(f"Could not read {uploaded_file.name}: {exc}")
    return pages


def split_into_chunks(pages: List[Dict], chunk_size: int = 1100, overlap: int = 180):
    chunks = []
    for item in pages:
        text = item["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "document": item["document"],
                    "page": item["page"],
                    "text": chunk_text,
                })
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


@st.cache_data(show_spinner=False)
def build_index(texts: tuple):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


if uploaded_files:
    all_pages = []
    for uploaded_file in uploaded_files:
        all_pages.extend(extract_pdf_pages(uploaded_file))

    chunks = split_into_chunks(all_pages)

    if not chunks:
        st.warning(
            "No selectable text was found. This may be a scanned PDF. "
            "Try a text-based PDF or add OCR support."
        )
        st.stop()

    st.success(
        f"Loaded {len(uploaded_files)} PDF file(s), "
        f"extracted {len(all_pages)} page(s), and created {len(chunks)} searchable passage(s)."
