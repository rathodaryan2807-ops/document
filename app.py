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
st.caption("Upload PDFs, ask questions, and find relevant passages with page references.")

with st.sidebar:
    st.header("About DocMind")
    st.write(
        "DocMind searches the text of uploaded PDFs and shows relevant passages "
        "with page numbers. This starter version does not require an external AI API key."
    )
    st.info("Text-based PDFs work best. Scanned PDFs require OCR.")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)


def extract_pdf_pages(uploaded_file) -> List[Dict]:
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
        f"Loaded {len(uploaded_files)} PDF file(s), extracted "
        f"{len(all_pages)} readable page(s), and created "
        f"{len(chunks)} searchable passage(s)."
    )

    tab_ask, tab_summary, tab_documents = st.tabs(
        ["💬 Ask a question", "📝 Quick summary", "📄 Documents"]
    )

    chunk_texts = tuple(chunk["text"] for chunk in chunks)
    try:
        vectorizer, matrix = build_index(chunk_texts)
    except ValueError:
        st.error("There is not enough readable text to build a search index.")
        st.stop()

    with tab_ask:
        question = st.text_input(
            "What would you like to know?",
            placeholder="Example: What are the main findings of this document?",
        )
        top_k = st.slider(
            "Number of passages to retrieve",
            min_value=1,
            max_value=min(8, len(chunks)),
            value=min(4, len(chunks)),
        )

        if st.button("Search documents", type="primary", disabled=not question.strip()):
            query_vector = vectorizer.transform([question])
            scores = cosine_similarity(query_vector, matrix).flatten()
            ranked_indices = scores.argsort()[::-1]
            results = [
                (int(idx), float(scores[idx]))
                for idx in ranked_indices[:top_k]
                if scores[idx] > 0
            ]

            if not results:
                st.warning("No matching passage was found. Try different keywords.")
            else:
                st.subheader("Most relevant passages")
                st.caption(
                    "These are retrieved source passages, not guaranteed AI-generated conclusions. "
                    "Review the cited context before relying on it."
                )
                for rank, (idx, score) in enumerate(results, start=1):
                    item = chunks[idx]
                    with st.expander(
                        f"{rank}. {item['document']} — page {item['page']} "
                        f"(match score: {score:.3f})",
                        expanded=(rank == 1),
                    ):
                        st.write(item["text"])
                        st.caption(f"Source: {item['document']}, page {item['page']}")

                st.subheader("Evidence for your answer")
                st.write(
                    "Use the retrieved passages above to prepare an answer, checking each "
                    "source page. This starter version retrieves evidence but does not "
                    "generate a free-form answer with a large language model."
                )

    with tab_summary:
        st.write(
            "This lightweight overview shows an extract from each readable page. "
            "It is not a substitute for reading the original document."
        )
        for item in all_pages:
            with st.expander(f"{item['document']} — page {item['page']}"):
                preview = item["text"]
                if len(preview) > 900:
                    preview = preview[:900].rsplit(" ", 1)[0] + "…"
                st.write(preview)

    with tab_documents:
        st.write("Files and pages with readable text:")
        for filename in sorted({item["document"] for item in all_pages}):
            file_pages = [item for item in all_pages if item["document"] == filename]
            st.markdown(f"**{filename}** — {len(file_pages)} readable page(s)")
            st.caption("Pages: " + ", ".join(str(item["page"]) for item in file_pages))

else:
    st.info("Start by uploading a PDF file above.")
    st.markdown(
        """
        **Features in this starter project**
        - Multiple PDF uploads
        - Page-level text extraction
        - Search for relevant passages using TF-IDF
        - Source filename and page references
        - Lightweight page-by-page overview
        """
    )

st.divider()
st.caption(
    "DocMind AI starter project • Keep confidential documents private and upload "
    "only files you are authorized to use."
)
