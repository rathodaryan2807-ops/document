import io
import os
import re
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    import fitz  # PyMuPDF: render scanned PDF pages for Gemini-based OCR
except ImportError:
    fitz = None

st.set_page_config(page_title="DocMind AI", page_icon="📚", layout="wide")

MODEL_NAME = "gemini-3.8-flash"

st.markdown("""
<style>
.main-header {font-size:2.25rem;font-weight:800;letter-spacing:-.03em}
.sub-header {color:#6b7280;margin-top:-.6rem}
.source-card {border:1px solid rgba(128,128,128,.3);border-radius:10px;padding:12px;margin:8px 0}
.small-muted {color:#6b7280;font-size:.88rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📚 DocMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your AI-powered document research assistant — upload, ask, compare, summarize, and learn.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Google Gemini API key",
        type="password",
        value=st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")),
        help="Create a key at https://aistudio.google.com/apikey. Keep it private.",
    )
    st.caption(f"Gemini model: `{MODEL_NAME}`")
    st.divider()
    st.markdown("**Features**")
    st.markdown("• Multiple PDF upload\n• Document comparison\n• Chat history\n• Smart summaries\n• MCQ quiz generator\n• Page citations\n• PDF/Word export\n• Scanned PDF OCR")
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

def gemini_client():
    if not api_key:
        return None
    if genai is None:
        raise RuntimeError("Google GenAI package is missing. Install requirements.txt.")
    return genai.Client(api_key=api_key)

def ask_gemini(prompt: str, images=None) -> str:
    client = gemini_client()
    if client is None:
        raise RuntimeError("Add your Google Gemini API key in the sidebar or Streamlit secrets.")
    contents = [prompt]
    if images:
        contents.extend(images)
    response = client.models.generate_content(model=MODEL_NAME, contents=contents)
    result = getattr(response, "text", None)
    if not result:
        raise RuntimeError("Gemini returned an empty response.")
    return result.strip()

def extract_pdf_pages(uploaded_file, allow_ocr: bool) -> List[Dict[str, Any]]:
    raw = uploaded_file.getvalue()
    filename = uploaded_file.name
    pages = []
    try:
        reader = PdfReader(io.BytesIO(raw))
        for page_num, page in enumerate(reader.pages, start=1):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append({"file": filename, "page": page_num, "text": txt, "ocr": False})
    except Exception as exc:
        st.warning(f"Could not read text layer in {filename}: {exc}")

    # If the PDF has no useful text layer, optionally OCR a limited number of pages with Gemini vision.
    extracted_chars = sum(len(p["text"]) for p in pages)
    if allow_ocr and extracted_chars < 80:
        if not api_key:
            st.warning(f"{filename} appears scanned. Add a Gemini API key to OCR it.")
            return pages
        if fitz is None:
            st.warning("Scanned PDF OCR requires PyMuPDF. Check requirements.txt.")
            return pages
        try:
            doc = fitz.open(stream=raw, filetype="pdf")
            max_pages = min(len(doc), 30)
            st.info(f"Using Gemini OCR for {filename} ({max_pages} pages max). This may use API quota.")
            progress = st.progress(0, text=f"OCR: {filename}")
            for i in range(max_pages):
                pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
                from PIL import Image
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                prompt = (
                    "Transcribe all readable text in this document page as faithfully as possible. "
                    "Preserve headings, lists, numbers, and tables in plain text. Do not summarize. "
                    "If no readable text exists, return an empty string."
                )
                text = ask_gemini(prompt, images=[image])
                if text.strip():
                    pages.append({"file": filename, "page": i + 1, "text": text.strip(), "ocr": True})
                progress.progress((i + 1) / max_pages, text=f"OCR page {i+1}/{max_pages}")
            progress.empty()
            doc.close()
        except Exception as exc:
            st.warning(f"Gemini OCR failed for {filename}: {exc}")
    return pages

def split_pages_into_chunks(pages: List[Dict[str, Any]], chunk_size: int = 1400, overlap: int = 180):
    chunks = []
    for p in pages:
        text = re.sub(r"\s+", " ", p["text"]).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append({
                "file": p["file"], "page": p["page"], "text": text[start:end],
                "ocr": p.get("ocr", False)
            })
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks

def retrieve(question: str, chunks: List[Dict[str, Any]], top_k: int = 5):
    if not chunks:
        return []
    corpus = [c["text"] for c in chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=30000)
        matrix = vectorizer.fit_transform(corpus + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = scores.argsort()[::-1][:min(top_k, len(chunks))]
        return [dict(chunks[i], score=float(scores[i])) for i in ranked if scores[i] > 0]
    except ValueError:
        return []

def format_sources(sources):
    if not sources:
        return "No matching source passages were found."
    out = []
    for s in sources:
        marker = " (OCR)" if s.get("ocr") else ""
        out.append(f"[{s['file']}, page {s['page']}{marker}]\n{s['text']}")
    return "\n\n".join(out)

def generate_answer(question, sources):
    context = format_sources(sources)
    prompt = f"""You are DocMind AI, a careful document research assistant.
Answer the user's question using ONLY the source excerpts below. If the excerpts do not contain enough evidence, say so clearly.
Cite every important claim using [filename, page N]. Do not invent page numbers or facts.
Be clear and structured.

SOURCE EXCERPTS:
{context}

USER QUESTION:
{question}
"""
    return ask_gemini(prompt)

def create_report_pdf(title: str, body: str, sources: List[Dict[str, Any]]):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib import colors
    from xml.sax.saxutils import escape

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=48, bottomMargin=42)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_LEFT, textColor=colors.HexColor("#243b53")))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], leading=15, spaceAfter=8))
    story = [Paragraph(escape(title), styles["ReportTitle"]), Spacer(1, 12)]
    for para in body.split("\n"):
        if para.strip():
            story.append(Paragraph(escape(para).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>"), styles["BodyCustom"]))
        else:
            story.append(Spacer(1, 6))
    if sources:
        story += [Spacer(1, 12), Paragraph("Retrieved source passages", styles["Heading2"])]
        for s in sources:
            story.append(Paragraph(escape(f"{s['file']} — page {s['page']}"), styles["Heading3"]))
            story.append(Paragraph(escape(s["text"][:1200]), styles["BodyCustom"]))
    doc.build(story)
    return buffer.getvalue()

def create_report_docx(title: str, body: str, sources: List[Dict[str, Any]]):
    from docx import Document
    from docx.shared import Inches
    doc = Document()
    doc.add_heading(title, 0)
    doc.add_paragraph(f"Generated by DocMind AI on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    for para in body.split("\n"):
        if para.strip():
            doc.add_paragraph(para)
    if sources:
        doc.add_heading("Retrieved source passages", level=1)
        for s in sources:
            doc.add_heading(f"{s['file']} — page {s['page']}", level=2)
            doc.add_paragraph(s["text"][:2500])
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

uploaded_files = st.file_uploader(
    "📄 Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload several PDFs to search across them and compare their content.",
)

ocr_enabled = st.checkbox("Enable Gemini OCR for scanned/image-based PDFs (up to 30 pages per scanned file)", value=True)

if uploaded_files:
    fingerprint = [(f.name, len(f.getvalue())) for f in uploaded_files]
    if st.session_state.get("loaded_fingerprint") != fingerprint:
        with st.spinner("Reading PDFs and preparing searchable passages..."):
            all_pages = []
            for f in uploaded_files:
                all_pages.extend(extract_pdf_pages(f, ocr_enabled))
            st.session_state.doc_chunks = split_pages_into_chunks(all_pages)
            st.session_state.doc_names = [f.name for f in uploaded_files]
            st.session_state.loaded_fingerprint = fingerprint
            st.session_state.last_answer = ""
            st.session_state.last_sources = []
    st.success(f"Loaded {len(st.session_state.doc_names)} PDF(s) and {len(st.session_state.doc_chunks)} searchable passages.")
    with st.expander("📚 Documents loaded"):
        for name in st.session_state.doc_names:
            st.write(f"• {name}")
else:
    st.session_state.doc_chunks = []
    st.session_state.doc_names = []
    st.session_state.loaded_fingerprint = None

if not st.session_state.doc_chunks:
    st.info("Upload one or more PDFs to begin. For scanned PDFs, keep OCR enabled and provide a Gemini API key.")
    st.stop()

tab_chat, tab_summary, tab_quiz, tab_compare, tab_history = st.tabs([
    "💬 Ask documents", "📝 Smart summary", "🧠 Quiz generator", "⚖️ Compare PDFs", "🕘 Chat history"
])

with tab_chat:
    st.subheader("Ask questions with source citations")
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                with st.expander("Sources"):
                    for src in item["sources"]:
                        st.caption(f"{src['file']} — page {src['page']}")
                        st.write(src["text"][:700])
    question = st.text_input("Your question", key="question_input", placeholder="e.g. What are the key findings across these documents?")
    col1, col2 = st.columns([1, 3])
    with col1:
        answer_button = st.button("Ask Gemini", type="primary", use_container_width=True)
    with col2:
        st.caption("Answers are grounded in retrieved passages. Verify important details against the original PDF.")
    if answer_button and question.strip():
        with st.spinner("Searching documents and generating an answer..."):
            sources = retrieve(question.strip(), st.session_state.doc_chunks, top_k=6)
            try:
                answer = generate_answer(question.strip(), sources) if api_key else "Gemini key not provided. Here are the most relevant passages instead:\n\n" + format_sources(sources)
            except Exception as exc:
                answer = "Gemini answer generation failed; showing retrieved passages instead.\n\n" + format_sources(sources) + f"\n\n**Gemini error:** {exc}"
            st.session_state.chat_history.append({"role": "user", "content": question.strip(), "sources": []})
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
            st.session_state.last_answer = answer
            st.session_state.last_sources = sources
            st.rerun()
    if st.session_state.get("last_answer"):
        st.divider()
        st.subheader("Export latest answer")
        c1, c2 = st.columns(2)
        with c1:
            try:
                pdf_data = create_report_pdf("DocMind AI — Answer Report", st.session_state.last_answer, st.session_state.get("last_sources", []))
                st.download_button("⬇️ Download PDF report", data=pdf_data, file_name="docmind_answer_report.pdf", mime="application/pdf", use_container_width=True)
            except Exception as exc:
                st.warning(f"PDF export unavailable: {exc}")
        with c2:
            try:
                docx_data = create_report_docx("DocMind AI — Answer Report", st.session_state.last_answer, st.session_state.get("last_sources", []))
                st.download_button("⬇️ Download Word report", data=docx_data, file_name="docmind_answer_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
            except Exception as exc:
                st.warning(f"Word export unavailable: {exc}")

with tab_summary:
    st.subheader("Generate a smart summary")
    selected_summary_doc = st.selectbox("Document to summarize", ["All uploaded PDFs"] + st.session_state.doc_names, key="summary_doc")
    summary_style = st.selectbox("Summary style", ["Executive summary", "Detailed notes", "Key points for revision"], key="summary_style")
    if st.button("✨ Generate summary", type="primary"):
        chosen = st.session_state.doc_chunks if selected_summary_doc == "All uploaded PDFs" else [c for c in st.session_state.doc_chunks if c["file"] == selected_summary_doc]
        # Keep prompt manageable; spread chunks across the document so the summary covers more than the first pages.
        sample = chosen if len(chosen) <= 18 else [chosen[i] for i in sorted(set(round(i*(len(chosen)-1)/17) for i in range(18)))]
        context = format_sources(sample)
        prompt = f"""Create a {summary_style.lower()} of the following document excerpts. Organize with headings and bullet points.
Use only the provided text. Mention important definitions, claims, numbers, and conclusions. If excerpts are incomplete, state that this is a summary of sampled passages.
Cite key statements as [filename, page N].

EXCERPTS:
{context}
"""
        try:
            with st.spinner("Generating summary..."):
                summary = ask_gemini(prompt)
            st.session_state.summary_result = summary
            st.session_state.summary_sources = sample
        except Exception as exc:
            st.error(f"Summary failed: {exc}")
    if st.session_state.get("summary_result"):
        st.markdown(st.session_state.summary_result)
        s1, s2 = st.columns(2)
        with s1:
            try:
                st.download_button("Download summary PDF", create_report_pdf("DocMind AI — Summary", st.session_state.summary_result, st.session_state.get("summary_sources", [])), "docmind_summary.pdf", "application/pdf")
            except Exception as exc:
                st.caption(f"PDF export unavailable: {exc}")
        with s2:
            try:
                st.download_button("Download summary Word", create_report_docx("DocMind AI — Summary", st.session_state.summary_result, st.session_state.get("summary_sources", [])), "docmind_summary.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as exc:
                st.caption(f"Word export unavailable: {exc}")

with tab_quiz:
    st.subheader("Create MCQs from your PDFs")
    quiz_doc = st.selectbox("Quiz source", ["All uploaded PDFs"] + st.session_state.doc_names, key="quiz_doc")
    quiz_count = st.slider("Number of questions", min_value=5, max_value=20, value=10, step=5)
    quiz_difficulty = st.selectbox("Difficulty", ["Mixed", "Easy", "Medium", "Hard"])
    if st.button("🧠 Generate quiz", type="primary"):
        chosen = st.session_state.doc_chunks if quiz_doc == "All uploaded PDFs" else [c for c in st.session_state.doc_chunks if c["file"] == quiz_doc]
        sample = chosen if len(chosen) <= 14 else [chosen[i] for i in sorted(set(round(i*(len(chosen)-1)/13) for i in range(14)))]
        prompt = f"""Create exactly {quiz_count} multiple-choice questions at {quiz_difficulty.lower()} difficulty from the source excerpts.
For each question provide four options A-D, one correct answer, and a one-sentence explanation with citation [filename, page N].
Do not use information absent from the excerpts. Format each question clearly in Markdown and put the answer after the options.

SOURCE EXCERPTS:
{format_sources(sample)}
"""
        try:
            with st.spinner("Building your quiz..."):
                st.session_state.quiz_result = ask_gemini(prompt)
                st.session_state.quiz_sources = sample
        except Exception as exc:
            st.error(f"Quiz generation failed: {exc}")
    if st.session_state.get("quiz_result"):
        st.markdown(st.session_state.quiz_result)
        try:
            st.download_button("Download quiz as Word", create_report_docx("DocMind AI — Quiz", st.session_state.quiz_result, st.session_state.get("quiz_sources", [])), "docmind_quiz.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        except Exception as exc:
            st.caption(f"Word export unavailable: {exc}")

with tab_compare:
    st.subheader("Compare two uploaded PDFs")
    if len(st.session_state.doc_names) < 2:
        st.info("Upload at least two PDFs to use document comparison.")
    else:
        left_doc = st.selectbox("First document", st.session_state.doc_names, index=0, key="compare_left")
        right_default = 1 if len(st.session_state.doc_names) > 1 else 0
        right_doc = st.selectbox("Second document", st.session_state.doc_names, index=right_default, key="compare_right")
        compare_focus = st.text_input("Comparison focus (optional)", placeholder="e.g. methods, conclusions, dates, pros and cons")
        if st.button("⚖️ Compare documents", type="primary"):
            left_chunks = [c for c in st.session_state.doc_chunks if c["file"] == left_doc]
            right_chunks = [c for c in st.session_state.doc_chunks if c["file"] == right_doc]
            def sample_doc(chunks):
                return chunks if len(chunks) <= 10 else [chunks[i] for i in sorted(set(round(i*(len(chunks)-1)/9) for i in range(10)))]
            prompt = f"""Compare these two documents fairly and clearly. Cover common themes, differences, contradictions if supported, and practical takeaways. Cite claims as [filename, page N]. Do not assume that a missing detail means it is false.
Focus: {compare_focus or 'main ideas, evidence, conclusions, and differences'}

DOCUMENT A:
{format_sources(sample_doc(left_chunks))}

DOCUMENT B:
{format_sources(sample_doc(right_chunks))}
"""
            try:
                with st.spinner("Comparing documents..."):
                    st.session_state.compare_result = ask_gemini(prompt)
                    st.session_state.compare_sources = sample_doc(left_chunks) + sample_doc(right_chunks)
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")
        if st.session_state.get("compare_result"):
            st.markdown(st.session_state.compare_result)
            try:
                st.download_button("Download comparison Word", create_report_docx("DocMind AI — Document Comparison", st.session_state.compare_result, st.session_state.get("compare_sources", [])), "docmind_comparison.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as exc:
                st.caption(f"Word export unavailable: {exc}")

with tab_history:
    st.subheader("Saved in this session")
    st.caption("Chat history is kept in this browser session while the Streamlit session remains active. It is not a permanent database.")
    if not st.session_state.chat_history:
        st.info("Your questions and answers will appear here.")
    else:
        for idx in range(0, len(st.session_state.chat_history), 2):
            user_item = st.session_state.chat_history[idx]
            assistant_item = st.session_state.chat_history[idx + 1] if idx + 1 < len(st.session_state.chat_history) else None
            with st.expander(user_item["content"][:100] or "Question"):
                st.markdown("**Question**")
                st.write(user_item["content"])
                if assistant_item:
                    st.markdown("**Answer**")
                    st.markdown(assistant_item["content"])
                    if assistant_item.get("sources"):
                        st.markdown("**Sources**")
                        for src in assistant_item["sources"]:
                            st.caption(f"{src['file']} — page {src['page']}")
        history_text = "\n\n".join(
            ("QUESTION: " + item["content"] if item["role"] == "user" else "ANSWER: " + item["content"])
            for item in st.session_state.chat_history
        )
        try:
            st.download_button("Download chat history (Word)", create_report_docx("DocMind AI — Chat History", history_text, []), "docmind_chat_history.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        except Exception as exc:
            st.caption(f"History export unavailable: {exc}")

st.divider()
st.caption("Privacy note: uploaded PDFs are processed by this Streamlit app. When Gemini features are used, relevant extracted text (and scanned page images for OCR) are sent to Google's Gemini API. Avoid uploading confidential documents unless permitted.")
