    for line in body.splitlines():
        story.append(Paragraph(escape(line), body_style) if line.strip() else Spacer(1, 5))
    if sources:
        story.append(Paragraph('Source passages', styles['Heading2']))
        for s in sources:
            story.append(Paragraph(escape(f"{s['file']} — page {s['page']}"), styles['Heading3']))
            story.append(Paragraph(escape(s['text'][:1200]), body_style))
    doc.build(story); return buf.getvalue()

uploads = st.file_uploader('📄 Upload one or more PDF documents', type=['pdf'], accept_multiple_files=True)
ocr = st.checkbox('Enable Gemini OCR for scanned PDFs (first 20 pages per file)', value=False)
if uploads:
    fingerprint = [(f.name, len(f.getvalue())) for f in uploads]
    if st.session_state.loaded_fingerprint != fingerprint:
        with st.spinner('Reading PDFs and preparing searchable passages...'):
            pages = []
            for file in uploads: pages.extend(extract_pdf_pages(file, ocr))
            st.session_state.doc_chunks = make_chunks(pages)
            st.session_state.doc_names = [f.name for f in uploads]
            st.session_state.loaded_fingerprint = fingerprint
            for key in ('chat_history', 'summary_result', 'quiz_result', 'compare_result', 'last_answer'):
                st.session_state[key] = [] if key == 'chat_history' else ''
    st.success(f"Loaded {len(st.session_state.doc_names)} PDF(s) and {len(st.session_state.doc_chunks)} searchable passages.")
else:
    st.session_state.doc_chunks, st.session_state.doc_names, st.session_state.loaded_fingerprint = [], [], None

if not st.session_state.doc_chunks:
    st.info('Upload a PDF with selectable text to begin. Enable OCR for scanned PDFs.')
    st.stop()

chat_tab, summary_tab, quiz_tab, compare_tab, history_tab = st.tabs(['💬 Ask documents', '📝 Smart summary', '🧠 Quiz generator', '⚖️ Compare PDFs', '🕘 Chat history'])
with chat_tab:
    st.subheader('Ask questions with page citations')
    for item in st.session_state.chat_history:
        with st.chat_message(item['role']):
            st.markdown(item['content'])
            if item.get('sources'):
                with st.expander('Sources'):
                    for src in item['sources']:
                        st.caption(f"{src['file']} — page {src['page']}"); st.write(src['text'][:700])
    question = st.text_input('Your question', key='question_input', placeholder='What are the main findings?')
    if st.button('Ask Gemini', type='primary'):
        if not question.strip(): st.warning('Please enter a question.')
        else:
            sources = retrieve(question.strip(), st.session_state.doc_chunks)
            try:
                with st.spinner('Searching documents and generating an answer...'):
                    answer = ask_gemini(f"Answer using ONLY these excerpts. If evidence is insufficient, say so. Cite claims as [filename, page N].\\n\\n{format_sources(sources)}\\n\\nQuestion: {question}")
            except Exception as exc:
                answer = f'Gemini could not answer. Relevant passages:\\n\\n{format_sources(sources)}\\n\\nAPI error: {exc}'
            st.session_state.chat_history += [{'role':'user','content':question.strip(),'sources':[]},{'role':'assistant','content':answer,'sources':sources}]
            st.session_state.last_answer, st.session_state.last_sources = answer, sources
            st.rerun()
    if st.session_state.last_answer:
        c1, c2 = st.columns(2)
        with c1: st.download_button('Download answer PDF', export_pdf('DocMind AI — Answer', st.session_state.last_answer, st.session_state.last_sources), 'docmind_answer.pdf', 'application/pdf')
        with c2: st.download_button('Download answer Word', export_docx('DocMind AI — Answer', st.session_state.last_answer, st.session_state.last_sources), 'docmind_answer.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')

with summary_tab:
    st.subheader('Generate a smart summary')
    doc_choice = st.selectbox('Document', ['All uploaded PDFs'] + st.session_state.doc_names, key='summary_doc')
    style = st.selectbox('Summary style', ['Executive summary', 'Detailed notes', 'Key points for revision'])
    if st.button('✨ Generate summary', type='primary'):
        selected = st.session_state.doc_chunks if doc_choice == 'All uploaded PDFs' else [c for c in st.session_state.doc_chunks if c['file'] == doc_choice]
        sources = sample_even(selected, 18)
        prompt = f'Create a {style.lower()} with headings and bullet points. Use only these excerpts. If sampled, disclose that coverage may be incomplete. Cite claims as [filename, page N].\\n\\n{format_sources(sources)}'
        try:
            with st.spinner('Generating summary...'):
                st.session_state.summary_result = ask_gemini(prompt); st.session_state.summary_sources = sources
        except Exception as exc: st.error(f'Summary failed: {exc}')
    if st.session_state.summary_result:
        st.markdown(st.session_state.summary_result)
        a,b=st.columns(2)
        with a: st.download_button('Download summary PDF', export_pdf('DocMind AI — Summary', st.session_state.summary_result, st.session_state.summary_sources), 'docmind_summary.pdf', 'application/pdf')
        with b: st.download_button('Download summary Word', export_docx('DocMind AI — Summary', st.session_state.summary_result, st.session_state.summary_sources), 'docmind_summary.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')

with quiz_tab:
    st.subheader('Create multiple-choice questions')
    quiz_doc = st.selectbox('Quiz source', ['All uploaded PDFs'] + st.session_state.doc_names, key='quiz_doc')
    count = st.slider('Number of questions', 5, 20, 10, step=5)
    difficulty = st.selectbox('Difficulty', ['Mixed', 'Easy', 'Medium', 'Hard'])
    if st.button('🧠 Generate quiz', type='primary'):
        selected = st.session_state.doc_chunks if quiz_doc == 'All uploaded PDFs' else [c for c in st.session_state.doc_chunks if c['file'] == quiz_doc]
        sources = sample_even(selected, 14)
        prompt = f'Create exactly {count} MCQs at {difficulty.lower()} difficulty. Each needs A-D options, one correct answer, explanation and page citation. Use only these excerpts.\\n\\n{format_sources(sources)}'
        try:
            with st.spinner('Building quiz...'):
                st.session_state.quiz_result = ask_gemini(prompt); st.session_state.quiz_sources = sources
        except Exception as exc: st.error(f'Quiz generation failed: {exc}')
    if st.session_state.quiz_result:
        st.markdown(st.session_state.quiz_result)
        st.download_button('Download quiz as Word', export_docx('DocMind AI — Quiz', st.session_state.quiz_result, st.session_state.quiz_sources), 'docmind_quiz.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')

with compare_tab:
    st.subheader('Compare two uploaded PDFs')
    if len(st.session_state.doc_names) < 2: st.info('Upload at least two PDFs to compare them.')
    else:
        first = st.selectbox('First document', st.session_state.doc_names, key='compare_first')
        second = st.selectbox('Second document', [n for n in st.session_state.doc_names if n != first], key='compare_second')
        focus = st.text_input('Comparison focus (optional)', placeholder='Methods, conclusions, dates, etc.')
        if st.button('⚖️ Compare documents', type='primary'):
            left = sample_even([c for c in st.session_state.doc_chunks if c['file'] == first], 10)
            right = sample_even([c for c in st.session_state.doc_chunks if c['file'] == second], 10)
            sources = left + right
            prompt = f'Compare common themes, differences and supported contradictions. Do not infer that missing information is false. Cite [filename, page N]. Focus: {focus or "main ideas and conclusions"}.\\n\\nDOCUMENT A:\\n{format_sources(left)}\\n\\nDOCUMENT B:\\n{format_sources(right)}'
            try:
                with st.spinner('Comparing documents...'):
                    st.session_state.compare_result = ask_gemini(prompt); st.session_state.compare_sources = sources
            except Exception as exc: st.error(f'Comparison failed: {exc}')
        if st.session_state.compare_result:
            st.markdown(st.session_state.compare_result)
            st.download_button('Download comparison as Word', export_docx('DocMind AI — PDF Comparison', st.session_state.compare_result, st.session_state.compare_sources), 'docmind_comparison.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')

with history_tab:
    st.subheader('Chat history')
    st.caption('History lasts for the current Streamlit session; it is not stored in a permanent database.')
    if not st.session_state.chat_history: st.info('Your questions and answers will appear here.')
    else:
        for i in range(0, len(st.session_state.chat_history), 2):
            user = st.session_state.chat_history[i]
            assistant = st.session_state.chat_history[i+1] if i+1 < len(st.session_state.chat_history) else None
            with st.expander(user['content'][:100] or 'Question'):
                st.markdown('**Question**'); st.write(user['content'])
                if assistant: st.markdown('**Answer**'); st.markdown(assistant['content'])
        history = '\\n\\n'.join(f"{x['role'].upper()}: {x['content']}" for x in st.session_state.chat_history)
        st.download_button('Download chat history as Word', export_docx('DocMind AI — Chat History', history, []), 'docmind_chat_history.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')

st.divider()
