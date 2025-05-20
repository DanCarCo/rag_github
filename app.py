import streamlit as st
import textwrap
from pdf_loader import extract_text_from_pdf
from embedder import create_vector_store, search_similar
from rag_answer import generate_answer
import os

os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"

st.set_page_config(page_title="PDF Q&A", layout="centered")

st.title("📄 Pregunta a tu PDF con GitHub AI")

uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")
if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    chunks = textwrap.wrap(raw_text, 500)
    index, vectors, texts = create_vector_store(chunks)
    st.success("PDF procesado. Ahora puedes hacer preguntas.")

    question = st.text_input("¿Qué quieres preguntar?")
    if question:
        relevant_chunks = search_similar(question, index, texts)
        context = "\n".join(relevant_chunks)
        answer = generate_answer(context, question)
        st.markdown("**Respuesta:**")
        st.write(answer)
