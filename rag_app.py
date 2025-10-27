"""
============================================================================
APLICACIÓN RAG INDEPENDIENTE
============================================================================

Sistema RAG educativo para aprender sobre Máquinas de Boltzmann.
Esta aplicación se ejecuta de forma independiente para evitar conflictos.

Autor: Sistema de Física
Versión: 1.0.0
"""

import os
import streamlit as st
import pandas as pd
from typing import List, Dict
import warnings

# --- Importaciones Problemáticas ---
# Se importan aquí para que los errores queden aislados
try:
    import fitz
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    LANGCHAIN_OK = True
except ImportError as e:
    LANGCHAIN_OK = False
    LANGCHAIN_ERROR = e
# -----------------------------------

warnings.filterwarnings('ignore')

def render_rag_app():
    """Renderiza la aplicación RAG completa"""
    st.set_page_config(page_title="🎓 RAG - Máquinas de Boltzmann", layout="wide")
    st.title("🎓 Aprende sobre Máquinas de Boltzmann")
    st.markdown("### *Sistema RAG con Papers Científicos y Groq AI*")

    # Verificar API key y papers
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        api_configured = True
        st.success("✅ API Key de Groq configurada")
    except:
        api_configured = False
        st.error("❌ API Key de Groq no encontrada en .streamlit/secrets.toml")

    pdf_files = [f for f in os.listdir("./articles/") if f.endswith(".pdf")]
    if pdf_files:
        st.success(f"✅ {len(pdf_files)} papers encontrados en ./articles/")
    else:
        st.warning("⚠️ No se encontraron papers en ./articles/")

    # Verificar si LangChain está disponible
    if not LANGCHAIN_OK:
        st.error(f"❌ Error de dependencias: {LANGCHAIN_ERROR}")
        st.markdown("""
        **Asegúrate de haber instalado las dependencias:**
        ```bash
        pip install "langchain<0.2" "langchain-community<0.1" "langchain-groq<0.2" "chromadb<0.5" "sentence-transformers<3" "pymupdf<1.24"
        ```
        """)
        return

    # Si todo está listo, mostrar el botón para inicializar
    if 'rag_system' not in st.session_state and api_configured and pdf_files:
        if st.button("🚀 INICIALIZAR SISTEMA RAG", type="primary"):
            with st.spinner("⚡ Inicializando sistema RAG... Esto puede tardar unos minutos."):
                try:
                    # 1. Procesar PDFs
                    documents = []
                    for pdf_file in pdf_files:
                        doc = fitz.open(os.path.join("./articles/", pdf_file))
                        for page_num, page in enumerate(doc, start=1):
                            documents.append(Document(page_content=page.get_text(), metadata={"source": pdf_file, "page": page_num}))
                    
                    # 2. Split
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    
                    # 3. Embeddings
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    
                    # 4. Vector Store
                    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db_rag")
                    
                    # 5. LLM
                    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                    
                    # 6. QA Chain
                    prompt_template = """Contexto: {context}\n\nPregunta: {question}\n\nRespuesta:"""
                    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                    
                    st.session_state.rag_system = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(),
                        chain_type_kwargs={"prompt": PROMPT},
                        return_source_documents=True,
                    )
                    st.success("✅ Sistema RAG inicializado exitosamente!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error durante la inicialización: {e}")

    # Si el sistema está listo, mostrar la interfaz de chat
    if 'rag_system' in st.session_state:
        st.success("🤖 **Sistema RAG Activo** - ¡Haz tus preguntas!")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Pregunta sobre RBMs..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Consultando papers..."):
                    response = st.session_state.rag_system({"query": prompt})
                    st.markdown(response["result"])
                    
                    # Mostrar fuentes
                    sources = {doc.metadata['source'] for doc in response['source_documents']}
                    st.caption(f"Fuentes consultadas: {', '.join(sources)}")

            st.session_state.messages.append({"role": "assistant", "content": response["result"]})

if __name__ == "__main__":
    render_rag_app()