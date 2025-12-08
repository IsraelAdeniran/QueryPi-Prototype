# QueryPi Prototype C – Streamlit UI for PDF upload and RAG question answering

import os
import streamlit as st

from rag_pipeline import (
    load_documents,
    split_documents,
    build_vector_store,
    rag_answer
)

DOCUMENT_FOLDER = "C:/QueryPi-Prototype/prototypeC/data/documents"


# Initialise system
@st.cache_resource
def initialise_system():
    docs = load_documents(DOCUMENT_FOLDER)
    chunks = split_documents(docs)
    vectordb = build_vector_store(chunks)
    return vectordb, len(docs), len(chunks)


st.set_page_config(page_title="QueryPi – Prototype C", layout="wide")
st.title("QueryPi – Prototype C")
st.write("Upload PDFs and ask questions about them.\n")


# File upload section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    save_path = os.path.join(DOCUMENT_FOLDER, uploaded_file.name)
    existed = os.path.exists(save_path)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if not existed:
        st.success(f"Uploaded file: {uploaded_file.name}")
        st.cache_resource.clear()
        st.rerun()
    else:
        st.info(f"{uploaded_file.name} already exists. No rebuild needed.")


# Manual rebuild button
if st.button("Rebuild database"):
    st.cache_resource.clear()
    st.rerun()


# Document list
st.subheader("Document list")
files = os.listdir(DOCUMENT_FOLDER)
if len(files) == 0:
    st.write("No documents have been uploaded.")
else:
    for f in files:
        st.write(f"- {f}")


# Initialise vector database
vectordb, doc_count, chunk_count = initialise_system()


# Status panel
st.subheader("Status panel")
st.write(f"Database status: {'Ready' if vectordb else 'Not built'}")
st.write(f"Documents loaded: {doc_count}")
st.write(f"Chunks created: {chunk_count}")


# Question and answer section
question = st.text_input("Ask a question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        if vectordb is None:
            st.error("No vector database. Upload a PDF or rebuild.")
        else:
            history = []
            result = rag_answer(vectordb, history, question)

            st.subheader("Tutor:")
            st.write(result["answer"])

            st.write(f"Similarity score: {result.get('confidence', 0.0):.2f}")

            if result.get("confidence", 0.0) == 0.0 and result.get("citations", []) == []:
                st.info("RAG not used. No relevant document information found.\n")
            else:
                st.info("RAG used for this question.")
                st.write(f"Rewritten query: {result.get('rewritten_query', '')}")

            # Citations
            st.subheader("Sources:")
            citations = result.get("citations", [])
            if not citations:
                st.write("- No sources found")
            else:
                seen = set()
                for c in citations:
                    key = (c["source"], c["page"])
                    if key in seen:
                        continue
                    seen.add(key)

                    source = c["source"]
                    page = c["page"]
                    score = c["score"]

                    if page is not None:
                        st.write(f"- {source}, page {page} (score {score:.2f})")
                    else:
                        st.write(f"- {source} (score {score:.2f})")