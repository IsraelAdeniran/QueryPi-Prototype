# RAG pipeline for QueryPi Prototype B.
# Handles loading documents, splitting, embeddings, vector search,
# query rewriting, context building, and answer generation.

import os
import subprocess
from typing import List, Dict, Any, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Full path to the Ollama executable (same style as Prototype A)
OLLAMA_EXE = r"C:\Users\adefo\AppData\Local\Programs\Ollama\ollama.exe"

# LLM model for rewriting queries and generating answers
LLM_MODEL_NAME = "llama3.2:latest"

# Embedding model used for creating vector representations
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"

# Where Chroma will save the vector index
PERSIST_DIR = "db"


# Send a prompt to Ollama and return the reply
def ask_ollama(prompt: str) -> str:
    process = subprocess.Popen(
        [OLLAMA_EXE, "run", LLM_MODEL_NAME],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    output, error = process.communicate(input=prompt)

    # Clean Windows encoding issues

    if process.returncode != 0:
        return f"[Model error]: {error or 'Unknown error'}"

    if not output.strip():
        return "[Model returned no output]"

    return output.strip()


# Load all PDF and TXT documents from the documents folder
def load_documents(doc_dir: str):
    docs = []

    if not os.path.isdir(doc_dir):
        raise FileNotFoundError(f"Document folder not found: {doc_dir}")

    for filename in os.listdir(doc_dir):
        path = os.path.join(doc_dir, filename)

        # Process only normal files
        if not os.path.isfile(path):
            continue

        lower = filename.lower()

        # Load PDF files
        if lower.endswith(".pdf"):
            loader = PyPDFLoader(path)

        # Load TXT files
        elif lower.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")

        # Skip everything else
        else:
            continue

        file_docs = loader.load()

        # Add missing metadata for citations
        for d in file_docs:
            d.metadata["source"] = path

        docs.extend(file_docs)

    return docs


# Split documents into overlapping text chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


# Build the vector database using BGE-small embeddings
def build_vector_store(chunks):
    os.makedirs(PERSIST_DIR, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Load existing DB instead of rebuilding every run
    if os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )


# Rewrite the question so retrieval works better
def rewrite_query(history: List[Dict[str, str]], question: str) -> str:
    # Rewrite only latest question
    prompt = (
        "Rewrite ONLY the student's latest question. "
        "Ignore all previous conversation history. "
        "Do NOT mix topics. "
        "Return a short, clearer, more searchable query. "
        "Do NOT answer the question.\n\n"
        f"Original question: {question}\n\n"
        "Rewritten:"
    )

    rewritten = ask_ollama(prompt)
    return rewritten.strip()


# Retrieve the top-k results (document chunks + similarity scores)
def retrieve_with_scores(vectordb: Chroma, query: str, k: int = 5):
    return vectordb.similarity_search_with_relevance_scores(query, k=k)


# Build context from retrieved chunks + gather citation info
def build_context(docs_and_scores) -> Tuple[str, float, List[Dict[str, Any]]]:
    context_parts = []
    citations = []
    best_similarity = 0.0

    for doc, score in docs_and_scores:
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page")

        context_parts.append(doc.page_content)

        # Convert distance → similarity
        similarity = 1 - float(score)

        if similarity > best_similarity:
            best_similarity = similarity

        citations.append({
            "source": source,
            "page": page,
            "score": similarity
        })

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, best_similarity, citations


# Generate a grounded answer using the context + conversation history
def generate_answer(history: List[Dict[str, str]],
                    question: str,
                    context: str) -> str:

    history_lines = []
    for turn in history[-3:]:
        history_lines.append(f"Student: {turn['user']}")
        history_lines.append(f"Tutor: {turn['bot']}")

    history_text = "\n".join(history_lines)

    prompt = (
        "You are QueryPi, an offline school tutor. "
        "Use ONLY the information in the context. "
        "If the answer is not in the context, say you are not sure. "
        "Do NOT invent information.\n\n"
        f"{history_text}\n\n"
        "Context:\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n\n"
        f"Student question: {question}\n\n"
        "Answer:"
    )

    answer = ask_ollama(prompt)
    return answer.strip()


# Full RAG pipeline: rewrite → retrieve → build context → answer
def rag_answer(vectordb: Chroma,
               history: List[Dict[str, str]],
               question: str) -> Dict[str, Any]:

    rewritten = rewrite_query(history, question)
    docs_and_scores = retrieve_with_scores(vectordb, rewritten, k=5)

    if not docs_and_scores:
        return {
            "answer": "I couldn't find anything related to your documents.",
            "confidence": 0.0,
            "citations": [],
            "rewritten_query": rewritten
        }

    context, best_score, citations = build_context(docs_and_scores)

    answer = generate_answer(history, question, context)

    return {
        "answer": answer,
        "confidence": float(best_score),
        "citations": citations,
        "rewritten_query": rewritten
    }