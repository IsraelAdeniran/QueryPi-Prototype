# RAG pipeline for QueryPi Prototype C.
# Handles loading documents, splitting, embeddings, vector search,
# query rewriting, context building, and answer generation.

import os
import subprocess
from typing import List, Dict, Any, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Full path to the Ollama executable (same style as Prototype A & B)
OLLAMA_EXE = r"C:\Users\adefo\AppData\Local\Programs\Ollama\ollama.exe"

# LLM model for rewriting queries and generating answers
LLM_MODEL_NAME = "llama3.2:latest"

# Embedding model used for creating vector representations
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

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


# Classify whether question is academic (yes/no)
def is_academic_question(question: str) -> bool:
    classify_prompt = (
        "You are a classifier. Determine if the student's question is an academic "
        "question related to school subjects (math, science, history, geography, etc.). "
        "Answer ONLY 'yes' or 'no'.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    result = ask_ollama(classify_prompt).strip().lower()

    if result.startswith("y"):
        return True
    return False


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

        # Cleaner metadata for citations
        for d in file_docs:
            d.metadata["source"] = os.path.basename(path)

        docs.extend(file_docs)

    return docs


# Split documents into overlapping text chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=200,
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
    # Enforce STRICT rewrite behaviour
    prompt = (
        "Rewrite the student's question using simple keywords. "
        "Add subject-specific keywords (e.g., history, biology, maths) to help retrieve the correct document topic."
        "Do not explain. Do not add anything. Do not change the meaning. "
        "Return ONLY the rewritten question.\n\n"
        f"Original question: {question}\n\n"
        "Rewritten:"
    )

    rewritten = ask_ollama(prompt)
    return rewritten.strip()


# Retrieve the top-k results (document chunks + similarity scores)
def retrieve_with_scores(vectordb: Chroma, query: str, k: int = 8):
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

    # Allow ignoring irrelevant context
    prompt = (
        "You are QueryPi, an offline school tutor. "
        "If the retrieved context is not relevant to the student's question, "
        "ignore the context and answer normally. "
        "If the context contains relevant facts, use them to answer. "
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

    # If question is not academic,  skip RAG entirely
    if not is_academic_question(question):
        answer = ask_ollama(
            f"You are QueryPi, a friendly offline tutor. "
            f"Answer the student's question naturally.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return {
            "answer": answer.strip(),
            "confidence": 0.0,
            "citations": [],
            "rewritten_query": question
        }

    rewritten = rewrite_query(history, question)
    docs_and_scores = retrieve_with_scores(vectordb, rewritten, k=8)

    if not docs_and_scores:
        return {
            "answer": "I couldn't find anything related to your documents.",
            "confidence": 0.0,
            "citations": [],
            "rewritten_query": rewritten
        }

    context, best_score, citations = build_context(docs_and_scores)

    # If context quality is too low, ignore RAG
    if best_score < 0.35:
        answer = ask_ollama(
            f"You are QueryPi, a friendly offline tutor. "
            f"Answer the student's question normally.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return {
            "answer": answer.strip(),
            "confidence": float(best_score),
            "citations": citations,
            "rewritten_query": rewritten
        }

    answer = generate_answer(history, question, context)

    return {
        "answer": answer,
        "confidence": float(best_score),
        "citations": citations,
        "rewritten_query": rewritten
    }