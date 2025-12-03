# QueryPi-Prototype
A Final Year Project

## Prototype A – Basic Offline Chatbot

Prototype A is the first working version of the QueryPi project.  
It introduces offline LLM interaction using a simple, clear architecture.

### Features
- Offline LLM inference via Ollama
- Educational system prompt
- Short-term memory (last 3 messages)
- Command handling
- Chat logging for transparency

### Requirements
- Python 3.10+
- Ollama installed
- Model: qwen2.5:0.5b

Pull the LLM:

    ollama pull qwen2.5:0.5b

### Run the Chatbot
    python app.py

### Commands
**/help** - Show help  
**/clear** - Clear memory  
**/quit** - Exit  

### Files
**app.py**   -        Main chatbot  
**chat_log.txt**  -   Automatically generated log  


## Prototype B – Enhanced RAG Chatbot (QueryPi)

This prototype implements an offline Retrieval-Augmented Generation system
using real models and locally stored documents.

It can:
- Load PDF/TXT documents
- Split them into text chunks
- Embed them using BAAI/bge-small-en
- Store embeddings in ChromaDB
- Rewrite questions using llama3.2:latest
- Retrieve relevant chunks
- Build a document-grounded context
- Generate answers with confidence and citations
- Maintain short conversational memory

### Requirements

    pip install langchain langchain-community chromadb pypdf sentence-transformers langchain-huggingface langchain-text-splitters

Pull the LLM:

    ollama pull llama3.2:latest

### Run

    python app.py

Place PDFs in:

prototypeB/data/documents/

### Example questions
- Explain photosynthesis simply.
- Summarise the algebra document.
- Give three key points from the history PDF.

### Files
**app.py**   -        Main chatbot  
**rag_pipeline.py**  - RAG pipleine
