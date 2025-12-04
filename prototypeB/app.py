from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma

from rag_pipeline import (
    load_documents,
    split_documents,
    build_vector_store,
    rag_answer,
)


# Set up the RAG system (documents, vector store, and memory)
def initialise_system() -> (Chroma, List[Dict[str, str]]):
    print("Loading documents from data/documents ...")
    docs = load_documents("data/documents")
    print(f"Loaded {len(docs)} document pages/items.")

    print("Splitting documents into chunks ...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    print("Building vector store (this may take a moment) ...")
    vectordb = build_vector_store(chunks)
    print("Vector store is ready and saved to the 'db' folder.\n")

    # Simple conversation history for conversational RAG
    history: List[Dict[str, str]] = []

    return vectordb, history


# Print the model's answer, confidence and citation details
def print_answer(result: Dict[str, Any]):
    print("\nTutor:")
    print(result["answer"])
    print()

    confidence = result.get("confidence", 0.0)

    # Similarity score
    print(f"(Similarity score: {confidence:.2f})")

    # Show if RAG was used or skipped
    if confidence == 0.0 and result.get("citations", []) == []:
        print("(RAG not used — either a conversational question or a topic not covered by your documents)")
        print("Sources: [no document sources used]\n")
        return
    else:
        print("(RAG used — academic question detected from your documents)")

    # Only print rewritten query if RAG was used
    rewritten = result.get("rewritten_query", "")
    if rewritten:
        print(f"Rewritten search query: {rewritten}")

    print("Sources:")
    citations = result.get("citations", [])

    if not citations:
        print("  - [no sources found]")
        print()
        return

    # Avoid printing the same source + page twice
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
            print(f"  - {source}, page {page} (score {score:.2f})")
        else:
            print(f"  - {source} (score {score:.2f})")

    print()


# Main chat loop (CLI) for the RAG tutor
def main():
    print("QueryPi Prototype B – Enhanced RAG Chatbot")
    print("Make sure Ollama is running and the 'llama3.2:latest' model is pulled.")
    print("Type 'quit' or 'exit' to leave.\n")

    vectordb, history = initialise_system()

    while True:
        question = input("You: ").strip()

        # Allow the user to exit the chatbot
        if question.lower() in {"quit", "exit"}:
            print("Exiting RAG chatbot. Goodbye!")
            break

        # Ignore empty messages
        if not question:
            continue

        # Run the full RAG pipeline for this question
        result = rag_answer(vectordb, history, question)

        # Show the answer and metadata
        print_answer(result)

        # Save this turn in conversation history
        history.append({"user": question, "bot": result["answer"]})


if __name__ == "__main__":
    main()