import subprocess
import time

# SYSTEM PROMPT: Defines the assistant's behaviour
SYSTEM_PROMPT = (
    "You are an educational assistant for QueryPi. "
    "Speak clearly, explain simply, and be helpful to students. "
    "Keep answers short unless the user asks for more detail."
)

# Stores last few messages for short-term memory
conversation_history = []
MAX_HISTORY = 3


# Build full prompt: system prompt + memory + latest user input
def build_prompt(user_message):
    history_text = ""

    # Add recent conversation turns to the prompt
    for turn in conversation_history[-MAX_HISTORY:]:
        history_text += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

    # Final composed prompt sent to the model
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{history_text}"
        f"User: {user_message}\nAssistant:"
    )

    return full_prompt


# Send prompt to Ollama model and return the output
def ask_ollama(prompt, model="phi3.5:latest"):
    OLLAMA = r"C:\Users\adefo\AppData\Local\Programs\Ollama\ollama.exe"

    process = subprocess.Popen(
        [OLLAMA, "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    output, error = process.communicate(input=prompt)

    # Sanitize output for Windows
    output = output.encode("utf-8", "replace").decode("utf-8")

    if process.returncode != 0:
        return f"[Model error]: {error or 'Unknown error'}"

    if not output.strip():
        return "[Model returned no output]"

    return output.strip()

# Log each interaction for debugging
def log_interaction(user, bot):
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"USER: {user}\n")
        f.write(f"BOT: {bot}\n")
        f.write("-----\n")


# Main chat loop (CLI)
def main():
    print("QueryPi Prototype A – Basic Offline Chatbot")
    print("Type '/help' for commands.\n")

    while True:
        user_message = input("You: ").strip()

        # Command Handling
        if user_message == "/quit":
            print("Exiting chatbot. Goodbye!")
            break

        if user_message == "/help":
            print("Commands:")
            print("/help  - Show help menu")
            print("/clear - Clear conversation memory")
            print("/quit  - Exit the chatbot")
            continue

        if user_message == "/clear":
            conversation_history.clear()
            print("Memory cleared.\n")
            continue

        # Normal Coversation Flow
        prompt = build_prompt(user_message)
        bot_response = ask_ollama(prompt)

        # Store turn in memory + log to file
        conversation_history.append({"user": user_message, "bot": bot_response})
        log_interaction(user_message, bot_response)

        # Display the response
        print(f"Bot: {bot_response}\n")
        time.sleep(0.2)


if __name__ == "__main__":
    main()