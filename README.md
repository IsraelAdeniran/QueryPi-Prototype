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

Pull the model:
    ollama pull qwen2.5:0.5b

### Run the Chatbot
    python app.py

### Commands
**/help** - Show help  
**/clear** - Clear memory  
**/quit** - Exit  

### Files
**app.py**           Main chatbot  
**chat_log.txt**     Automatically generated log  
