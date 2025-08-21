#Coding AI Assistant

An AI-powered coding assistant that combines RAG (Retrieval-Augmented Generation), advanced prompting strategies, and prompt compression to provide smarter, context-aware code help. It uses Groq LLMs, LangChain, and Qdrant for efficient retrieval and fast inference.

#Features

Context-Aware Assistance → Uses RAG with Qdrant to fetch relevant context from uploaded files (code, PDFs, text).

Agentic RAG → Dynamic agent-driven retrieval and response generation.

Advanced Prompting → Supports prompting strategies like contrastive, few-shot, ReAct, auto-CoT, program-of-thought.

Prompt Compression → Reduces long prompts while preserving quality to optimize LLM performance.

Memory Management → Stores sessions, messages, uploaded files, and contextual memory using SQLite + SQLAlchemy.

LLM Integration → Powered by Groq-hosted models like LLaMA 3 and Gemma for blazing fast responses.

#Tech Stack

Backend: FastAPI

Database: SQLite + SQLAlchemy ORM

Vector DB: Qdrant

AI/ML: LangChain, LangGraph, HuggingFace Embeddings, Groq LLM API

Utilities: AsyncIO, FPDF (for exporting if needed)

#Project Structure
Chatbot-with-RAG-Prompt_Engineering-and-Prompt-Compression/
│── backend/
│   ├── main.py                # FastAPI entry point, API routes & SSE
│   ├── database.py            # Session, message & memory DB models
│   ├── rag.py                 # Core RAG pipeline with Qdrant
│   ├── agentic_rag.py         # Agentic RAG orchestration
│   ├── advanced_prompting.py  # Prompting strategies
│   ├── prompt_compression.py  # Prompt compression engine
│   ├── requirements.txt       # Python dependencies
│
│── frontend/
│   ├── index.html             # Web UI
│   ├── script.js              # Frontend logic (SSE, API calls)
│   ├── style.css              # Styling

#Setup & Run
1. Clone the repo
git clone https://github.com/Himesh1511/Chatbot-with-RAG-Prompt_Engineering-and-Prompt-Compression.git
cd Coding-AI-Assistant

2. Setup environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
pip install -r requirements.txt

3. Run Qdrant (Vector DB)

You need Qdrant running locally (default http://localhost:6333).
Easiest way is via Docker:

docker run -p 6333:6333 qdrant/qdrant

4. Set Groq API Key
# Linux/Mac
export GROQ_API_KEY="your_api_key"

# Windows PowerShell
setx GROQ_API_KEY "your_api_key"

5. Start the backend
uvicorn main:app --reload --port 8000

#Future Improvements

VS Code / browser extension integration.

Add multi-model support (switch between Gemma, LLaMA, Mixtral).

Richer agent tool-calling workflows.

Fine-grained LLM observability & evaluation.
