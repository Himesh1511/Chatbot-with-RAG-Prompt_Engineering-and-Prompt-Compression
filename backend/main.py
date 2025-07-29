import os
import uuid
import shutil
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session as DBSession
from database import init_db, get_db, Session, Message, SessionFile, ContextualMemory
from rag import get_rag_chain, process_and_store_documents, retrieve_context_for_advanced_rag, process_large_code_file, estimate_tokens
from langchain_groq import ChatGroq
from advanced_prompting import (
    get_advanced_prompt, list_available_techniques, get_technique_description,
    advanced_prompting_engine
)
import requests

# Import compression functionality
try:
    from prompt_compression import (
        compress_prompt, compress_for_api, CompressionLevel, 
        CompressionResult, get_optimal_compression_level, prompt_compressor
    )
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    prompt_compressor = None
    print("Warning: Prompt compression module not available")

# Use absolute path for uploads directory
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

init_db()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)




def build_prompt(question, context="", use_advanced_prompting=False, technique=None):
    # Check if we should use advanced prompting techniques
    if use_advanced_prompting:
        return get_advanced_prompt(question, context, technique=technique)
    
    # Standard prompt building
    base_instructions = "You are a helpful AI assistant. Be concise in your responses."
    
    if context.strip():
        # When we have context from RAG with citations
        instructions = (
            f"{base_instructions} "
            f"You have been provided with relevant context from uploaded documents. "
            f"Use this context to provide a comprehensive answer."
        )
        
        # Check if context contains code files
        has_code_files = "=== CODE FILES FOR ANALYSIS ===" in context
        
        if has_code_files:
            prompt = (
                f"{instructions}\n\n"
                f"{context.strip()}\n\n"
                f"Question: {question.strip()}\n\n"
                f"Task: Provide a clear, comprehensive explanation of the topic. When explaining code, include relevant code snippets in your response.\n"
                f"IMPORTANT: When analyzing code, always include relevant code snippets in your response using proper markdown code blocks.\n"
                f"Show specific lines of code when explaining functions, classes, or discussing issues.\n\n"
                f"Assistant:"
            )
        else:
            prompt = (
                f"{instructions}\n\n"
                f"=== CONTEXT FROM UPLOADED DOCUMENTS ===\n"
                f"{context.strip()}\n"
                f"=== END CONTEXT ===\n\n"
                f"Question: {question.strip()}\n\n"
                f"Task: Provide a brief, factual answer based only on the context.\n"
                f"Do not explain your reasoning step by step. Keep it clear and concise.\n\n"
                f"Assistant:"
            )
        
    else:
        # When no context available, use general knowledge
        instructions = (
            f"{base_instructions} "
            f"Use your comprehensive knowledge to explain the user's question thoroughly."
        )
        
        prompt = (
            f"{instructions}\n\n"
            f"Question: {question.strip()}\n"
            f"Task: Provide a clear, comprehensive explanation of the topic.\n\n"
            f"Assistant:"
        )
    
    return prompt


@app.post("/chat")
def chat(
    session_id: str = Form(...),
    question: str = Form(...),
    db: DBSession = Depends(get_db),
):
    def update_contextual_memory(db_session, session_id, key, value, importance=1):
        memory = db_session.query(ContextualMemory).filter_by(session_id=session_id, memory_key=key).first()
        if memory:
            memory.memory_value = value
            memory.importance_score = importance
        else:
            new_memory = ContextualMemory(session_id=session_id, memory_key=key, memory_value=value, importance_score=importance)
            db_session.add(new_memory)
        db_session.commit()
    if not session_id or session_id == "null":
        s = Session()
        db.add(s)
        db.commit()
        session_id = s.id
    else:
        s = db.query(Session).filter(Session.id == session_id).first()
        if s is None:
            raise HTTPException(404, "Session not found")

    db.add(Message(session_id=session_id, role="user", content=question))
    db.commit()

    # Check if this session has any files attached
    files = db.query(SessionFile).filter_by(session_id=session_id).all()
    filepaths = [f.file_path for f in files]

    context, top_chunks = "", []
    if filepaths:
        print(f"[CHAT] Found {len(filepaths)} files in session {session_id}, using RAG")
        qa_chain = get_rag_chain(session_id)
        result = qa_chain({"query": question})
        answer = result.get("result", "Could not get a response.")
        top_chunks = [doc.page_content for doc in result.get("source_documents", [])]
        
        # Debug: Print retrieved context
        print(f"[MAIN DEBUG] Query: {question}")
        print(f"[MAIN DEBUG] Retrieved {len(top_chunks)} chunks")
        for i, chunk in enumerate(top_chunks):
            print(f"[MAIN DEBUG] Chunk {i+1}: {chunk[:200]}...")
        print(f"[MAIN DEBUG] Final answer: {answer[:200]}...")
    # If no files are in the session, just call the LLM directly
    else:
        print(f"[CHAT] No files in session {session_id}, using general knowledge")
        prompt = build_prompt(question, "")
        llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192")
        answer = llm.invoke(prompt).content
        top_chunks = []

    update_contextual_memory(db, session_id, "last_response", answer)

    db.add(Message(session_id=session_id, role="assistant", content=answer))
    db.commit()

    return {"answer": answer, "session_id": session_id, "chunks": top_chunks}


@app.post("/rag")
def rag_chat(
    session_id: str = Form(...),
    question: str = Form(...),
    technique: str = Form(None),
    use_advanced_prompting: bool = Form(False),
    files: list[UploadFile] = File(...),
    db: DBSession = Depends(get_db),
):
    print(f"[RAG ENDPOINT] Received {len(files)} files for session {session_id}")
    print(f"[RAG ENDPOINT] Question: {question[:100]}...")
    
    # Ensure session exists
    if not session_id or session_id == "null":
        s = Session()
        db.add(s)
        db.commit()
        session_id = s.id
        print(f"[RAG ENDPOINT] Created new session: {session_id}")
    
    filepaths = []
    code_files = []
    document_files = []
    
    # Define code file extensions
    code_extensions = {
        '.py', '.pyw', '.pyi',  # Python
        '.js', '.jsx', '.ts', '.tsx',  # JavaScript/TypeScript
        '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx',  # C/C++
        '.java', '.class',  # Java
        '.cs',  # C#
        '.php',  # PHP
        '.rb',  # Ruby
        '.go',  # Go
        '.rs',  # Rust
        '.swift',  # Swift
        '.kt', '.kts',  # Kotlin
        '.scala',  # Scala
        '.r', '.R',  # R
        '.m',  # MATLAB/Objective-C
        '.pl', '.pm',  # Perl
        '.lua',  # Lua
        '.sh', '.bash', '.zsh',  # Shell scripts
        '.ps1',  # PowerShell
        '.bat', '.cmd',  # Windows batch
        '.sql',  # SQL
        '.html', '.htm', '.xml',  # Markup
        '.css', '.scss', '.sass', '.less',  # Stylesheets
        '.json', '.yaml', '.yml', '.toml',  # Config files
        '.md', '.markdown',  # Markdown
        '.dockerfile',  # Docker
        '.gitignore', '.gitconfig',  # Git files
        '.makefile', '.mk',  # Make files
        '.cmake',  # CMake
        '.vue', '.dart', '.elm', '.ex', '.exs', '.erl', '.hrl',
        '.fs', '.fsx', '.hs', '.jl', '.nim', '.pas', '.pp',
        '.vb', '.vbs', '.asm', '.s', '.clj', '.cljs', '.coffee',
        '.groovy', '.tcl'
    }
    
    for i, file in enumerate(files):
        print(f"[RAG ENDPOINT] Processing file {i+1}: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
        dest = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
        try:
            with open(dest, "wb") as out:
                shutil.copyfileobj(file.file, out)
            filepaths.append(dest)
            db.add(SessionFile(session_id=session_id, file_path=dest))
            
            # Categorize files
            file_ext = os.path.splitext(file.filename)[1].lower()
            filename = file.filename.lower()
            
            if file_ext in code_extensions or filename in {'makefile', 'dockerfile', 'requirements.txt', 'package.json', 'composer.json'}:
                code_files.append(dest)
                print(f"[RAG ENDPOINT] Categorized as CODE file: {dest}")
            else:
                document_files.append(dest)
                print(f"[RAG ENDPOINT] Categorized as DOCUMENT file: {dest}")
            
            print(f"[RAG ENDPOINT] Saved file to: {dest}")
        except Exception as e:
            print(f"[RAG ENDPOINT] Error saving file {file.filename}: {e}")
            continue
    
    if not filepaths:
        print("[RAG ENDPOINT] No files were successfully saved")
        return {"answer": "❌ Failed to process uploaded files.", "chunks": [], "session_id": session_id}
    
    # Process document files (PDF, TXT) through RAG pipeline
    context_from_docs = ""
    if document_files:
        try:
            process_and_store_documents(document_files, session_id)
            print(f"[RAG ENDPOINT] Finished embedding {len(document_files)} document files for session {session_id}")
            
            # Get context from embedded documents
            if use_advanced_prompting:
                context_from_docs, _ = retrieve_context_for_advanced_rag(question, session_id)
            else:
                qa_chain = get_rag_chain(session_id)
                result = qa_chain({"query": question})
                context_from_docs = "\n\n".join([doc.page_content for doc in result.get("source_documents", [])])
        except Exception as e:
            print(f"[RAG ENDPOINT] Failed to embed document files: {e}")
    
    # Read code files directly for analysis with chunking support
    code_content = ""
    if code_files:
        print(f"[RAG ENDPOINT] Processing {len(code_files)} code files with chunking support")
        for code_file in code_files:
            try:
                filename = os.path.basename(code_file)
                print(f"[RAG ENDPOINT] Processing code file: {filename}")
                
                # Use the new chunking function to handle large files
                processed_content = process_large_code_file(code_file, question, "Explain", "Concise")
                
                # Add file markers for context
                code_content += f"\n\n=== FILE: {filename} ===\n{processed_content}\n=== END OF {filename} ===\n"
                
                # Log token estimate for debugging
                tokens_estimate = estimate_tokens(processed_content)
                print(f"[RAG ENDPOINT] File {filename} processed with ~{tokens_estimate} tokens")
                
            except Exception as e:
                print(f"[RAG ENDPOINT] Error processing code file {code_file}: {e}")
                filename = os.path.basename(code_file)
                code_content += f"\n\n=== FILE: {filename} (ERROR) ===\nError processing file: {str(e)}\n=== END OF {filename} ===\n"
    
    # Combine contexts
    combined_context = ""
    if context_from_docs:
        combined_context += f"=== CONTEXT FROM DOCUMENTS ===\n{context_from_docs}\n"
    if code_content:
        combined_context += f"=== CODE FILES FOR ANALYSIS ===\n{code_content}\n"
    
    # Generate response
    if use_advanced_prompting:
        prompt = build_prompt(question, combined_context, use_advanced_prompting=True, technique=technique)
        llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192")
        answer = llm.invoke(prompt).content
        top_chunks = []
    else:
        if document_files and not code_files:
            # Only documents - use traditional RAG
            qa_chain = get_rag_chain(session_id)
            result = qa_chain({"query": question})
            answer = result.get("result", "Could not get a response.")
            top_chunks = [doc.page_content for doc in result.get("source_documents", [])]
        else:
            # Code files or mixed - use direct analysis
            prompt = build_prompt(question, combined_context)
            llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192")
            answer = llm.invoke(prompt).content
            top_chunks = []

    db.add(Message(session_id=session_id, role="user", content=question))
    db.add(Message(session_id=session_id, role="assistant", content=answer))
    db.commit()

    # Return info about file processing
    file_info = {
        "total_files": len(filepaths),
        "code_files": len(code_files),
        "document_files": len(document_files),
        "processing_method": "direct_analysis" if code_files else "rag_embedding"
    }

    return {
        "answer": answer, 
        "chunks": top_chunks, 
        "session_id": session_id, 
        "technique_used": technique or "standard",
        "file_info": file_info
    }


@app.get("/sessions")
def list_sessions(db: DBSession = Depends(get_db)):
    sessions = db.query(Session).all()
    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at,
            "file_count": len(s.files)
        }
        for s in sessions
    ]

@app.post("/sessions")
def create_session(db: DBSession = Depends(get_db)):
    s = Session()
    db.add(s)
    db.commit()
    return {"id": s.id}

@app.post("/sessions/{session_id}/save")
def save_session(session_id: str, title: str = Form(...), db: DBSession = Depends(get_db)):
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(404, "Session not found")
    s.title = title
    db.commit()
    return {"status": "saved", "title": title}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: DBSession = Depends(get_db)):
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(404, "Session not found")
    for f in s.files:
        try:
            if os.path.exists(f.file_path):
                os.remove(f.file_path)
        except Exception:
            pass
    db.delete(s)
    db.commit()
    return {"status": "deleted"}

@app.get("/history/{session_id}")
def get_history(session_id: str, db: DBSession = Depends(get_db)):
    msgs = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at).all()
    return [{"role": m.role, "content": m.content, "created_at": m.created_at} for m in msgs]

@app.delete("/history/{session_id}")
def delete_history(session_id: str, db: DBSession = Depends(get_db)):
    msgs = db.query(Message).filter(Message.session_id == session_id).all()
    for m in msgs:
        db.delete(m)
    db.commit()
    return {"status": "deleted"}

@app.post("/settings/{session_id}")
def update_settings(session_id: str, db: DBSession = Depends(get_db)):
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(404, "Session not found")
    # Settings endpoint preserved for compatibility but no longer updates mode/tone
    db.commit()
    return {"status": "updated"}

# Advanced Prompting Endpoints
@app.post("/chat/advanced")
def advanced_chat(
    session_id: str = Form(...),
    question: str = Form(...),
    technique: str = Form(None),  # guided_contrast_learning, example_guided_learning, reason_act_framework, automated_chain_of_thought, executable_reasoning
    db: DBSession = Depends(get_db),
):
    """Enhanced chat endpoint with advanced prompting techniques"""
    if not session_id or session_id == "null":
        s = Session()
        db.add(s)
        db.commit()
        session_id = s.id
    else:
        s = db.query(Session).filter(Session.id == session_id).first()
        if s is None:
            raise HTTPException(404, "Session not found")

    db.add(Message(session_id=session_id, role="user", content=question))
    db.commit()

    # Check if this session has any files attached
    files = db.query(SessionFile).filter_by(session_id=session_id).all()
    filepaths = [f.file_path for f in files]

    context, top_chunks = "", []
    if filepaths:
        print(f"[ADVANCED CHAT] Found {len(filepaths)} files in session {session_id}, using RAG")
        context, top_chunks = retrieve_context_for_advanced_rag(question, session_id)
    else:
        print(f"[ADVANCED CHAT] No files in session {session_id}, using general knowledge")

    # Use advanced prompting
    prompt = build_prompt(question, context, use_advanced_prompting=True, technique=technique)
    print(f"[ADVANCED CHAT] Using technique: {technique or 'auto-detected'}")
    print(f"[ADVANCED CHAT] Built prompt with context length: {len(context)}")

    llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192")
    answer = llm.invoke(prompt).content

    db.add(Message(session_id=session_id, role="assistant", content=answer))
    db.commit()

    return {
        "answer": answer, 
        "session_id": session_id, 
        "chunks": top_chunks,
        "technique_used": technique or "auto_detected",
        "prompt_length_chars": len(prompt),
        "prompt_length_tokens": len(prompt.split())
    }




@app.get("/prompting/techniques")
def get_available_techniques():
    """Get list of available prompting techniques"""
    techniques = list_available_techniques()
    return {
        "techniques": [
            {
                "name": technique,
                "description": get_technique_description(technique)
            }
            for technique in techniques
        ]
    }


@app.get("/")
def root():
    return {
        "status": "ok",
        "compression_available": COMPRESSION_AVAILABLE,
        "features": {
            "advanced_prompting": True,
            "prompt_compression": COMPRESSION_AVAILABLE,
            "rag": True,
            "session_management": True
        }
    }
