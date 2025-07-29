
import os
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import numpy as np
from qdrant_client import QdrantClient
import uuid

from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Qdrant collection and server URL settings
QDRANT_COLLECTION = "vector-chunks"
QDRANT_URL = "http://localhost:6333"

def load_documents(filepaths: List[str]) -> List[any]:
    """
    Loads documents from a list of file paths using the appropriate LangChain loader.
    Supports PDF, TXT, and various programming language files.
    """
    documents = []
    
    # Define supported programming language extensions
    programming_extensions = {
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
        '.vue',  # Vue.js
        '.dart',  # Dart
        '.elm',  # Elm
        '.ex', '.exs',  # Elixir
        '.erl', '.hrl',  # Erlang
        '.fs', '.fsx',  # F#
        '.hs',  # Haskell
        '.jl',  # Julia
        '.nim',  # Nim
        '.pas', '.pp',  # Pascal
        '.vb', '.vbs',  # Visual Basic
        '.asm', '.s',  # Assembly
        '.clj', '.cljs',  # Clojure
        '.coffee',  # CoffeeScript
        '.groovy',  # Groovy
        '.tcl',  # Tcl
    }
    
    for path in filepaths:
        if not os.path.exists(path):
            print(f"[LOADER] File not found, skipping: {path}")
            continue

        try:
            file_ext = Path(path).suffix.lower()
            filename = Path(path).name.lower()
            
            if path.lower().endswith(".pdf"):
                loader = PyMuPDFLoader(path)
                documents.extend(loader.load())
                print(f"[LOADER] Successfully loaded PDF: {path}")
            elif file_ext in programming_extensions or filename in {'makefile', 'dockerfile', 'requirements.txt', 'package.json', 'composer.json'}:
                # Load programming files as text with UTF-8 encoding
                try:
                    loader = TextLoader(path, encoding="utf-8")
                    loaded_docs = loader.load()
                    
                    # Add file type metadata for better context
                    for doc in loaded_docs:
                        doc.metadata["file_type"] = file_ext or "config"
                        doc.metadata["language"] = get_language_from_extension(file_ext, filename)
                        doc.metadata["source_file"] = Path(path).name
                    
                    documents.extend(loaded_docs)
                    print(f"[LOADER] Successfully loaded {get_language_from_extension(file_ext, filename)} file: {path}")
                except UnicodeDecodeError:
                    # Try with different encodings if UTF-8 fails
                    for encoding in ['latin-1', 'cp1252', 'ascii']:
                        try:
                            loader = TextLoader(path, encoding=encoding)
                            loaded_docs = loader.load()
                            
                            for doc in loaded_docs:
                                doc.metadata["file_type"] = file_ext or "config"
                                doc.metadata["language"] = get_language_from_extension(file_ext, filename)
                                doc.metadata["source_file"] = Path(path).name
                            
                            documents.extend(loaded_docs)
                            print(f"[LOADER] Successfully loaded {get_language_from_extension(file_ext, filename)} file with {encoding} encoding: {path}")
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        print(f"[LOADER] Failed to decode file with any encoding: {path}")
            elif path.lower().endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
                documents.extend(loader.load())
                print(f"[LOADER] Successfully loaded TXT: {path}")
            else:
                print(f"[LOADER] Unsupported file type, skipping: {path}")
        except Exception as e:
            print(f"[LOADER] Error loading file {path}: {e}")
            
    return documents

def get_language_from_extension(ext: str, filename: str) -> str:
    """
    Maps file extensions to programming language names for better context.
    """
    language_map = {
        '.py': 'Python', '.pyw': 'Python', '.pyi': 'Python',
        '.js': 'JavaScript', '.jsx': 'JavaScript', 
        '.ts': 'TypeScript', '.tsx': 'TypeScript',
        '.c': 'C', '.h': 'C Header',
        '.cpp': 'C++', '.hpp': 'C++ Header', '.cc': 'C++', '.cxx': 'C++',
        '.java': 'Java', '.class': 'Java Bytecode',
        '.cs': 'C#',
        '.php': 'PHP',
        '.rb': 'Ruby',
        '.go': 'Go',
        '.rs': 'Rust',
        '.swift': 'Swift',
        '.kt': 'Kotlin', '.kts': 'Kotlin Script',
        '.scala': 'Scala',
        '.r': 'R', '.R': 'R',
        '.m': 'MATLAB/Objective-C',
        '.pl': 'Perl', '.pm': 'Perl Module',
        '.lua': 'Lua',
        '.sh': 'Shell Script', '.bash': 'Bash Script', '.zsh': 'Zsh Script',
        '.ps1': 'PowerShell',
        '.bat': 'Batch Script', '.cmd': 'Command Script',
        '.sql': 'SQL',
        '.html': 'HTML', '.htm': 'HTML', '.xml': 'XML',
        '.css': 'CSS', '.scss': 'SCSS', '.sass': 'Sass', '.less': 'Less',
        '.json': 'JSON', '.yaml': 'YAML', '.yml': 'YAML', '.toml': 'TOML',
        '.md': 'Markdown', '.markdown': 'Markdown',
        '.vue': 'Vue.js',
        '.dart': 'Dart',
        '.elm': 'Elm',
        '.ex': 'Elixir', '.exs': 'Elixir Script',
        '.erl': 'Erlang', '.hrl': 'Erlang Header',
        '.fs': 'F#', '.fsx': 'F# Script',
        '.hs': 'Haskell',
        '.jl': 'Julia',
        '.nim': 'Nim',
        '.pas': 'Pascal', '.pp': 'Pascal',
        '.vb': 'Visual Basic', '.vbs': 'VBScript',
        '.asm': 'Assembly', '.s': 'Assembly',
        '.clj': 'Clojure', '.cljs': 'ClojureScript',
        '.coffee': 'CoffeeScript',
        '.groovy': 'Groovy',
        '.tcl': 'Tcl',
    }
    
    # Check special filenames
    if filename in {'makefile', 'dockerfile'}:
        return filename.title()
    elif filename in {'requirements.txt', 'package.json', 'composer.json'}:
        return 'Configuration'
    
    return language_map.get(ext, 'Code')

# --- File Parsing Logic ---

def get_rag_chain(session_id: str) -> RetrievalQA:
    """
    Creates and returns a LangChain RetrievalQA chain for standard RAG queries.
    """
    embeddings = get_embeddings_model()
    
    client = QdrantClient(url=QDRANT_URL)
    
    # Check if collection exists, if not create it
    ensure_collection_exists(client, embeddings)
    
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=get_embeddings_model()
    )
    
    # Remove session filter to fix retrieval issue
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )
    
    llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
    
    default_template = """
    You are a helpful AI assistant.
    Use the context from uploaded documents to answer the question.
    Be concise and do not explain your reasoning step-by-step.
    
    Context: {context}
    
    Question: {question}
    
    Assistant:"""
    prompt = PromptTemplate(
        template=default_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

def retrieve_context_for_advanced_rag(query: str, session_id: str) -> Tuple[str, List[str]]:
    """
    Retrieves context from the vector store for a specific session, 
    to be used with the advanced prompting module.
    """
    embeddings = get_embeddings_model()
    client = QdrantClient(url=QDRANT_URL)
    
    # Check if collection exists, if not create it
    ensure_collection_exists(client, embeddings)
    
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=get_embeddings_model()
    )
    
    # Remove session filter to fix retrieval issue
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )
    
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)
    
    # Debug: Print what was retrieved
    print(f"[RAG DEBUG] Query: {query}")
    print(f"[RAG DEBUG] Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs):
        print(f"[RAG DEBUG] Chunk {i+1}: {doc.page_content[:200]}...")
    
    # Format the context string and source chunks
    context_str = "\n\n".join([doc.page_content for doc in docs])
    source_chunks = [doc.page_content for doc in docs]
    
    return context_str, source_chunks

# Singleton model instance for embedding
_embedding_model = None

def get_embeddings_model():
    """ Loads the SentenceTransformer model as a LangChain Embeddings object. """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedding_model

def ensure_collection_exists(client: QdrantClient, embeddings):
    """
    Ensures that the Qdrant collection exists. If not, creates it with proper vector configuration.
    """
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if QDRANT_COLLECTION not in collection_names:
            print(f"[QDRANT] Collection '{QDRANT_COLLECTION}' not found. Creating...")
            
            # Get embedding dimension (all-MiniLM-L6-v2 has 384 dimensions)
            embedding_dimension = 384
            
            client.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
            )
            print(f"[QDRANT] Collection '{QDRANT_COLLECTION}' created successfully.")
        else:
            print(f"[QDRANT] Collection '{QDRANT_COLLECTION}' already exists.")
            
    except Exception as e:
        print(f"[QDRANT] Error checking/creating collection: {e}")
        # Re-raise the exception so the calling function can handle it
        raise e


def estimate_tokens(text: str) -> int:
    """Estimate token count assuming 1 token ≈ 4 characters"""
    return len(text) // 4

def chunk_text(text: str, max_tokens: int = 3000) -> List[str]:
    """Split text into chunks that fit within token limits"""
    max_chars = max_tokens * 4
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def chunk_code_intelligently(file_content: str, max_tokens: int = 3000) -> List[str]:
    """Chunk code intelligently by lines and logical sections"""
    lines = file_content.split('\n')
    chunks, current_chunk, current_size = [], [], 0
    for line in lines:
        tokens = estimate_tokens(line)
        if current_size + tokens > max_tokens:
            chunks.append('\n'.join(current_chunk))
            current_chunk, current_size = [], 0
        current_chunk.append(line)
        current_size += tokens
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return chunks

def process_large_code_file(file_path: str, question: str, mode: str, tone: str, max_tokens: int = 3000) -> str:
    """Handle large files by chunking and processing relevant parts"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        encodings = ['latin-1', 'cp1252', 'ascii']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            return f"Error: Could not read {file_path} due to encoding issues."
    except Exception as e:
        return f"Error: {str(e)}"
    tokens_estimate = estimate_tokens(content)
    if tokens_estimate <= max_tokens:
        return content
    chunks = chunk_code_intelligently(content, max_tokens)
    relevant_chunks = find_relevant_chunks(question, chunks)
    return '\n'.join(relevant_chunks)

def find_relevant_chunks(question: str, chunks: List[str]) -> List[str]:
    """Identify and return relevant code chunks based on the question"""
    keywords = re.findall(r'\w+', question.lower())
    scored_chunks = [(chunk, sum(kw in chunk.lower() for kw in keywords)) for chunk in chunks]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:3]]


# Existing Function
def process_and_store_documents(filepaths: List[str], session_id: str):
    """
    Loads, splits, and stores documents in the Qdrant vector store for a specific session.
    """
    try:
        print("[PROCESS] Stage 1: Loading documents...")
        documents = load_documents(filepaths)
        if not documents:
            print(f"[PROCESS] No documents were loaded for session {session_id}.")
            return
        print(f"[PROCESS] Stage 1 successful. Loaded {len(documents)} documents.")
        # Debug: Print first 200 chars of each document
        for i, doc in enumerate(documents):
            print(f"[PROCESS] Document {i+1} preview: {doc.page_content[:200]}...")

        print("[PROCESS] Stage 2: Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"[PROCESS] Stage 2 successful. Created {len(split_docs)} chunks.")

        print("[PROCESS] Stage 3: Adding session_id to metadata...")
        for doc in split_docs:
            doc.metadata["session_id"] = session_id
        print("[PROCESS] Stage 3 successful.")

        print("[PROCESS] Stage 4: Getting embeddings model...")
        embeddings = get_embeddings_model()
        print("[PROCESS] Stage 4 successful.")

        print("[PROCESS] Stage 5: Ensuring Qdrant collection exists...")
        client = QdrantClient(url=QDRANT_URL)
        ensure_collection_exists(client, embeddings)
        print("[PROCESS] Stage 5 successful.")

        print("[PROCESS] Stage 6: Storing documents in Qdrant...")
        vectorstore = Qdrant.from_documents(
            documents=split_docs,
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION
        )
        print("[PROCESS] Stage 6 successful. Documents stored in Qdrant.")

    except Exception as e:
        print(f"\n\n--- DETAILED ERROR ---")
        print(f"An error occurred in process_and_store_documents for session {session_id}.")
        import traceback
        traceback.print_exc()
        print(f"--- END DETAILED ERROR ---\n\n")
        # Re-raise the exception so the API can handle it
        raise e




