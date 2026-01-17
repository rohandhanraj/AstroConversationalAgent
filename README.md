# 🌟 Astro Conversational Insight Agent

A production-ready conversational AI service for personalized astrological insights using **RAG (Retrieval Augmented Generation)**, **LangChain**, and **LangGraph**.

## 📋 Features

### Core Features
- ✅ **Multi-turn Conversational AI** with context-aware responses
- ✅ **RAG Implementation** using FAISS/ChromaDB for grounded astrological knowledge
- ✅ **Personalized Responses** based on zodiac signs, moon signs, ascendant, and nakshatra
- ✅ **Multi-language Support** (English and Hindi) with automatic translation
- ✅ **Session Management** with conversation memory
- ✅ **LangGraph Workflow** for structured agent processing
- ✅ **Multiple LLM Support** (OpenAI GPT, Ollama models)

### Bonus Features
- ✅ **Confidence Scoring** for response quality assessment
- ✅ **Hallucination Filtering** to ensure safe predictions
- ✅ **Model Fallback** mechanism for reliability
- ✅ **Embeddings-based Retrieval** with similarity scoring
- ✅ **Error Handling** with retry policies and safe fallbacks
- ✅ **FastAPI REST API** with comprehensive documentation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Layer                           │
│  (REST API endpoints, request validation, error handling)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  LangGraph Agent                            │
│  (Conversation flow: retrieve → generate → filter → translate)│
└──────────┬───────────┬───────────┬──────────────────────────┘
           │           │           │
┌──────────▼───┐  ┌───▼────────┐  ┌▼─────────────────────────┐
│  Retrieval   │  │    LLM     │  │   Translation Service    │
│    Engine    │  │  Wrapper   │  │   (Google Translate)     │
│ (FAISS/Chroma)│ │(OpenAI/Ollama)│└────────────────────────┘
└──────────────┘  └────────────┘
       │                 │
┌──────▼──────┐   ┌─────▼──────┐
│   Vector    │   │  Session   │
│   Store     │   │  Memory    │
└─────────────┘   └────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.9+
- **UV** (recommended for 10-100x faster installs) OR pip
- OpenAI API key (for GPT models) OR Ollama installed locally
- 4GB+ RAM for embeddings and vector store

#### Local Ollama Installation
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Pull local model LLama3.1
```bash
ollama pull llama3.1
```

### Step 1: Install UV (Recommended)

UV is an ultra-fast Python package installer. Installation is optional but highly recommended.

```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS (Homebrew)
brew install uv

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip
pip install uv
```

Why UV? It's **10-100x faster** than pip!

### Step 2: Clone and Setup

**With UV (Fast ⚡):**
```bash
# Create project directory
mkdir astro-agent
cd astro-agent

# Create virtual environment with UV
uv venv

# Activate
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (super fast!)
uv pip install -r requirements.txt
```

**Without UV (Traditional):**
```bash
# Create project directory
mkdir astro-agent
cd astro-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

**For OpenAI:**
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your_api_key_here
```

**For Ollama (Local):**
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
```

### Step 4: Initialize Vector Store

```bash
# The vector store will be built automatically on first run
# Or you can build it manually:
python -c "from app.retrieval import RetrievalEngine; engine = RetrievalEngine(data_dir='./data'); engine.build_vector_store()"
```

## 🚀 Running the Application

### Start the Server

**Quick Start (with automated script):**
```bash
# Development mode with auto-reload
uv run start.py
```

**Manual Start:**
```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```


Server will be available at: `http://localhost:8000`

### Performance Note

If you're using **UV** for package management (recommended), dependency installation is **10-100x faster**:
- First install: ~12 seconds (vs ~120 seconds with pip)
- Cached install: ~2 seconds (vs ~90 seconds with pip)


### API Documentation

Interactive API docs: `http://localhost:8000/docs`

## 📡 API Usage

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. Build User Profile

```bash
curl -X POST http://localhost:8000/profile \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Ritika",
    "birth_date": "1995-08-20",
    "birth_time": "14:30",
    "birth_place": "Jaipur, India",
    "preferred_language": "en"
  }'
```

### 3. Chat (Main Endpoint)

**English Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123",
    "message": "How will my month be in career?",
    "user_profile": {
      "name": "Ritika",
      "birth_date": "1995-08-20",
      "birth_time": "14:30",
      "birth_place": "Jaipur, India",
      "preferred_language": "en"
    }
  }'
```

**Hindi Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123",
    "message": "मेरे करियर में यह महीना कैसा रहेगा?",
    "user_profile": {
      "name": "Ritika",
      "birth_date": "1995-08-20",
      "birth_time": "14:30",
      "birth_place": "Jaipur, India",
      "preferred_language": "hi"
    }
  }'
```

**Response:**
```json
{
  "response": "आपके लिए यह महीना अवसर लेकर आ रहा है...",
  "context_used": ["career_transits", "leo_traits"],
  "zodiac": "Leo",
  "confidence_score": 0.85,
  "error": null
}
```

### 4. Session Management

**Get Session:**
```bash
curl http://localhost:8000/session/user-123
```

**Clear Session History:**
```bash
curl -X POST http://localhost:8000/session/user-123/clear
```

**Delete Session:**
```bash
curl -X DELETE http://localhost:8000/session/user-123
```


## 🔧 Component Details

### 1. LLM Wrapper (`app/llm_wrapper.py`)
- Abstraction over OpenAI and Ollama models
- Supports model fallback
- Async operations support

```python
from app.llm_wrapper import create_llm

# OpenAI
llm = create_llm(provider="openai", model_name="gpt-3.5-turbo")

# Ollama
llm = create_llm(provider="ollama", model_name="llama3.1")
```

### 2. Retrieval Engine (`app/retrieval.py`)
- FAISS and ChromaDB support
- Sentence transformers for embeddings
- Metadata filtering

```python
from app.retrieval import RetrievalEngine

engine = RetrievalEngine(
    data_dir="./data",
    vector_store_type="faiss"
)
engine.build_vector_store()

# Retrieve context
results = engine.retrieve("career advice for Leo", k=3)
```

### 3. LangGraph Agent (`app/agent.py`)
- Multi-step workflow: retrieve → generate → filter → translate
- State management across conversation
- Confidence scoring

```python
from app.agent import AstroConversationalAgent

agent = AstroConversationalAgent(llm, retrieval, translation)
response = agent.process_query(
    query="How will my career be?",
    user_profile=profile,
    conversation_history=history,
    language="en"
)
```

### 4. Memory Management (`app/memory.py`)
- Session-based conversation tracking
- Automatic cleanup of old sessions
- Profile persistence

```python
from app.memory import SessionManager

manager = SessionManager()
session = manager.get_session("user-123")
session.add_message("user", "Hello")
```

## 📊 Data Format

### Knowledge Base Structure

```
data/
├── vedic_astrology.txt        # Main astrological knowledge
├── planetary_traits.json      # Planet characteristics
└── zodiac_personality.json    # Zodiac sign profiles
```

### Adding Custom Knowledge

1. Add text files to `data/` directory
2. Add JSON files with structured data
3. Rebuild vector store: `engine.build_vector_store()`

## 🌐 Multi-language Support

The system supports **English** and **Hindi** through:

1. **Automatic Translation**: Responses are translated to target language
2. **Multilingual Prompts**: Hindi-specific system prompts
3. **Bidirectional**: Handles both English queries with Hindi responses and vice versa

```python
# Hindi response
response = agent.process_query(
    query="मेरे करियर में...",
    user_profile=profile,
    conversation_history=[],
    language="hi"
)
```

## 🔒 Safety Features

### 1. Hallucination Filtering
- Detects toxic keywords (death, doom, curse)
- Flags overconfident predictions
- Adds disclaimers when needed

### 2. Input Validation
- Birth date format validation
- Required field checking
- Language code validation

### 3. Error Handling
- Graceful degradation
- Fallback responses
- Detailed error messages

## ⚡ Performance Optimization

### 1. Prompt Caching
Enable in LLM wrapper to reduce API calls for repeated system prompts.

### 2. Vector Store Persistence
```python
# Save vector store to disk
engine.build_vector_store()
# Loads from disk on restart (faster)
engine.load_vector_store()
```

### 3. Session Cleanup
Automatic cleanup of old sessions to manage memory.

## 🎯 Example Queries

### Career Questions
- "What should I focus on in my career this month?"
- "Which planet is affecting my professional life?"
- "When is a good time to change jobs?"

### Love & Relationships
- "Which planet is affecting my love life?"
- "How can I improve my relationships?"
- "What does my Venus placement mean?"

### General Guidance
- "Why is today stressful for me?"
- "What are my strengths according to my chart?"
- "How can I use astrology for personal growth?"

