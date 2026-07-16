# Morocco Tourism Agent

An agentic retrieval-augmented generation (RAG) tourism assistant for Morocco.
The project uses LangChain, LangGraph, Google Gemini, and a persistent ChromaDB
vector store.

The current retrieval flow is:

```text
PDF loading -> chunk splitting -> Gemini embeddings -> ChromaDB -> retrieval
```

The first agent workflow is assembled manually with LangGraph `StateGraph`:

```text
Question -> intent classification -> retrieval -> answer generation
         -> validation -> final grounded response
```

## Project structure

```text
MoroccoTourismAgent/
|-- app/
|   |-- graph/       # LangGraph state, nodes, edges, and workflow
|   |-- rag/         # Document ingestion and retrieval pipeline
|   |-- tools/       # Tools exposed to the agent
|   |-- llm/         # Gemini chat-model configuration
|   |-- memory/      # Conversation memory
|   |-- evaluation/  # Evaluation questions and utilities
|   `-- utils/       # Shared configuration and logging helpers
|-- data/
|   |-- documents/   # Source tourism PDFs
|   `-- vectordb/    # Local persistent ChromaDB data
|-- scripts/         # Indexing and retrieval smoke-test commands
|-- tests/           # Automated tests
|-- requirements.txt
|-- pyproject.toml
`-- .env.example
```

## Setup

### 1. Create and activate a virtual environment

Python 3.11 or newer is required.

Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

Install the complete project environment:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Gemini LangChain integration can also be installed or upgraded directly:

```powershell
pip install -U langchain-google-genai
```

### 3. Create and configure a Gemini API key

1. Sign in to [Google AI Studio](https://aistudio.google.com/apikey).
2. Select **Create API key** and copy the generated key.
3. Copy the environment template in the project root:

   ```powershell
   Copy-Item .env.example .env
   ```

4. Set the key in `.env` without quotes or extra spaces:

   ```dotenv
   GOOGLE_API_KEY=your_api_key_here
   GEMINI_CHAT_MODEL=gemini-3.5-flash
   ENABLE_ANSWER_VALIDATION=true
   ```

Never commit `.env` or paste its contents into logs. The application explicitly
loads this root file with `python-dotenv`.

Verify that Python can detect the variable without printing its value:

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(bool(os.getenv('GOOGLE_API_KEY')))"
```

The command should print `True`. A missing key produces a clear configuration
error when a Gemini model is created.

## Index and retrieve tourism documents

Place PDF files in `data/documents/` before indexing.

### Remove an index created with the previous embedding provider

Embedding providers produce incompatible vector representations. Before
re-indexing after the switch from OpenAI to Gemini, remove the old incomplete
Chroma database manually and recreate its directory:

```powershell
Remove-Item -Recurse -Force data\vectordb
New-Item -ItemType Directory data\vectordb
```

The Python application never deletes this directory automatically.

### Build the Gemini-backed index

```powershell
python -m scripts.index_documents
```

This loads the PDFs, splits their pages into chunks, generates embeddings with
`models/gemini-embedding-001`, and stores them in the existing
`morocco_tourism` Chroma collection.

### Test semantic retrieval

```powershell
python -m scripts.test_retrieval
```

The retrieval smoke test reopens the persistent collection and prints relevant
document previews for a sample Marrakech query.

### Test the LangGraph workflow

```powershell
python -m scripts.test_graph
```

This runs factual and itinerary questions through classification, retrieval,
grounded Gemini generation, validation, and at most one revision.

Run one question or customize the default 10-second pause between suite cases:

```powershell
python -m scripts.test_graph --question "Plan a two-day trip to Chefchaouen."
python -m scripts.test_graph --pause-seconds 15
```

Temporary Gemini `503`, `UNAVAILABLE`, and high-demand responses are retried
within a maximum of three attempts. Invalid requests, authentication failures,
missing models, quota errors, and missing configuration are not retried.

### Test conversation memory

```powershell
python -m scripts.test_memory
```

The memory smoke test uses LangGraph `MemorySaver` to retain accepted messages
and simple travel preferences within one `thread_id`, then confirms that a
different thread cannot access them. Memory lasts for the lifetime of the
compiled graph instance and is not persisted across process restarts.

## Development checks

Compile the application and scripts without making an API request:

```powershell
python -m compileall app scripts
```

The reusable Gemini chat model is configured in `app/llm/model.py` with
temperature `0.2`. Set `GEMINI_CHAT_MODEL` in `.env` to choose the model; it
defaults to the stable `gemini-3.5-flash` model.

Set `ENABLE_ANSWER_VALIDATION=false` during quota-constrained development to
skip the Gemini validation request. Deterministic intent classification uses no
Gemini request, so disabled validation reduces normal graph execution to one
generate-content request per question. Set it back to `true` to restore answer
validation and the existing maximum one-revision loop.
