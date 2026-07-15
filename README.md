# Morocco Tourism Agent

A production-oriented foundation for an agentic retrieval-augmented generation
(RAG) assistant focused on tourism in Morocco. The future assistant will use
LangGraph to coordinate its workflow, LangChain and OpenAI for language-model
integration, and ChromaDB for semantic retrieval.

This repository currently contains project scaffolding only. Agent and RAG
business logic will be implemented in later stages.

## Planned features

- Retrieval over curated Moroccan tourism documents
- Agentic workflow orchestration with LangGraph
- Itinerary generation and destination comparison tools
- Budget-aware travel recommendations
- Conversation memory
- Automated response evaluation
- Streamlit user interface

## Folder structure

```text
MoroccoTourismAgent/
|-- app/
|   |-- graph/       # LangGraph state, nodes, edges, and workflow
|   |-- rag/         # Document ingestion and retrieval pipeline
|   |-- tools/       # Tools exposed to the agent
|   |-- llm/         # Language-model configuration
|   |-- memory/      # Conversation memory
|   |-- evaluation/  # Evaluation questions and utilities
|   `-- utils/       # Shared configuration and logging helpers
|-- data/
|   |-- documents/   # Source tourism documents
|   `-- vectordb/    # Local ChromaDB data (not committed)
|-- tests/           # Automated tests
|-- main.py          # Application entry point
|-- requirements.txt
|-- pyproject.toml
`-- .env.example
```

## Installation

### Prerequisites

- Python 3.11 or newer
- An OpenAI API key for future model integration

### Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Configure environment variables

Copy `.env.example` to `.env`, then add your own API credentials:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux, use `cp .env.example .env` instead. Never commit the
resulting `.env` file.

### Launch the project

Run the placeholder command-line entry point:

```bash
python main.py
```

The Streamlit launch command will be documented when the UI is implemented.

## Index and test tourism documents

Place one or more PDF files in `data/documents/` and configure
`OPENAI_API_KEY` in `.env`. Then create the persistent vector index:

```bash
python -m scripts.index_documents
```

Run the sample Marrakech retrieval query against the completed index:

```bash
python -m scripts.test_retrieval
```

## Future documentation

- Architecture and graph design
- Data ingestion guide
- Evaluation methodology
- Development and testing workflow
- Deployment guide
- Contribution guidelines
