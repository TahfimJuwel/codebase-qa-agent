🤖 Autonomous Codebase Intelligence Q&A System

An agentic codebase question-answering system built with LangGraph, Qdrant, and Sentence-Transformers.

The system performs syntax-preserving AST-based code extraction, dense semantic vector retrieval, and autonomous multi-tool reasoning over real-world codebases such as psf/requests.

It delivers context-aware answers with exact file paths and line-number citations, targeting sub-3-second query responses.

🎯 Target Repository
Property	Details
Target Repository	psf/requests
Repository Language	Python
Primary Framework	LangGraph
Vector Database	Qdrant
Embedding Model	sentence-transformers/all-MiniLM-L6-v2
Embedding Dimensions	384
Primary Focus	Codebase Intelligence & Semantic Retrieval
Areas of Focus
Connection pooling and connection management
Transport Adapters (HTTPAdapter)
SSL/TLS verification hierarchies
Session configurations
HTTP request and response pipelines
🏗️ Architecture & System Flow

The system uses a deterministic LangGraph state machine to orchestrate autonomous code retrieval, evidence collection, and response synthesis.

text
                 ┌──────────────────────────────┐
                 │      User Natural Query      │
                 └──────────────┬───────────────┘
                                │
                                ▼
          ┌──────────────────────────────────────────┐
          │       LangGraph Custom State Machine     │
          │                                          │
          │  • Persistent In-Memory State            │
          │  • Multi-Turn Conversational Memory      │
          │  • MemorySaver Checkpointer              │
          └──────────────────┬───────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
       ┌──────────────────┐     ┌──────────────────┐
       │ Technical Query  │     │ Out-of-Scope     │
       │                  │     │ Query            │
       └────────┬─────────┘     └────────┬─────────┘
                │                        │
                ▼                        ▼
       ┌──────────────────┐     ┌──────────────────┐
       │ Autonomous       │     │ Immediate Safe   │
       │ Query Router     │     │ Refusal          │
       └────────┬─────────┘     └──────────────────┘
                │
                ▼
       ┌─────────────────────────────────────────┐
       │         Autonomous Tool Selection       │
       └────────────────────┬────────────────────┘
                            │
       ┌────────────────────┼──────────────────────┐
       │                    │                      │
       ▼                    ▼                      ▼
┌──────────────┐    ┌──────────────┐      ┌────────────────┐
│ search_code  │    │ read_file    │      │ list_directory │
└──────┬───────┘    └──────┬───────┘      └───────┬────────┘
       │                   │                      │
       ▼                   ▼                      ▼
┌─────────────────────────────────────────────────────────┐
│             Cloned Target Repository                    │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Local Qdrant Index                    │  │
│  │                                                   │  │
│  │  • AST-Based Code Chunks                           │  │
│  │  • Dense Semantic Embeddings                       │  │
│  │  • File Path Metadata                              │  │
│  │  • Exact Source Line Numbers                       │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  summarize_module    │
                │  & Evidence          │
                │  Collection          │
                └──────────┬───────────┘
                           │
                           ▼
       ┌─────────────────────────────────────────┐
       │       Deterministic Synthesis Node      │
       │                                         │
       │  • Validates Retrieved Code Evidence    │
       │  • Generates Context-Aware Answers     │
       │  • Injects Exact File/Line Citations   │
       └────────────────────┬────────────────────┘
                            │
                            ▼
       ┌─────────────────────────────────────────┐
       │          Formatted CLI Output           │
       │                                         │
       │  Answers + Code References + Citations │
       └─────────────────────────────────────────┘

Execution Graph

The agent follows a controlled execution pipeline:

text
Agent (Autonomous Tool Choice)
             |
             v
         Tool Node
             |
             v
      Synthesize Node
             |
             v
            END


This architecture prevents uncontrolled tool-calling loops and ensures predictable execution.

⚙️ Key Technical Features
1. AST-Aware Structural Code Chunking

Unlike traditional character-based or token-based splitting, the system uses Python's native ast module to preserve the logical structure of source code.

Implementation: src/parser.py

Key capabilities:

Extracts complete ClassDef and FunctionDef AST nodes.
Preserves original source line numbers using node.lineno and node.end_lineno.
Includes enclosing class context when extracting methods.
Prevents functions and classes from being arbitrarily split across chunks.
Maintains source-level traceability for accurate citations.
2. Embedded Vector Database & Local Embeddings

The system uses Qdrant in embedded local-storage mode.

Vector Database: ./data/vector_db

Embedding Model: sentence-transformers/all-MiniLM-L6-v2

Key capabilities:

Generates dense semantic embeddings locally.
Uses 384-dimensional embedding vectors.
Stores AST chunks and associated metadata persistently.
Supports semantic similarity-based code retrieval.
Avoids repeated indexing and embedding generation when an existing valid index is available.

Once indexed, the vector database persists on disk, eliminating unnecessary re-indexing during subsequent application startups.

3. Deterministic LangGraph Execution

The system uses a custom LangGraph state machine instead of an unrestricted ReAct execution loop.

Key capabilities:

Autonomous model-driven tool selection.
Dedicated tool execution nodes.
Controlled state transitions.
Multi-turn conversational memory using MemorySaver.
Deterministic synthesis and termination.

The execution graph prevents infinite tool-calling loops and reduces unnecessary API requests.

4. Multi-Tool Autonomous Reasoning

The agent has access to specialized tools for exploring and understanding the repository.

Tool	Responsibility
search_code	Semantic search across indexed code chunks
read_file	Read source files for deeper inspection
list_directory	Explore repository structure
summarize_module	Generate module-level explanations

The model autonomously selects the appropriate tool based on the user's technical question.

5. Anti-Hallucination & Exact Code Citations

The system prioritizes retrieved source-code evidence over unsupported model-generated assumptions.

Key capabilities:

Rejects out-of-scope queries such as blockchain and smart contracts.
Validates retrieved code evidence before synthesis.
Uses exact source file paths and line ranges.
Grounds technical explanations in actual repository implementations.

Citation format:

text
[file_path:start_line-end_line]


Example:

text
[src/requests/adapters.py:307-363]

6. Performance-Oriented Design

The architecture is designed for fast, efficient codebase question answering.

Metric	Target / Configuration
Query Response Time	< 3 seconds
Repository	psf/requests
Files Parsed	81
AST Chunks Generated	786
Embedding Dimensions	384
Vector Storage	Local Qdrant
Embedding Generation	Local
Execution Framework	LangGraph

Performance figures represent the intended/project configuration and may vary depending on hardware, model API latency, and repository size.

🛠️ Technology Stack
Technology	Purpose
Python 3.10+	Core programming language
LangGraph	Agent orchestration and state management
LangChain	LLM and tool integration
Qdrant	Semantic vector database
Sentence-Transformers	Local embedding generation
all-MiniLM-L6-v2	Dense semantic embedding model
Groq API	LLM inference
Python AST	Structural source-code parsing
MemorySaver	Conversational state persistence
📂 Complete Project Structure
text
codebase-qa-agent/
│
├── .env                  # Holds GROQ_API_KEY (never pushed to Git)
├── .gitignore            # Keeps data/, vector_db/, and venv/ out of version control
├── requirements.txt      # Pinned dependency manifest
├── main.py               # CLI entrypoint; renders UI & multi-turn streaming loop
│
└── src/
    ├── config.py         # Central paths, model parameters, and target repo configuration
    ├── ingest.py         # Git shallow cloner & null-byte binary filtering scanner
    ├── parser.py         # AST-based syntax tree code parser (functions, classes, lines)
    ├── vectorstore.py    # Local Qdrant embedded client & SentenceTransformer embedder
    ├── tools.py          # 4 safe tools (search_code, read_file, list_directory, summarize_module)
    └── agent.py          # LangGraph custom state machine with memory & synthesis

🚀 Setup & Installation
1. Prerequisites

Make sure you have the following installed:

Python 3.10 or higher
Git
An API key from the Groq Console
2. Clone the Repository
bash
git clone <YOUR_PRIVATE_REPO_URL>

cd codebase-qa-agent

3. Create and Activate Virtual Environment

Windows (PowerShell):

powershell
python -m venv venv

.\venv\Scripts\Activate.ps1


macOS / Linux:

bash
python -m venv venv

source venv/bin/activate

4. Install Dependencies
bash
pip install -r requirements.txt

5. Configure Environment Variables

Create a .env file in the project's root directory.

env
GROQ_API_KEY="gsk_your_groq_api_key_here"


Replace the placeholder with your actual Groq API key.

Important: Never commit your .env file or expose your API key publicly.

6. One-Time Codebase Indexing

Run the following command to clone, parse, embed, and index the target repository into the local Qdrant database:

bash
python -m src.vectorstore


The initial indexing process takes approximately 30 seconds under the intended setup.

Expected indexing statistics:

text
Repository Files Parsed : 81
AST Chunks Generated   : 786
Embedding Dimensions   : 384
Vector Database        : ./data/vector_db


Once indexing is complete, the vector database is persisted locally.

You do not need to repeat the indexing process every time you launch the application, provided the existing index remains valid.

💻 Usage

After completing the installation and indexing steps, launch the conversational terminal interface:

bash
python main.py


You can now ask natural-language questions about the target codebase.

Example Questions

Connection Pooling

text
How does the Requests library manage HTTP connection pooling?


Transport Adapters

text
Explain how HTTPAdapter configures connection pools.


SSL/TLS Verification

text
How does Requests handle SSL certificate verification?


Session Configuration

text
What happens when verify=False is passed to a session?


Request Pipeline

text
Explain the complete lifecycle of a request from Session.request() to the adapter.


Code Exploration

text
Which methods are responsible for configuring retry behavior?


The agent retrieves relevant source-code evidence and generates answers with exact file and line-number references.

🛡️ Rules, Integrity & AI Disclosure

As required by the assessment guidelines:

AI Tool Usage:

An AI coding assistant (Claude / ChatGPT) was utilized as an interactive co-pilot during development for debugging grpc model endpoint deprecations and designing LangGraph state machine edge topologies.

All code chunking logic (src/parser.py), path traversal security wrappers (src/tools.py), and LangGraph cycle orchestration (src/agent.py) were written, reviewed, and verified manually.

🔒 Safety & Scope Control

The system is designed specifically for technical codebase question answering.

Only relevant repository-related questions are processed.
Out-of-scope queries are rejected before unnecessary tool execution.
Repository access is restricted to the cloned target codebase.
Retrieved source code is used as evidence for answer generation.
API credentials should be stored in environment variables.
📜 License

Add your preferred open-source license here (for example, MIT) if you intend to distribute this project publicly.

Built with Python, LangGraph, Qdrant, and Sentence-Transformers. 🚀
