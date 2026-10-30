# 🤖 Autonomous Codebase Intelligence Q&A System

An agentic codebase question-answering system built with **LangGraph**, **Qdrant**, and **Sentence-Transformers**. It performs syntax-preserving AST code extraction, dense semantic vector retrieval, and autonomous multi-tool reasoning over real-world codebases (`psf/requests`), delivering sub-3-second responses with exact file and line-number citations.

---

## 🎯 Target Repository
- **Target Repository:** [`psf/requests`](https://github.com/psf/requests)
- **Repository Language:** Python
- **Key Modules Analyzed:** Transport Adapters (`HTTPAdapter`), connection pooling (`urllib3`), SSL/TLS verification hierarchies, session configurations, and request lifecycles.

---

## 🏗️ Architecture & System Flow

```mermaid
flowchart TD
    User(["👤 User Natural Language Query"]) --> Engine["🧠 LangGraph Custom State Machine\n(In-Memory MemorySaver)"]
    
    Engine --> Check{Query Type?}
    
    Check -- Out of Scope --> Refusal["🛑 Immediate Safe Refusal\n(Zero Token Waste)"]
    Check -- Technical Query --> Router["⚙️ Autonomous Router\n(Model Selects Best Tool)"]
    
    Router --> T1["🔍 search_code"]
    Router --> T2["📄 read_file"]
    Router --> T3["📁 list_directory"]
    Router --> T4["📋 summarize_module"]
    
    T1 --> Qdrant[("💾 Embedded Qdrant Index\nAST Chunks + Exact Line Metadata")]
    T2 --> Disk[("📂 Cloned Target Codebase\nSandboxed Local Repo")]
    T3 --> Disk
    T4 --> Disk
    
    Qdrant --> Synth["📝 Deterministic Synthesis Node\n(Validates Evidence & Injects Citations)"]
    Disk --> Synth
    
    Synth --> Output(["🖥️ Rich CLI Streaming Output\n[file_path:start_line-end_line]"])
