# ✨ Multi Tool Agent

A production-grade AI workstation built with **Streamlit**, **LangGraph**, and **Google Gemini** — engineered for **multi-modal document intelligence**, **automated data analysis**, **SQL exploration**, **image understanding**, and **live web search** — all orchestrated through a stateful tool-routing agent with **optional LangSmith observability**.

This isn't just a "chat with your data" demo. It combines a **LangGraph agent with dynamic tool routing**, **hybrid retrieval (BM25 + FAISS)**, **adaptive query optimization (HyDE, Multi-Query, Decomposition)**, a **self-correcting Python sandbox for data analysis**, **MySQL safety-validated SQL generation**, and **full end-to-end tracing** — wired into one cohesive pipeline.

## ✨ Features

- **🧠 LangGraph Agent with Tool Routing**
  A stateful ReAct-style agent decides which tool to invoke per turn:
  - `document_rag_tool` — PDF / DOCX / TXT / Web URL
  - `pandas_dataframe_tool` — CSV / Excel calculations
  - `mysql_database_tool` — natural-language → safe SQL
  - `vision_reasoning_tool` — image / diagram understanding
  - `web_search_tool` — live DuckDuckGo search (fallback + general knowledge)

- **📄 Smart Document Ingestion**
  High-fidelity structural parsing via [Docling](https://github.com/DS4SD/docling), with automatic fallback to PyPDF / python-docx / plain-text readers for resilience.

- **🔍 Hybrid Retrieval-Augmented Generation (RAG)**
  - Dense vector search via **FAISS** (semantic similarity)
  - Sparse keyword search via **BM25** (exact term matching)
  - Combined through a weighted **Ensemble Retriever**

- **🎯 Adaptive Query Optimization Engine**
  Classifies every query and picks the best strategy:

  | Strategy | When it's used |
  |---|---|
  | **Direct Lookup** | Simple, exact-match questions (IDs, names, specific facts) |
  | **Multi-Query Expansion** | Queries needing broader vocabulary / synonym coverage |
  | **Query Decomposition** | Complex, multi-part or comparative questions |
  | **HyDE Projection** | Open-ended, conceptual questions — generates a hypothetical answer to improve vector alignment |

- **📊 Self-Correcting Data Analysis Agent**
  Upload a CSV/Excel and ask questions in plain English. The system:
  1. Classifies the question as THEORY (schema lookup) or CODE (computation)
  2. Generates Python/pandas code targeting a result variable
  3. Executes it in a **hardened sandbox** (`_SafeProxy` + AST-level escape validation)
  4. If it fails, feeds the error back to the LLM in a **Reflexion loop** to self-correct (up to 3 attempts)

- **🛡️ Hardened Pandas Sandbox**
  - `_SafeProxy` wraps `df`/`pd` to block MRO-based escapes (`df.__class__.__mro__[1].__subclasses__()`)
  - AST-level static validator rejects imports, `eval`/`exec`/`open`/`__import__`, and forbidden dunder attribute access **before** code ever runs
  - Restricted builtin dictionary (no `getattr`, `setattr`, `__import__`)

- **🗄️ MySQL Server Explorer**
  - Connect to any MySQL server with host/port/user/password
  - Auto-discovers databases and tables
  - **Safety-validated SQL**: all generated `db.table` refs are checked against the discovered schema before execution
  - **Aggregation-aware LIMIT injection**: skips auto-appending LIMIT to COUNT/SUM/GROUP BY queries
  - Smart classification: display (table preview), schema, calculation, general (off-topic → returns `NOT_MYSQL_QUERY` so the agent falls back to web search)

- **🖼️ Multi-Modal Vision Engine**
  Images (PNG, JPG, WEBP, **GIF**) are Base64-encoded and routed to **Gemini Vision** for Q&A.

- **🗂️ Multi-Session Workspace**
  Create, switch between, and delete isolated chat sessions — each with its own attached resource, conversation history, and LangGraph workflow instance.

- **🧠 Auto-Summarization**
  When conversation history exceeds 10 messages, older turns are condensed into a rolling summary that re-stamps on every new turn (so older context isn't silently lost).

- **💾 Persistent 7-Day History**
  Atomic JSON-backed storage with thread-safe load+modify+write under a single lock — survives app restarts.

- **🔁 Local-First Embeddings with Cloud Fallback**
  Tries a local **Ollama** embedding model first; if unreachable, seamlessly falls back to **Google Generative AI Embeddings**.

- **🔭 LangSmith Observability (Optional, Opt-In)**
  First-class tracing across every LLM chain, LangGraph node, retriever, SQL generator, and reflexion cycle. See the dedicated [LangSmith Observability](#-langsmith-observability) section below.

## 🏗️ Architecture Overview

```
User Input
│
▼
LangGraph Agent (ReAct-style tool router)
│
├── document_rag_tool ─────► Hybrid Retriever (BM25 + FAISS)
│      │
│      ▼
│      Gemini Synthesis
│
├── pandas_dataframe_tool ─► Sandbox Validator (AST)
│      │
│      ▼
│      Sandbox Exec + Reflexion Loop (up to 3x)
│
├── mysql_database_tool ───► Query Classifier
│      ├── display ──► Direct table fetch
│      ├── schema  ──► Schema dump
│      ├── calc    ──► LLM SQL Gen + Schema Validate + LIMIT
│      └── general ──► returns 'NOT_MYSQL_QUERY'
│
├── vision_reasoning_tool ─► Gemini Vision (multi-modal)
│
└── web_search_tool ───────► DuckDuckGo live results

╔══════════════════════════════════════════════════════╗
║              LangSmith Observability Layer            ║
║  ─────────────────────────────────────────────────── ║
║  • Every LangGraph node, LLM call, and retriever       ║
║  • Hierarchical runs grouped per turn + session        ║
║  • Token usage, latency, cost, and errors captured      ║
║  • Reflexion cycles appear as nested sub-runs           ║
║  • Activated via env vars — fully opt-in                ║
╚══════════════════════════════════════════════════════╝
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI / App Framework | [Streamlit](https://streamlit.io/) |
| Agent Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) + InMemorySaver checkpointer |
| LLM & Vision | Google **Gemini** (gemini-2.5-flash by default) |
| LLM Framework | [LangChain](https://www.langchain.com/) / langchain-classic |
| Document Parsing | [Docling](https://github.com/DS4SD/docling), PyPDF, python-docx (fallbacks) |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) |
| Keyword Search | BM25 (`langchain_community.retrievers.BM25Retriever`) |
| Embeddings | Ollama (nomic-embed-text, local) → Google text-embedding-004 (fallback) |
| Data Analysis | Pandas (sandboxed execution + AST-level validation) |
| Database | MySQL via PyMySQL + SQLAlchemy |
| Web Search | [DuckDuckGo Search](https://duckduckgo.com/) |
| **Observability** | [**LangSmith**](https://smith.langchain.com/) (`langsmith`) |
| Env Management | python-dotenv |

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, create one with at least:

```
streamlit
pandas
requests
python-dotenv
pymysql
sqlalchemy
langchain-core
langchain-community
langchain-classic
langchain-google-genai
langchain-ollama
langchain-text-splitters
faiss-cpu
rank_bm25
langgraph
docling
pypdf
python-docx
openpyxl
xlrd
duckduckgo-search
langsmith
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
# --- Required: Google Gemini API ---
GOOGLE_API_KEY=your_google_api_key_here
CHAT_MODEL=gemini-2.5-flash
VISION_MODEL=gemini-2.5-flash
EMBED_MODEL=nomic-embed-text
GOOGLE_EMBED_MODEL=models/text-embedding-004

# --- Optional: LangSmith Observability ---
# Leave LANGCHAIN_TRACING_V2=false (or unset) to disable tracing entirely.
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=multi-tool-agent
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

*Never commit your `.env` file. Make sure it's listed in `.gitignore`.*

### 5. (Optional) Run Ollama locally for free local embeddings

```bash
ollama pull nomic-embed-text
ollama serve
```

If Ollama isn't running, the app automatically falls back to Google's embedding API.

### 6. Run the app

```bash
streamlit run app.py
```

*(Replace `app.py` with your actual filename if different)*

### 7. (Optional) Run via CLI instead of Streamlit

`cli.py` mirrors `app.py` exactly — same agent, same tools, same routing logic — just without the Streamlit UI, for terminal-based usage or headless environments:

```bash
python cli.py
```

## 🚀 Usage

1. Launch the app and open the sidebar.
2. Choose your attachment method:
   - 📎 **Files** — upload a PDF, DOCX, TXT, CSV, XLSX, or image, or paste a web URL.
   - 🗄️ **MySQL** — enter host/port/user/password and click Connect Server.
3. Click **"Process Attachment"** (files) or **"Connect Server"** (MySQL).
4. Ask questions in the chat box:
   - **Documents:** *"Summarize section 3 of the attached PDF."*
   - **Datasets:** *"What is the average revenue by region?"*
   - **MySQL:** *"Show me the top 10 customers by total spend."* / *"How many orders were placed last month?"*
   - **Images:** *"What does this architecture diagram show?"*
   - **Web:** *"What's the latest news about NVIDIA?"* (the agent auto-routes off-topic MySQL questions to web search)
5. The agent dynamically picks the right tool(s) for each turn — watch the LangGraph orchestrator spin.
6. Create additional workspace sessions from the sidebar to keep separate conversations and resources isolated.
7. If LangSmith is configured, open [smith.langchain.com](https://smith.langchain.com) to inspect the live trace tree of your request.

## 🔭 LangSmith Observability

The workstation is fully wired for LangSmith tracing out of the box. Tracing is opt-in — if you don't set the env vars below, the app runs exactly as before with zero overhead.

### Enabling Tracing

Set the following in your `.env` file:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=multi-tool-agent
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

You can get an API key by creating a free account at [smith.langchain.com](https://smith.langchain.com).

The sidebar of the app shows a live **🟢 LangSmith Connected** / **🔴 LangSmith Disabled** status panel, including the active project name and the reason for any disabled state.

### What Gets Traced

Every user request produces a hierarchical trace tree. Here are two common shapes:

**RAG query against a PDF:**

```
turn_<session>_<timestamp> [tags: multi-tool-agent, turn]
├── mysql_sql_generator (or rag_synthesis) [named LLM run]
└── agent_turn [agent node LLM run]
    └── tools [LangGraph tool dispatch]
```

**Pandas analysis with a reflexion retry:**

```
turn_<session>_<timestamp> [tags: multi-tool-agent, turn]
├── pandas_query_classifier [named LLM run]
├── pandas_code_generator [named LLM run]
├── [sandbox exec] [inline, no LLM cost]
├── pandas_reflexion_cycle_1 [named LLM run] ← if first attempt failed
├── [sandbox exec retry]
└── pandas_reflexion_cycle_2 [named LLM run] ← if second attempt failed
```

### What Data Is Captured

Each trace automatically records:

- **Latency** per run (milliseconds)
- **Token usage** (input, output, total) and estimated cost
- **Full prompts and completions** for every LLM call
- **Inputs and outputs** for retriever calls, SQL generators, and sandbox executions
- **Exceptions and stack traces** when something fails
- **Custom metadata** you can filter by in the dashboard:
  - `session_id` — which Streamlit session made the request
  - `active_file` — which file was attached at request time
  - `user_prompt_preview` — first 120 chars of the user's question

### How to Use It for Debugging

- **Filter by tag** — in the LangSmith UI, filter runs by tags: `multi-tool-agent`, `turn` to isolate single conversation turns.
- **Trace a reflexion loop failure** — open the parent turn and you'll see each `pandas_reflexion_cycle_N` as a child. Click into the failed cycle to see the exact exception, the LLM's "fix" attempt, and the next error.
- **Inspect SQL safety net** — when a MySQL query gets rejected for referencing an unknown table, the trace shows the generated SQL and the validation failure side-by-side.
- **Compare tool routing** — when an off-topic question routes through MySQL → `NOT_MYSQL_QUERY` → web search, the trace tree shows every hop, making it easy to tune the agent's system prompt.
- **Monitor cost** — group runs by `metadata.active_file` to see which resources are driving the most token spend.
- **Spot bad queries** — sort runs by latency or cost to identify user questions that trigger expensive multi-query expansion or repeated reflexion cycles.

### Disabling Tracing

Tracing is automatically disabled if any of the following are true:

- The `langsmith` Python package isn't installed
- `LANGCHAIN_TRACING_V2` is not set to `"true"`
- `LANGCHAIN_API_KEY` is empty or missing
- The LangSmith API endpoint is unreachable (the app will print a warning and continue without tracing)

In every "disabled" case, the app's functionality is completely unaffected — you simply won't see traces in the dashboard.

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application (single-file architecture)
├── cli.py                  # Terminal/CLI entry point — identical to app.py, minus Streamlit
├── requirements.txt        # Python dependencies
├── .env                     # Environment variables (not committed)
├── .gitignore
├── chat_history_db.json    # Auto-generated 7-day rolling chat history
└── README.md
```

## ⚠️ Known Issues / Roadmap

- [x] **Sandbox MRO escape:** `df.__class__.__mro__[1].__subclasses__()` could escape via pandas internals. *(Resolved — `_SafeProxy` blocks dunder access on df/pd.)*
- [x] **Sandbox escape via fresh literals:** `().__class__.__mro__[1].__subclasses__()` could escape because `_SafeProxy` only guards df/pd. *(Resolved — AST-level `validate_sandbox_code()` rejects forbidden dunder/imports/builtins before exec.)*
- [x] **MySQL SQL could reference unknown tables.** *(Resolved — every `db.table` ref is validated against `discovered_databases`.)*
- [x] **MySQL auto-LIMIT broke aggregations.** *(Resolved — `_looks_like_aggregation()` skips LIMIT injection for COUNT/SUM/GROUP BY etc.)*
- [x] **.docx had no fallback when Docling was unavailable.** *(Resolved — python-docx is now a structured-doc fallback.)*
- [x] **LangSmith observability integration.** *(Resolved — see LangSmith Observability.)*
- [ ] Add persistent storage for FAISS indexes (currently in-memory per session).
- [ ] Add unit tests for the retrieval strategy classifier and sandbox validator.
- [ ] Support additional LLM providers (OpenAI, Anthropic) behind a unified interface.
- [ ] Surface LangSmith run URLs directly inside the Streamlit UI for one-click debugging.

## 🤝 Contributing

This is a personal/learning project, but suggestions and PRs are welcome. Feel free to open an issue if you spot a bug or have an idea for improvement.

## 👤 Author

Built by **NEREAN DALMIA** as a hands-on project to learn agentic AI systems, hybrid retrieval, LangGraph orchestration, and LLM observability.

- **GitHub:** NEREAN DALMIA
- **LinkedIn:** NEREAN DALMIA