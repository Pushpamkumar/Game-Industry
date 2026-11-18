# UdaPlay: Autonomous Video Game Research & Intelligence Agent

An advanced, stateful Agentic AI system engineered to answer complex video game queries through a two-tier information retrieval architecture: primary Retrieval-Augmented Generation (RAG) over a local vector database, coupled with dynamic quality evaluation and automated web search fallback via the Tavily Search API.

---

## 1. Executive Summary & Problem Scenario

In the gaming analytics and media industry, analysts, executives, and gamers require reliable, immediate answers to multi-faceted questions spanning decades of gaming history—from launch platforms and release dates to studio developments and current industry projects.

Standard Large Language Models (LLMs) suffer from:
1. **Knowledge Cutoffs & Hallucinations:** Inability to reliably recall historical release dates or platform details without grounding.
2. **Lack of Dynamic Awareness:** Inability to separate known catalog data from live web developments.
3. **Rigid Single-Turn Interactions:** Inability to retain context during complex exploratory dialogues.

**UdaPlay** solves this by implementing an autonomous, state-machine-driven research agent featuring:
- **Tier-1 Local RAG:** High-speed semantic similarity search over an embedded vector database of video games using **ChromaDB** and OpenAI embeddings.
- **Verification Evaluation Gate:** A dedicated LLM evaluation layer that assesses whether retrieved internal context satisfies the user inquiry before any response is generated.
- **Tier-2 Live Web Fallback:** Automated fallback to the **Tavily Search API** when internal knowledge is evaluated as incomplete, ambiguous, or absent.
- **Stateful Conversation Memory:** Persistent short-term memory allowing seamless multi-turn follow-ups and anaphoric resolution within isolated session contexts.
- **Decoupled Architecture & MCP Support:** Full implementation in interactive Jupyter notebooks and an extensible Model Context Protocol (MCP) client-server variant.

---

## 2. System Architecture

The agent operates as a cyclical state machine. The state machine coordinates message preparation, LLM reasoning, structured tool invocation, quality evaluation, and final answer generation.

### 2.1 Information Retrieval & Verification Flow

```mermaid
flowchart TD
    A([User Query]) --> B[Agent State Machine: message_prep]
    B --> C[LLM Processor: gpt-4o]
    C --> D{Tool Call Generated?}
    
    D -- "retrieve_game" --> E[(Local ChromaDB: 'udaplay')]
    E --> F[Tool Result Appended to State]
    F --> C
    
    D -- "evaluate_retrieval" --> G{Is Retrieval Sufficient?}
    G -- "Yes (useful=True)" --> H[Synthesize Grounded Response]
    G -- "No (useful=False)" --> I[Dispatch web_search Tool]
    
    I --> J[Tavily Search API]
    J --> K[External Web Context Appended]
    K --> C
    
    D -- "No (Final Output)" --> L([Formatted Answer with Citations])
```

### 2.2 Core State Machine Workflow

The agent lifecycle is governed by `StateMachine[AgentState]` (`lib/agents.py`):
1. **`EntryPoint`:** Receives the incoming user query and binds to the specified `session_id`.
2. **`message_prep`:** Retrieves historical conversation context for the session from memory, binds system instructions, and appends the new user query.
3. **`llm_processor`:** Dispatches the message payload to `gpt-4o` with registered tool definitions. Captures token usage and parses tool call intents.
4. **`tool_executor`:** Dispatches matching tools (`retrieve_game`, `retrieve_all_games`, `evaluate_retrieval`, `web_search`), encapsulates outputs into `ToolMessage` instances, and routes execution back to `llm_processor`.
5. **`Termination`:** Reached when the LLM produces a terminal textual response without pending tool calls. The final state is committed to memory and returned.

---

## 3. Two-Tier Retrieval & Verification Mechanism

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Agent as UdaPlay Agent
    participant DB as ChromaDB (Local)
    participant Eval as Evaluator Gate
    participant Web as Tavily Web Search
    
    User->>Agent: Query (e.g. "Was Mortal Kombat X released for PS5?")
    Agent->>DB: retrieve_game(query="Mortal Kombat X PS5", num_matches=5)
    DB-->>Agent: [Local matching documents or empty]
    Agent->>Eval: evaluate_retrieval(query, retrieved_docs)
    Eval-->>Agent: useful=False ("Local catalog lacks PS5 details for MKX")
    Note over Agent: Local knowledge insufficient -> Trigger Tier-2 Fallback
    Agent->>Web: web_search(query="Mortal Kombat X release PlayStation 5")
    Web-->>Agent: [Tavily extracted answer, URLs, snippets]
    Agent->>User: Final Answer with structured points & source citations
```

### 3.1 Tier-1: Local Semantic Search (ChromaDB)
- **Path & Storage:** Persistent client initialized at `./chromadb` hosting the collection `udaplay`.
- **Embedding Pipeline:** Utilizes OpenAI's embedding API (`OPENAI_BASE_URL` and `OPENAI_API_KEY`).
- **Data Models:** Strongly typed via Pydantic (`GameDocument`), guaranteeing consistent attributes (`Platform`, `Name`, `YearOfRelease`, `Description`).

### 3.2 Verification Gate (`evaluate_retrieval`)
Rather than blindly relying on vector distance thresholds, UdaPlay utilizes an LLM evaluation gate with a strict system instruction:
- Takes the original question and retrieved candidates as inputs.
- Parses output into a structured Pydantic model (`RetrievalEvaluation` with `useful: bool` and `description: str`).
- If `useful=False`, the agent recognizes the knowledge gap and proactively queries the web fallback.

### 3.3 Tier-2: Live Web Fallback (`web_search`)
- Integrated with the **Tavily Search API**, optimized specifically for factual search workflows.
- Extracts contextual answer snippets, source URLs, and relevance metrics.
- Mandates source URL attribution in the final output.

---

## 4. Project Structure & Codebase Map

```
.
├── Udaplay_01_starter_project.ipynb   # Part 1: Vector database initialization & RAG data prep
├── Udaplay_02_starter_project.ipynb   # Part 2: Stateful agent implementation, tools & benchmark queries
├── games/                             # Curated video game knowledge catalog (48 JSON entries)
│   ├── 001.json                       # Gran Turismo (PS1, 1997)
│   ├── 002.json                       # Grand Theft Auto: San Andreas (PS2, 2004)
│   ├── ...                            # Curated retro, modern, and next-gen titles
│   └── 048.json                       # GTA 6 (PS5, 2025)
├── lib/                               # Reusable orchestration & foundational modules
│   ├── agents.py                      # Agent orchestrator, AgentState schema, and workflow builder
│   ├── state_machine.py               # Deterministic finite state machine framework
│   ├── llm.py                         # OpenAI-compatible API wrapper supporting function calling
│   ├── messages.py                    # Typed message taxonomy (System, User, AI, Tool)
│   ├── tooling.py                     # Tool abstractions, metadata serialization, and @tool decorator
│   ├── vector_db.py                   # ChromaDB VectorStore abstraction and collection manager
│   ├── memory.py                      # Session-based short-term conversation memory
│   └── documents.py                   # Document and Corpus data structures
├── mcp_server.py                      # Model Context Protocol (MCP) server exposing tools over stdio
├── udaplay_mcp_client_agent.py        # MCP research client with OpenAI function-calling loop
├── docs/images/                       # Architectural diagrams and execution artifacts
├── requirements.txt                   # Complete locked dependencies for reproducibility
├── .env.example                       # Environment credential template
└── README.md                          # Comprehensive system documentation & evaluation report
```

---

## 5. Environment Setup & Installation

### 5.1 Prerequisites
- **Python:** 3.11 or later
- **Operating System:** Windows, macOS, or Linux
- **API Credentials:**
  - OpenAI-compatible API key & Base URL (e.g., Udacity Vocareum or standard OpenAI)
  - Tavily Search API key (Free tier available at [tavily.com](https://app.tavily.com/))

### 5.2 Virtual Environment & Dependencies

Using standard `pip`:
```bash
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Alternatively, using `uv`:
```bash
uv venv
.\venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### 5.3 Configuration (`.env`)

Create a `.env` file in the project root based on `.env.example`:

```env
# OpenAI API Configuration (Vocareum or OpenAI)
OPENAI_API_KEY="voc-************************"
OPENAI_BASE_URL="https://openai.vocareum.com/v1"

# Tavily Search API Configuration
TAVILY_API_KEY="tvly-************************"
```

---

## 6. Execution Guide

### 6.1 Phase 1: Local Vector Database Ingestion (Notebook 01)
Open and run all cells in `Udaplay_01_starter_project.ipynb`:
1. Validates environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`).
2. Initializes persistent storage in `./chromadb`.
3. Processes and embeds all 48 video game records from `games/` into the `udaplay` collection.
4. Validates single-record lookups and performs cross-tabulated generation distribution analytics across platforms.

### 6.2 Phase 2: Agent Execution & Benchmark Queries (Notebook 02)
Open and run `Udaplay_02_starter_project.ipynb`:
1. Binds the three core tools: `retrieve_game`, `evaluate_retrieval`, and `web_search`.
2. Instantiates `Agent(model_name="gpt-4o", tools=tools, instructions=instructions)`.
3. Demonstrates autonomous query processing across benchmark scenarios:
   - **Internal RAG Queries:** Direct resolution from ChromaDB without external web traffic.
   - **External Web Fallbacks:** Knowledge gap detection, quality evaluation, and web search execution.
   - **Multi-Turn Session Continuity:** Conversational context tracking across sequential user turns.

### 6.3 Bonus Phase: Model Context Protocol (MCP) Interface
UdaPlay includes a decoupled MCP implementation demonstrating tool exposure across independent processes via standard I/O:

1. **Start the MCP Client Agent:**
   ```powershell
   python udaplay_mcp_client_agent.py
   ```
   *The client automatically spawns `mcp_server.py` over stdio, queries available tools, registers them with the OpenAI client, and starts an interactive prompt.*

2. **Interactive Testing:**
   ```text
   UdaPlay > When was Pokémon Gold and Silver released?
   >> Dispatching MCP Tool: retrieve_game | Args: {'query': 'Pokémon Gold and Silver release date'}
   >> Dispatching MCP Tool: evaluate_retrieval | Args: {...}
   
   Final Answer:
   Pokémon Gold and Silver was released in 1999 for the Game Boy Color...
   ```

---

## 7. Benchmark Query Demonstrations

Below are key verification trajectories captured during execution:

### Scenario A: Local Vector Database Hit (Internal RAG)
- **User Query:** *"When was Pokémon Gold and Silver released?"*
- **Trajectory:**
  1. `assistant` calls `retrieve_game(query="Pokémon Gold and Silver release date")`.
  2. `tool` returns `GameDocument(Platform='Game Boy Color', Name='Pokémon Gold and Silver', YearOfRelease=1999, ...)`.
  3. `assistant` calls `evaluate_retrieval(question=..., retrieved_docs=...)`.
  4. `tool` returns `useful=True` (*"Document provides release year 1999 for Pokémon Gold and Silver on Game Boy Color"*).
  5. `assistant` synthesizes grounded answer directly from internal data.
- **Citation:** Local ChromaDB Knowledge Base (`games/006.json`).

### Scenario B: Semantic Reasoning on Franchise History
- **User Query:** *"Which one was the first 3D platformer Mario game?"*
- **Trajectory:**
  1. `assistant` calls `retrieve_game(query="first 3D platformer Mario game")`.
  2. `tool` retrieves `Super Mario 64` (1996, N64) and `Super Mario World` (1990, SNES).
  3. `assistant` calls `evaluate_retrieval`.
  4. `tool` evaluates and confirms `Super Mario 64` satisfies the 3D platformer criteria.
  5. `assistant` delivers structured summary detailing launch year (1996) and Nintendo 64 platform.

### Scenario C: Web Search Fallback (Missing Internal Knowledge)
- **User Query:** *"Was Mortal Kombat X released for PlayStation 5?"*
- **Trajectory:**
  1. `assistant` calls `retrieve_game(query="Mortal Kombat X PlayStation 5")`.
  2. Local DB yields no matching records for Mortal Kombat X.
  3. `assistant` calls `evaluate_retrieval`.
  4. `tool` returns `useful=False` (*"None of the retrieved records mention Mortal Kombat X or its availability on PlayStation 5"*).
  5. `assistant` triggers `web_search(query="Mortal Kombat X release for PlayStation 5")`.
  6. `tool` returns Tavily search payload including official PlayStation Store links and release history.
  7. `assistant` synthesizes a comprehensive final answer clarifying that Mortal Kombat X launched on PS4 in April 2015 and is playable on PS5 via backward compatibility.
- **Citation:** Explicit URL citation (`https://store.playstation.com/...`, `https://en.wikipedia.org/...`).

### Scenario D: Multi-Turn Conversation & Anaphoric Resolution
- **Turn 1:** *"Which game was published in 2017 on Nintendo Switch?"*
  - Local retrieval discovers `Mario Kart 8 Deluxe`.
- **Turn 2:** *"What is the name of this game?"*
  - Agent leverages session context (`session_id="conversation"`) to resolve *"this game"* to `Mario Kart 8 Deluxe`.
- **Turn 3:** *"Can you tell me its publisher?"*
  - The local database summary lacks the publisher field for this entry; the evaluation gate identifies the gap and triggers `web_search` to confirm Nintendo as the publisher.

---

## 8. Visual Execution Artifacts & Execution Logs

The project repository includes captured execution snapshots under `docs/images/` demonstrating each key milestone:

### 1. Vector Database Construction & Ingestion
Verifying the ingestion of the 48 video game records into persistent ChromaDB storage:
![Vector Database Ingestion](docs/images/01_build_vector_db.png)

### 2. State Machine Pipeline
Visualization of the cyclical orchestrator coordinating transitions between prompt preparation, LLM processing, tool dispatch, and termination:
![State Machine Workflow](docs/images/02_state_machine.png)

### 3. Tool Dispatch (`retrieve_game`)
Real-time execution log showing assistant tool call generation and structured document retrieval:
![Tool Dispatch Trace](docs/images/03_agent_tool_call.png)

### 4. Retrieval Verification Gate (`evaluate_retrieval`)
Quality evaluation assessment output confirming factual sufficiency before final response generation:
![Evaluation Gate Trace](docs/images/04_evaluation_call.png)

### 5. Final Grounded Response Synthesis
End-to-end execution showing structured final answer with highlighted formatting and source attribution:
![Final Answer Output](docs/images/05_final_answer.png)

---

## 9. Stand Out Engineering Highlights

This implementation incorporates several advanced capabilities that go beyond standard project specifications:

1. **Enriched Knowledge Base (48 Curated Games):**
   Extended the foundational 15-game catalog to 48 richly documented games spanning 1990 through 2025 across 12 distinct platforms (SNES, N64, Game Boy Color, Game Boy Advance, GameCube, Wii, Switch, PS1–PS5, Xbox 360, Xbox One, Xbox Series X/S, and PC).
2. **Quality Evaluation Gatekeeper:**
   Rather than performing blind fallback or hallucinating when vector distance scores are borderline, the agent utilizes a dedicated evaluation prompt with structured Pydantic schema enforcement (`RetrievalEvaluation`).
3. **Session-Isolated Conversation Memory:**
   Full session isolation with state roll-forward via `ShortTermMemory`, enabling multiple concurrent dialogue threads without cross-contamination.
4. **Dual Interface Options:**
   Full agentic implementation in standard Jupyter notebooks alongside a modern Model Context Protocol (MCP) client-server architecture with stdio transport.
5. **Structured Output Formatting:**
   Clean terminal rendering using `rich` panels, ANSI syntax coloring, Markdown conversion, and Pydantic V2 model validation throughout.

---

## 10. Troubleshooting & FAQ

| Issue | Root Cause | Resolution |
| :--- | :--- | :--- |
| `OPENAI_API_KEY is not set` | Missing or misnamed `.env` file | Ensure `.env` exists in project root with valid `OPENAI_API_KEY` and `OPENAI_BASE_URL`. |
| `Collection 'udaplay' does not exist` | Notebook 02 executed before Notebook 01 | Run all cells in `Udaplay_01_starter_project.ipynb` first to create and populate the ChromaDB directory. |
| `sqlite3 version mismatch` | Older Linux/cloud environment sqlite3 library | The codebase includes an automatic `pysqlite3` shim that redirects imports when available. |
| `Tavily API 401 / Unauthorized` | Expired or incorrect Tavily key | Generate a free API key at [tavily.com](https://app.tavily.com/) and update `TAVILY_API_KEY` in `.env`. |
| ChromaDB telemetry warning | Normal benign notification in ChromaDB 1.0+ | Telemetry warnings (`capture() takes 1 positional argument...`) are harmless and do not impact execution. |

---

## 11. Verification Checklist Before Submission

- [x] Vector store created at `./chromadb` with collection name `udaplay`.
- [x] Both notebooks (`Udaplay_01_starter_project.ipynb` and `Udaplay_02_starter_project.ipynb`) contain complete executed cell outputs.
- [x] Three core tools (`retrieve_game`, `evaluate_retrieval`, `web_search`) implemented and tested.
- [x] Evaluation gate reliably triggers web fallback when internal knowledge is lacking.
- [x] Stateful multi-turn conversation verified across repeated queries.
- [x] Web citations include source URLs.
- [x] Code is clean, well-typed, and completely free of plagiarism markers or hardcoded credentials.
- [x] `.env.example` provided for safe reproduction.
