# Chief Scout OS

Chief Scout OS is an advanced, enterprise-grade data engineering and AI-driven football scouting platform. The system orchestrates a robust data collection pipeline, a unified relational database, and an autonomous, stateful multi-tool AI agent to streamline player discovery, statistical comparison, and qualitative profile analysis.

Designed to assist scouting departments, the application acts as an intelligent layer over comprehensive quantitative and qualitative football datasets, allowing users to interact with complex data through standard natural language.

---

## Live Deployment

The platform is fully containerized and hosted in production via Streamlit Community Cloud. You can interact with the live AI scouting agent, issue complex filter queries, and test the dynamic visual analysis loops directly in your browser:

🔗 **Live Production App:** [Chief Scout OS Dashboard](https://smashing-monkeys-oeza6ea4nvkwjc8ktk6len.streamlit.app/)

---

## System Architecture & Data Infrastructure

The platform is divided into three distinct operational layers: the Data Ingestion Pipeline, the Unified Relational Engine, and the Agentic Retrieval Layer.

### 1. Data Ingestion & Transformation Pipeline
The system includes automated web scraping and ingestion scripts configured to harvest, clean, and standardize multi-domain data from premier football data aggregators:
* **FbRef:** Comprehensive performance analytics, including passing networks, progressive metrics, and defensive actions.
* **Understat:** Granular expected metrics ($xG$, $xA$, $xG$ Chain) mapped across match timelines.
* **Capology:** Financial data, salary structures, contract lengths, and estimated transfer market valuations.

### 2. Relational & Vector Storage Layer
* **Quantitative Data (SQLite):** Ingested data is consolidated into a unified master relational database. Schema features are indexed to join attacking profiles, defensive metrics, financial contracts, and **EA Sports FC ratings** for macroscopic sorting.
* **Qualitative Data (Vector Database):** Semi-structured and unstructured scouting text focusing on soft traits (e.g., player temperament, tactical adaptability, and historical injury reports) are embedded and stored within a vector store for semantic retrieval.

---

## Core Scouting Engine Capabilities

The scouting engine provides high-fidelity analysis through a suite of deterministic algorithms exposed to the AI agent via specialized tool interfaces:

### Quantitative Analysis
* **Player Lookups:** Instant extraction of individual historical performance profiles and contract specifics.
* **Algorithmic Filtering & Discovery:** Multi-conditional sorting (e.g., *"Find left-footed center-backs under 23 with >80% progressive pass accuracy making under €50k/week"*).
* **Percentile Engineering:** Automatic computation of player rank benchmarks across specific leagues, positions, and operational tiers to measure statistical dominance.

### Qualitative & Agentic RAG
* **Scouting Report Ingestion:** Semantic vector searching regarding player behavioral traits and physical robustness.
* **Dynamic Web-Search Fallback:** If qualitative data for a targeted player is missing from the vector database, the agent automatically executes an online search, extracts relevant background intel, summarizes the findings, and dynamically populates the vector store for future sessions.

---

## AI Agent Implementation & Performance Tuning

The conversational layer utilizes **LangGraph** and **LangChain** to orchestrate tool selection and maintain multi-turn, stateful chat history across sessions using a robust checkpointer system.

### Advanced Token Optimization (State Scrubbing)
To support real-time user interaction, the platform generates complex data visualizations (such as Matplotlib percentile comparison charts). These charts are encoded into massive base64 strings and written back to the LangGraph thread state. 

To prevent context-window exhaustion and **HTTP 429 Resource Exhausted** errors, the application incorporates a custom middleware interceptor inside the agent loop:

```python
# Technical snippet of the context window safety valve
def state_trimmer_modifier(state) -> list:
    """Intercepts state, strips heavy base64 strings, and passes lean payloads to the LLM."""
    cleaned_messages = []
    for msg in state["messages"]:
        if getattr(msg, "type", "") == "tool" and "image_base64" in str(msg.content):
            # Re-package the message with a lightweight placeholder string
            clean_msg = msg.__class__(
                content=json.dumps({"status": "success", "image_base64": "[SCRUBBED]"}),
                tool_call_id=msg.tool_call_id
            )
            cleaned_messages.append(clean_msg)
        else:
            cleaned_messages.append(msg)
    return [("system", system_prompt)] + trim_messages(cleaned_messages, strategy="last")
```

This ensures that while Streamlit retains and draws the chart historical data natively on screen, the underlying Large Language Model (gemini-3.1-flash-lite) receives only lightweight text data tokens—maintaining speed, keeping context memory perfect, and reducing API costs.

## Installation & Local Environment Setup

Prerequisites
- Python 3.11+
- SQLite3

### 1. Repository Setup

```bash
git clone [https://github.com/marvinwong12/smashing-monkeys.git](https://github.com/marvinwong12/smashing-monkeys.git)
cd smashing-monkeys
```

### 2. Dependency Allocation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environmental Variables
```Ini, TOML
GOOGLE_API_KEY="AIzaSy..."
DATABASE_PATH="data/master_scouting.db"
VECTOR_DB_PATH="data/vector_store/"
```

### 4. Launch Application
Boot the Streamlit frontend cluster:
```bash
streamlit run app.py
```
