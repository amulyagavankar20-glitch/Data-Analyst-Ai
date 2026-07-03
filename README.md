---
title: AI Data Scientist Agent
emoji: 📊
sdk: streamlit
app_file: app.py
pinned: false
license: mit
sdk_version: 1.49.1
---

# 🤖 AI Data Scientist Agent

A multi-agent, agentic AI system that automates exploratory data analysis end-to-end — from raw CSV/XLSX upload to cleaned data, visualizations, and natural-language insights — using an orchestrated **Planner → Coder → Executor → Explainer** pipeline built on LangGraph.

## Why this project

Most "chat with your data" tools stop at Q&A. This agent goes further: it plans a multi-step analysis, writes and executes real Python code against the dataset, **self-corrects when the code fails** (closed-loop error feedback, up to 3 retries), and explains the results in plain language — mirroring how a data analyst actually works.

## Architecture

```
User Query
   │
   ▼
┌─────────────┐     intent = analysis
│ Query Agent │ ──────────────────────┐
└─────────────┘                       │
   │ intent = chit-chat               ▼
   ▼                          ┌───────────────┐
  END                         │ Planner Agent │  breaks request into steps
                              └───────────────┘
                                       │
                                       ▼
                              ┌────────────────┐
                     ┌───────▶│ Codegen Agent  │  writes pandas/matplotlib code
                     │        └────────────────┘
                     │                 │
              error, │                 ▼
              retry<3 │        ┌─────────────────┐
                     └─────────│ Executor Agent  │  runs code, captures output/figures
                              └─────────────────┘
                                       │ success
                                       ▼
                              ┌───────────────┐
                              │  Viz Agent    │
                              └───────────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ Explainer Agent  │  synthesizes plain-language findings
                              └──────────────────┘
                                       │
                                       ▼
                                      END
```

**Agent roles:**

1. **Query Agent** — classifies intent (data analysis vs. general chat), routing the graph accordingly.
2. **Planner Agent** — decomposes the user's question into concrete, executable analysis steps given the dataset schema.
3. **Codegen Agent** — generates pandas/matplotlib/seaborn code scoped to the plan and schema.
4. **Executor Agent** — runs the generated code, captures stdout, figures, and DataFrame mutations.
5. **Self-Correction Loop** — on execution failure, the error is fed back to the Codegen Agent for an automatic fix, up to 3 retries, before the graph proceeds.
6. **Explainer Agent** — turns raw execution output into a clear, natural-language explanation of findings.

## Key Features

| Feature | Description |
|---|---|
| 🔐 Secure Authentication | Firebase-backed sign-up/sign-in with token-based session handling |
| 📂 Smart Data Ingestion | CSV/XLSX upload with automatic schema and health checks |
| 📊 Auto Data Health | Immediate detection of missing values, duplicates, and column-level schema insights |
| 🧹 NL-Driven Data Cleaning | Clean datasets via natural language ("remove duplicates", "fill missing values in X") |
| 📈 Dynamic Visualization | Seaborn/Matplotlib plots generated on demand through chat |
| 🔄 Self-Correcting Execution | Closed-loop retry mechanism recovers from code-generation errors automatically |
| 💾 Stateful Sessions | DataFrame mutations persist across turns, enabling iterative, conversational analysis |

## Tech Stack

- **Orchestration**: LangGraph, LangChain
- **LLM Inference**: Groq (Llama-3.1-8b-instant)
- **Frontend**: Streamlit (custom dark theme)
- **Auth**: Firebase Authentication (REST API)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Getting Started

### Local Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/amulyagavankar20-glitch/Data-Analyst-Ai.git
    cd Data-Analyst-Ai
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables — create a `.env` file in the root directory:

    ```env
    GROQ_API_KEY=your_groq_key
    FIREBASE_API_KEY=your_firebase_key
    ```

4. Run the application:

    ```bash
    streamlit run app.py
    ```

## Usage

1. **Sign up / Log in** via the authentication screen.
2. **Upload** a CSV or XLSX file from the sidebar.
3. **Review** the automatic data health report.
4. **Ask** questions or give instructions in chat — e.g. "show correlation heatmap" or "clean nulls in Age".
5. **Iterate** — the agent retains dataset state across the conversation for progressive analysis.

## Roadmap

- Sandboxed code execution for stronger isolation
- Automated evaluation suite tracking retry success rate and latency
- CI pipeline (lint + test on push)
- Docker-based deployment
