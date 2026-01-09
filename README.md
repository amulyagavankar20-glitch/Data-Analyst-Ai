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

An **AI-powered Data Scientist Agent** designed to streamline data analysis, cleaning, and visualization using a multi-agent orchestrated workflow.

## ✨ Key Features

- **Smart Data Loading**: Supports CSV and Excel files with automatic health checks (missing values, duplicates).
- **Automated Data Cleaning**: Clean datasets through natural language instructions.
- **Dynamic Visualization**: Generate insightful plots using Seaborn and Matplotlib.
- **Intelligent Analysis**: Ask complex questions about your data and get reasoned answers.
- **Iterative Problem Solving**: Integrated "Planner-Coder-Executor" loop that self-corrects on failure.
- **Professional Reports**: Generate and download automated analysis reports.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Agent Framework**: [LangGraph](https://www.langchain.com/langgraph) & [LangChain](https://www.langchain.com/)
- **LLM**: [Groq](https://groq.com/) (Llama-3.1-8b-instant)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Report Generation**: ReportLab

## 🏗️ Architecture

The project uses a specialized **LangGraph** workflow to handle data science tasks:

1. **Query Agent**: Detects user intent (analysis vs. chat).
2. **Planner Agent**: Breaks down the request into executable technical steps.
3. **Codegen Agent**: Writes Python code to perform the analysis.
4. **Executor Agent**: Runs the code in a sandbox, capturing outputs and figures.
    - *Self-Correction*: If the code fails, it loops back to the Codegen Agent with the error message for fixing (up to 3 retries).
5. **Explainer Agent**: Synthesizes the execution results into a clear, natural language explanation.

## 🚀 Getting Started

### Local Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/ai-ds-agent.git
    cd ai-ds-agent
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your Groq API key:

    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

4. **Run the application**:

    ```bash
    streamlit run app.py
    ```

### Docker Setup

1. **Build the image**:

    ```bash
    docker build -t ai-ds-agent .
    ```

2. **Run the container**:

    ```bash
    docker run -p 8501:8501 --env-file .env ai-ds-agent
    ```

## 📖 Usage

1. Upload your dataset (CSV/XLSX) via the sidebar.
2. Review the **Data Health** section for initial insights.
3. Type your question or instruction in the chat box (e.g., "Show me the correlation between X and Y" or "Clean the missing values in column Z").
4. The agent will plan, execute code, and display results/plots directly in the chat.
