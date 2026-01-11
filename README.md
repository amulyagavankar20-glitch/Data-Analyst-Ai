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

- **🔐 Secure Authentication**: Integrated Firebase Authentication for secure user sessions.
- **📂 Smart Data Loading**: Supports CSV and Excel files with automatic health checks.
- **📊 Auto Data Health**: Immediate identification of missing values, duplicates, and schema insights upon upload.
- **🧹 Automated Data Cleaning**: Clean datasets through natural language instructions (e.g., "Remove duplicates", "Fill missing values in X").
- **📈 Dynamic Visualization**: Generate insightful plots using Seaborn and Matplotlib via chat.
- **💡 Intelligent Analysis**: Ask complex questions and get reasoned answers with code execution transparency.
- **🔄 Iterative Problem Solving**: Integrated "Planner-Coder-Executor" loop that self-corrects based on execution errors.
- **💾 State Persistence**: Data modifications persist throughout the session for iterative cleaning and analysis.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Custom Dark Theme)
- **Agent Framework**: [LangGraph](https://www.langchain.com/langgraph) & [LangChain](https://www.langchain.com/)
- **LLM**: [Groq](https://groq.com/) (Llama-3.1-8b-instant)
- **Authentication**: [Firebase](https://firebase.google.com/)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 🏗️ Architecture

The project uses a specialized **LangGraph** workflow to handle data science tasks:

1. **Query Agent**: Detects if the user intent is data analysis or general chat.
2. **Planner Agent**: Breaks down the request into executable technical steps.
3. **Codegen Agent**: Writes Python code optimized for the current dataset schema.
4. **Executor Agent**: Runs the code in a sandbox, capturing outputs and figures.
    - *Self-Correction*: If the code fails, it loops back to the Codegen Agent with the error message for fixing (up to 3 retries).
5. **Explainer Agent**: Synthesizes the execution results into a clear, natural language explanation.

## 🚀 Getting Started

### Local Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/amulyagavankar20-glitch/Data-Analyst-Ai.git
    cd Data-Analyst-Ai
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Environment Variables**:
    Create a `.env` file in the root directory:

    ```env
    GROQ_API_KEY=your_groq_key
    FIREBASE_API_KEY=your_firebase_key
    ```

4. **Run the application**:

    ```bash
    streamlit run app.py
    ```

## 📖 Usage

1. **Login/Sign Up**: Create an account or log in via the authentication screen.
2. **Upload Data**: Use the sidebar to upload a CSV or XLSX file.
3. **Check Health**: Review the automatic data health report.
4. **Interact**: Type instructions or questions in the chat (e.g., "Show correlation heatmap" or "Clean nulls in Age").
5. **Refine**: Continue the conversation; the agent remembers previous data changes.
