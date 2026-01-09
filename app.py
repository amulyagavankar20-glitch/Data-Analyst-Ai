import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

# --- 1. PRE-IMPORT CHECK ---
# We check for the API key BEFORE importing the graph to avoid crashes during import
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🔑 **GROQ_API_KEY is missing!**")
    st.markdown("""
    To fix this on Streamlit Cloud:
    1. Go to your **Streamlit Dashboard**.
    2. Click on the **'...'** (three dots) next to your app.
    3. Select **Settings** > **Secrets**.
    4. Add your key in this format:
    ```toml
    GROQ_API_KEY = "your_key_here"
    ```
    """)
    st.info("If you are running locally, ensure your `.env` file contains `GROQ_API_KEY`.")
    st.stop()

# --- 2. IMPORT MODULES ---
import sys
try:
    # Ensure root directory is in sys.path
    root_path = os.path.dirname(os.path.abspath(__file__))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    from src.agents.graph import app as agent_app
    from src.agents.utils import AgentState
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.write(f"Debug Info - Current Path: {os.getcwd()}")
    st.write(f"Debug Info - Python Path: {sys.path}")
    st.write(f"Debug Info - Root Path: {root_path}")
    st.info("💡 Tip: Ensure 'src' folder exists in your GitHub repo root and contains __init__.py.")
    st.stop()

st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")

# ===== Custom CSS (Dark Theme) =====
st.markdown("""
    <style>
    /* Force Dark Theme colors for consistency */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00B4D8 !important; /* Cyan accent */
    }
    
    /* Chat Messages */
    .stChatMessage {
        background-color: #262730;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #363945;
    }
    
    /* User Message distinct color */
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #1E1E1E;
        border: 1px solid #4A4A4A;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #171921;
    }
    
    /* Success/Info messages */
    .stAlert {
        background-color: #262730;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Sidebar: Data Upload =====
with st.sidebar:
    st.title("📂 Data & Config")
    
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state["df"] = df
            st.success(f"Loaded {len(df)} rows!")
            
            # --- Auto EDA (Data Health Check) ---
            st.subheader("📊 Data Health")
            
            # Check Missing
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                st.warning(f"⚠️ {missing_count} missing values found!")
            else:
                st.success("✅ No missing values.")
                
            # Check Duplicates
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                st.warning(f"⚠️ {dup_count} duplicate rows found!")
            else:
                st.success("✅ No duplicates.")
                
            with st.expander("Detailed Schema"):
                st.write(df.dtypes.astype(str))
                st.write(df.describe())
                
        except Exception as e:
            st.error(f"Error loading file: {e}")

# ===== Main Chat Interface =====

st.title("🤖 AI Data Analyst")
st.markdown("**Ask questions about your data in plain English.**")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="Hello! Upload a dataset and ask me anything about it.")
    ]

# Display History
for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# User Input
if prompt := st.chat_input("What would you like to know?"):
    # 1. Append User Message
    st.session_state["messages"].append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Check for Data
    if "df" not in st.session_state:
        response = "Please upload a dataset first so I can analyze it!"
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state["messages"].append(AIMessage(content=response))
    else:
        # 3. Invoke Agent Graph
        df = st.session_state["df"]
        
        # Construct Schema String
        schema_info = f"""
        Columns: {list(df.columns)}
        Types: {df.dtypes.to_dict()}
        Sample: {df.head(3).to_dict()}
        """
        
        # Prepare State
        initial_state = {
            "messages": st.session_state["messages"],
            "dataset_schema": schema_info,
            "dataframe": df,
            "plan": [],
            "code": "",
            "code_output": "",
            "explanation": "",
            "figures": []
        }
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    result = agent_app.invoke(initial_state)
                    
                    final_response = result.get("explanation", "I completed the task but have no explanation.")
                    st.write(final_response)
                    
                    # Store response
                    st.session_state["messages"].append(AIMessage(content=final_response))
                    
                    # Show Figures if any
                    figures = result.get("figures", [])
                    if figures:
                        for fig in figures:
                            st.pyplot(fig)
                            # Ideally store figures in history too, but that's complex with Pickle.
                            # For now, immediate display is fine.
                    
                    # Show intermediate steps (Optional Debugging)
                    with st.expander("View Analysis Steps"):
                        for m in result['messages']:
                            st.caption(f"{m.type}: {m.content}")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
