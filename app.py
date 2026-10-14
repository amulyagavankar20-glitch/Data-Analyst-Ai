"""Streamlit UI for authenticated, isolated data analysis."""
import hashlib
import io
import logging
import os
import zipfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
st.set_page_config(page_title="AI Data Analyst", page_icon="A", layout="wide")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Configure it before starting the app.")
    st.stop()

try:
    from langchain_core.messages import AIMessage, HumanMessage
    from src.agents.graph import app as agent_app
    from src.auth import (sign_in_with_email_and_password,
                          sign_up_with_email_and_password, validate_session,
                          initialize_auth_db, revoke_session, AuthUnavailable)
except ImportError:
    logger.exception("Application module import failed")
    st.error("The application could not start because a required component is unavailable.")
    st.stop()


@st.cache_resource(show_spinner=False)
def _initialize_auth_schema():
    initialize_auth_db()


try:
    _initialize_auth_schema()
except AuthUnavailable:
    logger.exception("PostgreSQL authentication is unavailable")
    st.error("Authentication storage is unavailable. Check DATABASE_URL and PostgreSQL connectivity.")
    st.stop()

st.markdown("""<style>
:root {
  --canvas: #0b0f14;
  --panel: #121820;
  --panel-raised: #171f29;
  --line: rgba(173, 190, 210, .14);
  --text: #edf3f8;
  --muted: #9aa8b7;
  --accent: #27c2d5;
}
.stApp { background: radial-gradient(ellipse at 70% -20%, #15303a 0, var(--canvas) 46%); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
main .block-container { max-width: 1160px; padding: 2.5rem 2.5rem 5rem; }
h1, h2, h3 { color: var(--text) !important; letter-spacing: -.035em; }
h1 { font-size: clamp(2rem, 4vw, 3rem) !important; font-weight: 750 !important; }
[data-testid="stSidebar"] { background: #10161e; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text) !important; }
.workspace-brand { display:flex; align-items:center; gap:.7rem; margin: .15rem 0 1.6rem; }
.workspace-mark { display:grid; place-items:center; width:2.35rem; height:2.35rem; border-radius:.75rem;
  background:linear-gradient(135deg,#33d2df,#148ba7); color:#07151b; font-weight:900; font-size:1.2rem; }
.workspace-name { font-size:1rem; font-weight:700; color:var(--text); line-height:1.2; }
.workspace-kicker { font-size:.65rem; letter-spacing:.14em; color:var(--muted); margin-top:.2rem; }
.page-kicker { color:var(--accent); text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; margin:0 0 .35rem; }
.page-subtitle { color:var(--muted); font-size:1.03rem; margin-top:-.45rem; margin-bottom:1.7rem; }
.dataset-status { display:inline-flex; align-items:center; gap:.45rem; border:1px solid var(--line);
  background:rgba(18,24,32,.8); color:#b8c6d3; padding:.35rem .7rem; border-radius:999px;
  font-size:.75rem; margin:.2rem 0 1rem; }
.dataset-dot { width:.45rem; height:.45rem; border-radius:50%; background:#39d39a; }
.stChatMessage { background:rgba(23,31,41,.88); border:1px solid var(--line); border-radius:1rem;
  padding:1.05rem 1.15rem; margin:.8rem 0; }
[data-testid="stChatInput"] textarea { border:1px solid var(--line); border-radius:.85rem;
  background:#121820; color:var(--text); }
[data-testid="stChatInput"] textarea:focus { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }
[data-testid="stFileUploader"] section { background:#121820; border:1px dashed #43515e; border-radius:.9rem; }
[data-testid="stExpander"] { background:rgba(18,24,32,.68); border:1px solid var(--line); border-radius:.8rem; }
[data-testid="stMetric"] { background:var(--panel-raised); border:1px solid var(--line); border-radius:.8rem; padding:.65rem .8rem; }
div.stButton > button, [data-testid="stFormSubmitButton"] button { border-radius:.7rem; font-weight:650; }
div.stButton > button:hover { border-color:var(--accent); color:var(--accent); }
@media (max-width: 700px) { main .block-container { padding:1.5rem 1rem 4rem; } }
</style>""", unsafe_allow_html=True)

if "user" not in st.session_state:
    _, auth_col, _ = st.columns([1, 1.15, 1])
    with auth_col:
        st.markdown('<div class="workspace-brand"><div class="workspace-mark">A</div><div><div class="workspace-name">AI Data Analyst</div><div class="workspace-kicker">PRIVATE DATA WORKSPACE</div></div></div>', unsafe_allow_html=True)
        st.markdown('<p class="page-kicker">Welcome</p>', unsafe_allow_html=True)
        st.title("Sign in to continue")
        st.markdown('<p class="page-subtitle">Your datasets and analysis history stay in your workspace.</p>', unsafe_allow_html=True)
        auth_mode = st.radio("Account", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")
        with st.form("auth_form"):
            email = st.text_input("Email", max_chars=254)
            password = st.text_input("Password", type="password", max_chars=128,
                                     help="Use at least 12 characters.")
            submitted = st.form_submit_button("Continue", use_container_width=True, type="primary")
        if submitted:
            auth_result = (sign_in_with_email_and_password(email, password)
                           if auth_mode == "Login"
                           else sign_up_with_email_and_password(email, password))
            if auth_result.get("session_token"):
                st.session_state["user"] = auth_result
                st.rerun()
            st.error(auth_result.get("error", {}).get("message", "Authentication failed."))
    st.stop()

try:
    validated_user = validate_session(st.session_state["user"])
except AuthUnavailable:
    st.error("Sign-in verification is temporarily unavailable. Please retry in a moment.")
    st.stop()
if not validated_user:
    st.session_state.pop("user", None)
    st.session_state.pop("df", None)
    st.session_state.pop("upload_digest", None)
    st.session_state.pop("analysis_artifacts", None)
    st.session_state.pop("pending_prompt", None)
    st.session_state["messages"] = [AIMessage(content="Upload a dataset and ask me anything about it.")]
    st.warning("Your session expired or could not be verified. Please sign in again.")
    st.rerun()
st.session_state["user"] = validated_user

frame = st.session_state.get("df")
st.markdown('<p class="page-kicker">AI-powered data exploration</p>', unsafe_allow_html=True)
st.title("Make sense of your data.")
st.markdown('<p class="page-subtitle">Ask a question in plain English. Get a clear answer, analysis, and charts.</p>', unsafe_allow_html=True)
if frame is not None:
    st.markdown(f'<div class="dataset-status"><span class="dataset-dot"></span> Dataset ready <span>·</span> {len(frame):,} rows <span>·</span> {len(frame.columns):,} columns</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="dataset-status"><span class="dataset-dot" style="background:#788797"></span> Upload a dataset to get started</div>', unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="Upload a dataset and ask me anything about it.")]
if "analysis_artifacts" not in st.session_state:
    st.session_state["analysis_artifacts"] = {}

# Upload handling is deliberately below the auth gate.
with st.sidebar:
    st.markdown('<div class="workspace-brand"><div class="workspace-mark">A</div><div><div class="workspace-name">AI Data Analyst</div><div class="workspace-kicker">YOUR WORKSPACE</div></div></div>', unsafe_allow_html=True)
    st.markdown("#### Privacy")
    share_values = st.checkbox(
        "Share dataset values with Groq",
        value=False,
        help="Off by default. Turning this on sends up to 3 sample rows and computed results to Groq for analysis summaries.",
    )
    st.markdown("#### Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], key="dataset_upload",
                                     help="CSV or XLSX · maximum 20 MB")
    if uploaded_file is not None:
        raw = uploaded_file.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) > 20 * 1024 * 1024:
            st.error("File is too large. The maximum upload size is 20 MB.")
        elif st.session_state.get("upload_digest") != digest:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    frame = pd.read_csv(io.BytesIO(raw), nrows=500_001)
                else:
                    with zipfile.ZipFile(io.BytesIO(raw)) as workbook:
                        members = workbook.infolist()
                        if (len(members) > 10_000
                                or sum(item.file_size for item in members) > 128 * 1024 * 1024
                                or any(item.file_size > 64 * 1024 * 1024
                                       or (item.compress_size == 0 and item.file_size > 0)
                                       or (item.compress_size and item.file_size / item.compress_size > 200)
                                       for item in members)):
                            raise ValueError("Workbook archive exceeds decompression limits")
                    frame = pd.read_excel(io.BytesIO(raw), nrows=500_001)
                if frame.empty or len(frame.columns) == 0:
                    raise ValueError("Empty dataset")
                if len(frame) > 500_000 or len(frame.columns) > 200:
                    raise ValueError("Dataset dimensions exceed limits")
                if not frame.columns.is_unique:
                    raise ValueError("Duplicate column names")
                if any(str(col).startswith("Unnamed:") for col in frame.columns):
                    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")]
                if frame.empty or len(frame.columns) == 0:
                    raise ValueError("No usable columns")
                st.session_state["df"] = frame
                st.session_state["upload_digest"] = digest
                st.success(f"Loaded {len(frame):,} rows.")
            except Exception:
                logger.exception("Dataset upload processing failed")
                st.error("This file could not be read. Check that it is a valid, non-empty CSV or Excel file.")

    if "df" in st.session_state:
        frame = st.session_state["df"]
        st.markdown("#### Data health")
        metric_cols = st.columns(2)
        metric_cols[0].metric("Rows", f"{len(frame):,}")
        metric_cols[1].metric("Columns", f"{len(frame.columns):,}")
        st.caption(f"{int(frame.isnull().sum().sum()):,} missing values · {int(frame.duplicated().sum()):,} duplicate rows")
        with st.expander("Explore schema"):
            schema_view = pd.DataFrame({
                "Column": frame.columns.astype(str),
                "Type": frame.dtypes.astype(str).values,
                "Missing": frame.isnull().sum().values,
            })
            st.dataframe(schema_view, hide_index=True, use_container_width=True, height=280)

with st.sidebar:
    if st.button("Logout"):
        try:
            revoke_session(st.session_state.get("user"))
        except AuthUnavailable:
            logger.warning("Session could not be revoked because PostgreSQL is unavailable")
        for key in ("user", "df", "upload_digest", "analysis_artifacts", "pending_prompt"):
            st.session_state.pop(key, None)
        st.session_state["messages"] = [AIMessage(content="Upload a dataset and ask me anything about it.")]
        st.rerun()

if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")
    if "df" not in st.session_state:
        st.session_state["messages"].append(
            AIMessage(content="Please upload a dataset first so I can analyze it."))
    else:
        frame = st.session_state["df"]
        schema = (f"Columns: {list(frame.columns)}\nTypes: {frame.dtypes.astype(str).to_dict()}\n"
                  f"Shape: {frame.shape}")
        if share_values:
            schema += f"\nUser opted in to sharing dataset values. Sample rows: {frame.head(3).to_dict()}"
        state = {
            "messages": st.session_state["messages"], "dataset_schema": schema,
            "dataframe": frame, "plan": [], "code": "", "code_output": "",
            "explanation": "", "charts": [], "error": None, "retry_count": 0,
            "retry_history": [], "intent": "", "analysis_failed": False,
            "share_values": share_values,
        }
        try:
            with st.spinner("Analyzing your question..."):
                result = agent_app.invoke(state)
        except Exception:
            logger.exception("Analysis request failed")
            answer = "Analysis could not be completed. Please try again or check the service configuration."
        else:
            answer = result.get("explanation", "Analysis could not be completed.")
            if not result.get("analysis_failed") and result.get("dataframe") is not None:
                st.session_state["df"] = result["dataframe"]
            artifact = {
                "code_output": result.get("code_output", ""),
                "charts": result.get("charts", []),
                "steps": [(item.type, item.content) for item in result.get("messages", [])],
            }
            if artifact["code_output"] or artifact["charts"] or artifact["steps"]:
                st.session_state["analysis_artifacts"][len(st.session_state["messages"])] = artifact
        st.session_state["messages"].append(AIMessage(content=answer))
    st.rerun()

for message_index, message in enumerate(st.session_state["messages"]):
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)
    artifact = st.session_state["analysis_artifacts"].get(message_index)
    if artifact:
        if artifact["code_output"]:
            with st.expander("Analysis result"):
                st.code(artifact["code_output"])
        if artifact["charts"]:
            with st.container(border=True):
                st.markdown("#### Interactive visualizations")
                for chart_index, chart in enumerate(artifact["charts"]):
                    chart_height = chart.layout.height
                    if chart_height is None or chart_height < 320:
                        chart_height = 420
                    else:
                        chart_height = min(chart_height, 620)
                    chart.update_layout(height=chart_height, autosize=True)
                    st.plotly_chart(
                        chart,
                        use_container_width=True,
                        theme="streamlit",
                        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
                        key=f"analysis_chart_{message_index}_{chart_index}",
                    )
        if artifact["steps"]:
            with st.expander("View Analysis Steps"):
                for item_type, item_content in artifact["steps"]:
                    st.caption(f"{item_type}: {item_content}")

if prompt := st.chat_input("What would you like to know?"):
    st.session_state["messages"].append(HumanMessage(content=prompt))
    st.session_state["pending_prompt"] = prompt
    st.rerun()
